#!/usr/bin/env python3
"""Regenerate and extract candidates for selected hard private rows."""
from __future__ import annotations

import argparse, csv, json, os, sys, time
from pathlib import Path
from typing import Any
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SOLVE_PROMPT = """You are an expert mathematician. Solve concisely and accurately.

Rules:
- Do not ramble.
- Use at most 25 reasoning steps.
- End with exactly one \\boxed{{}}.
- If options are provided, box only option letter(s).
- If multiple [ANS] blanks exist, box exactly that many comma-separated answers in blank order.
- Never output multiple \\boxed{{}}.
- Do not round unless asked.

Question:
{question}

Options:
{options_if_any}
"""
SOLVE_PROMPTS = [
    ('concise_solve', SOLVE_PROMPT),
    (
        'verify_carefully',
        """You are an expert mathematician. Solve the problem carefully, checking each step before the final answer.

Rules:
- Keep the solution concise.
- Verify arithmetic, units, signs, and requested formatting.
- End with exactly one \boxed{{}}.
- If options are provided, box only option letter(s).
- If multiple [ANS] blanks exist, box exactly that many comma-separated answers in blank order.
- Never output multiple \boxed{{}}.
- Do not round unless asked.

Question:
{question}

Options:
{options_if_any}
""",
    ),
    (
        'alternative_method',
        """You are an expert mathematician. Solve using a reliable method, then check the final requested operation.

Rules:
- Prefer a direct or alternative method that avoids rambling.
- Check whether the problem asks for a remainder, rounded value, option letter, interval, or multiple blanks.
- End with exactly one \boxed{{}}.
- If options are provided, box only option letter(s).
- If multiple [ANS] blanks exist, box exactly that many comma-separated answers in blank order.
- Never output multiple \boxed{{}}.
- Do not round unless asked.

Question:
{question}

Options:
{options_if_any}
""",
    ),
]

EXTRACT_PROMPT = """Extract the final answer from the reasoning.

Rules:
- Output exactly one \\boxed{{}}.
- No explanation.
- If options exist, output only option letter(s).
- If multiple [ANS] blanks exist, output exactly that many comma-separated answers.
- Apply the final requested operation, e.g. remainder/mod/rounding.

Question:
{question}
Options:
{options_if_any}
Reasoning:
{reasoning}
"""
TEMPERATURES = [0.4, 0.5, 0.6, 0.7]
SAMPLES_PER_TEMP = 2
SEEDS = [104729, 104759, 104761, 104773, 104779, 104789, 104801, 104803]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default=str(REPO_ROOT/'checkpoints/merged_rs_sft_public526_110_spiral2_review10_15'))
    ap.add_argument('--tokenizer', default=None)
    ap.add_argument('--hard-rows', type=Path, default=REPO_ROOT/'hard_rows_all.csv')
    ap.add_argument('--private', type=Path, default=REPO_ROOT/'competition-data/private.jsonl')
    ap.add_argument('--output', type=Path, default=REPO_ROOT/'artifacts/private_reasoning_paths/regenerated_candidates.jsonl')
    ap.add_argument('--max-tokens', type=int, default=5000)
    ap.add_argument('--extract-max-tokens', type=int, default=512)
    ap.add_argument('--max-model-len', type=int, default=12288)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-num-seqs', type=int, default=64)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--tensor-parallel-size', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def read_hard_ids(path: Path) -> list[int]:
    if not path.exists():
        raise SystemExit(f'hard rows file not found: {path}')
    ids=[]
    with path.open(encoding='utf-8', newline='') as f:
        sample=f.read(4096); f.seek(0)
        has_header=csv.Sniffer().has_header(sample) if sample.strip() else False
        if has_header:
            reader=csv.DictReader(f)
            for row in reader:
                val=None
                for key in ('id','qid','question_id','row_id'):
                    if key in row and str(row[key]).strip(): val=row[key]; break
                if val is None:
                    val=next((v for v in row.values() if str(v).strip().isdigit()), None)
                if val is not None: ids.append(int(val))
        else:
            reader=csv.reader(f)
            for row in reader:
                if row and str(row[0]).strip().isdigit(): ids.append(int(row[0]))
    seen=set(); out=[]
    for x in ids:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


def read_private(path: Path) -> dict[int, dict[str, Any]]:
    rows={}
    with path.open(encoding='utf-8') as f:
        for fallback,line in enumerate(f):
            if not line.strip(): continue
            row=json.loads(line)
            qid=int(row.get('id', row.get('question_id', fallback)))
            rows[qid]=row
    return rows


def options_text(row: dict[str, Any]) -> str:
    opts=row.get('options')
    if not isinstance(opts, list) or not opts: return ''
    return '\n'.join(f'{chr(ord("A")+i)}. {opt}' for i,opt in enumerate(opts))


def render_prompt(tokenizer: Any, user_prompt: str, system: str = 'You are an expert mathematician.') -> str:
    msgs=[{'role':'system','content':system},{'role':'user','content':user_prompt}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f'{system}\n\nUser:\n{user_prompt}\n\nAssistant:\n'


def init_vllm(args):
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    from transformers import AutoTokenizer
    from vllm import LLM
    model_path=Path(args.model)
    model=str(model_path.resolve()) if model_path.exists() else args.model
    tok_path=args.tokenizer or model
    tok=AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token is not None: tok.pad_token=tok.eos_token
    llm=LLM(model=model, tokenizer=tok_path, trust_remote_code=True, dtype=args.dtype,
            max_model_len=args.max_model_len, tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs,
            enforce_eager=False, generation_config='vllm')
    return llm,tok


def sampling(n, temp, max_tokens, seed):
    from vllm import SamplingParams
    return SamplingParams(n=n, temperature=temp, top_p=1.0, min_p=0.05 if temp>0 else 0.0,
                          max_tokens=max_tokens, seed=seed, stop=['<|im_end|>','<|endoftext|>'])


def completed_keys(path: Path) -> set[tuple[int,float,int,int]]:
    done=set()
    if not path.exists(): return done
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                r=json.loads(line)
                done.add((int(r['question_id']), float(r['temperature']), int(r['seed']), int(r['sample_index'])))
            except Exception: pass
    return done


def sample_plan() -> list[dict[str, Any]]:
    plan=[]
    sample_index=0
    for temp in TEMPERATURES:
        for rep in range(SAMPLES_PER_TEMP):
            prompt_name, prompt_template = SOLVE_PROMPTS[sample_index % len(SOLVE_PROMPTS)]
            plan.append({
                'temperature': temp,
                'seed': SEEDS[sample_index % len(SEEDS)],
                'sample_index': sample_index,
                'prompt_name': prompt_name,
                'prompt_template': prompt_template,
            })
            sample_index += 1
    return plan


def main():
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(line_buffering=True)
    args=parse_args(); started=time.time()
    hard_ids=read_hard_ids(args.hard_rows)
    if args.limit: hard_ids=hard_ids[:args.limit]
    private=read_private(args.private)
    missing=[x for x in hard_ids if x not in private]
    if missing: raise SystemExit(f'{len(missing)} hard ids missing from private.jsonl, first={missing[:10]}')
    done=completed_keys(args.output) if args.resume else set()
    print(f'[regen] hard_rows={len(hard_ids)} completed_candidates={len(done)} output={args.output}', flush=True)
    llm,tok=init_vllm(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan=sample_plan()
    with args.output.open('a',encoding='utf-8') as out_f:
        for spec in plan:
            temp=float(spec['temperature']); seed=int(spec['seed']); sample_idx=int(spec['sample_index'])
            jobs=[]
            for qid in hard_ids:
                key=(qid,temp,seed,sample_idx)
                if key in done: continue
                row=private[qid]
                prompt=spec['prompt_template'].format(question=row.get('question','').rstrip(), options_if_any=options_text(row))
                rendered=render_prompt(tok, prompt)
                jobs.append((qid,row,prompt,rendered,spec['prompt_name'],sample_idx,seed,temp))
            for i in tqdm(range(0,len(jobs),args.batch_size), desc=f"solve temp={temp} sample={sample_idx} prompt={spec['prompt_name']}"):
                batch=jobs[i:i+args.batch_size]
                prompts=[x[3] for x in batch]
                outs=llm.generate(prompts, sampling(1,temp,args.max_tokens,seed))
                extract_jobs=[]
                for (qid,row,prompt,_,prompt_name,sample_idx,seed,temp),out in zip(batch,outs):
                    reasoning=out.outputs[0].text.strip() if out.outputs else ''
                    ep=EXTRACT_PROMPT.format(question=row.get('question','').rstrip(), options_if_any=options_text(row), reasoning=reasoning)
                    extract_jobs.append({'qid':qid,'row':row,'solve_prompt':prompt,'prompt_name':prompt_name,'reasoning':reasoning,'temperature':temp,'seed':seed,'sample_index':sample_idx,'extract_prompt':ep})
                for j in range(0,len(extract_jobs),args.batch_size):
                    eb=extract_jobs[j:j+args.batch_size]
                    eprompts=[render_prompt(tok, x['extract_prompt'], system='You extract final answers exactly.') + '\\boxed{' for x in eb]
                    eouts=llm.generate(eprompts, sampling(1,0.0,args.extract_max_tokens,99991))
                    for item,eout in zip(eb,eouts):
                        extracted_suffix=eout.outputs[0].text.strip() if eout.outputs else ''
                        extraction='\\boxed{' + extracted_suffix
                        rec={
                            'question_id':item['qid'], 'temperature':item['temperature'], 'seed':item['seed'], 'sample_index':item['sample_index'],
                            'prompt_name':item['prompt_name'], 'reasoning':item['reasoning'], 'extraction':extraction, 'prompt':item['solve_prompt'],
                        }
                        out_f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                    out_f.flush()
    print(f'[regen] done elapsed_min={(time.time()-started)/60:.2f} output={args.output}', flush=True)

if __name__=='__main__': main()
