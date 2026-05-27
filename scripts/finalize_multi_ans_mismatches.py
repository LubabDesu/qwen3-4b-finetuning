#!/usr/bin/env python3
"""Redo draft_5 multi-[ANS] rows whose boxed answer count does not match blank count."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
BOX_RE = re.compile(r"\\boxed\s*\{")
SYSTEM_PROMPT = """You are an answer extraction assistant. Use the provided question and prior reasoning to extract the final answers. Output exactly one boxed answer and nothing else."""
TEMPLATE = """Question:
{question}

This question has exactly {n_ans} [ANS] blanks. The final boxed answer must contain exactly {n_ans} comma-separated answers in blank order.

Extract the appropriate {n_ans} answers from the prior reasoning below.

Rules:
- Output exactly one \\boxed{{}}.
- Put exactly {n_ans} answers inside the box, comma-separated, in the same order as the [ANS] blanks.
- Do not explain.
- Do not include labels like (a), (b), x=, y= unless the answer itself requires it.
- If the prior reasoning is inconsistent, use the question and the most committed final values from the reasoning.

Prior reasoning:
{reasoning}
"""
STRICT_TEMPLATE = """Question:
{question}

There are exactly {n_ans} [ANS] blanks in this question.

Your task is slot filling:
- Slot 1 is the answer to the first [ANS].
- Slot {n_ans} is the answer to the last [ANS].

Use the prior reasoning only to identify the slot values.

Hard rules:
- Output exactly {n_ans} answers.
- Separate answers with exactly {comma_count} commas.
- Do not output rejected candidates.
- Do not output intermediate values.
- Do not include more than {n_ans} comma-separated fields.
- Do not include fewer than {n_ans} comma-separated fields.
- Do not output labels like (a), (b), x=, y= unless the blank itself requires it.
- Do not explain.
- If you are uncertain, choose the most likely value for each slot, but still output exactly {n_ans} answers.

Prior reasoning:
{reasoning}

Now output exactly {n_ans} answers in one box.
"""
DEFAULT_GENERATION_FILES = [
    REPO_ROOT / "artifacts/private_reasoning_paths/rs_sft_public526_110_n8_6000.jsonl",
    REPO_ROOT / "artifacts/private_reasoning_paths/rs_sft_public526_110_zero_box_recovery_n4_8000.jsonl",
    REPO_ROOT / "artifacts/private_reasoning_paths/rs_sft_public526_110_still_zero_box_inject80_n4_8000.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(REPO_ROOT/'checkpoints/merged_rs_sft_public526_110_spiral2_review10_15'))
    parser.add_argument('--tokenizer', default=None)
    parser.add_argument('--questions', type=Path, default=REPO_ROOT/'competition-data/private.jsonl')
    parser.add_argument('--submission', type=Path, default=REPO_ROOT/'artifacts/private_reasoning_paths/submission_draft_5.csv')
    parser.add_argument('--audit', type=Path, default=REPO_ROOT/'artifacts/private_reasoning_paths/submission_draft_5_multi_ans_mismatch_audit.jsonl')
    parser.add_argument('--generation-file', type=Path, action='append', default=None)
    parser.add_argument('--output', type=Path, default=REPO_ROOT/'artifacts/private_reasoning_paths/submission_draft_5_multi_ans_redo_outputs.jsonl')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--min-p', type=float, default=0.0)
    parser.add_argument('--max-tokens', type=int, default=160)
    parser.add_argument('--max-model-len', type=int, default=12288)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--max-num-seqs', type=int, default=64)
    parser.add_argument('--reasoning-tail-chars', type=int, default=10000)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--strict-slot-prompt', action='store_true', help='Use stricter slot-filling/count prompt for stubborn rows.')
    parser.add_argument('--only-qids', default=None, help='Comma-separated question IDs to process from the audit file.')
    return parser.parse_args()


def read_questions(path: Path) -> dict[int, dict[str, Any]]:
    out={}
    with path.open(encoding='utf-8') as f:
        for fallback,line in enumerate(f):
            if not line.strip(): continue
            row=json.loads(line)
            qid=int(row.get('id', row.get('question_id', fallback)))
            out[qid]=row
    return out


def question_text(row: dict[str, Any]) -> str:
    text=(row.get('question') or '').rstrip()
    opts=row.get('options')
    if isinstance(opts, list) and opts:
        lines=['\nOptions:']
        for i,opt in enumerate(opts):
            lines.append(f"{chr(ord('A')+i)}. {opt}")
        text += '\n' + '\n'.join(lines)
    return text


def read_submission(path: Path) -> dict[int, str]:
    out={}
    with path.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            out[int(row['id'])]=row['response'] or ''
    return out


def read_completed(path: Path) -> set[int]:
    done=set()
    if not path.exists(): return done
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try: done.add(int(json.loads(line)['question_id']))
            except Exception: pass
    return done


def load_raw_generations(paths: list[Path]) -> dict[int, list[str]]:
    out={}
    for p in paths:
        if not p.exists(): continue
        with p.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                row=json.loads(line); qid=int(row['question_id'])
                out.setdefault(qid, []).extend(row.get('generations') or [])
    return out


def choose_reasoning(qid: int, current: str, raw: dict[int, list[str]], n_ans: int) -> tuple[str, str]:
    candidates=[]
    if current:
        candidates.append(('submission', current))
    for gen in raw.get(qid, []):
        candidates.append(('raw_generation', gen or ''))
    def score(item):
        source,text=item
        s=0
        if len(text) > 1000: s += 20
        if BOX_RE.search(text): s += 5
        # Prefer traces that mention final/answer and have commas or enough numeric/letter values near the tail.
        tail=text[-3000:]
        s += 5 * len(re.findall(r'final answer|answers? (?:is|are)|therefore|thus', tail, re.I))
        s += min(15, tail.count(','))
        vals=len(re.findall(r'-?\d+(?:\.\d+)?|\b[A-J]\b', tail))
        s += min(15, vals)
        if source == 'submission' and len(text) < 500: s -= 15
        return s
    if not candidates:
        return '', 'missing'
    source,text=max(candidates, key=score)
    return text, source


def render_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':user_prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{SYSTEM_PROMPT}\n\nUser:\n{user_prompt}\n\nAssistant:\n"


def init_vllm(args: argparse.Namespace) -> tuple[Any, Any]:
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    from transformers import AutoTokenizer
    from vllm import LLM
    model_path=Path(args.model)
    model=str(model_path.resolve()) if model_path.exists() else args.model
    tokenizer_path=args.tokenizer or model
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token=tokenizer.eos_token
    llm=LLM(model=model, tokenizer=tokenizer_path, trust_remote_code=True, dtype=args.dtype, max_model_len=args.max_model_len, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs, enforce_eager=False, generation_config='vllm')
    return llm, tokenizer


def build_sampling(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams
    return SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, min_p=args.min_p, max_tokens=args.max_tokens, stop=['<|im_end|>','<|endoftext|>'])


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(line_buffering=True)
    args=parse_args()
    questions=read_questions(args.questions)
    submission=read_submission(args.submission)
    raw=load_raw_generations(args.generation_file or DEFAULT_GENERATION_FILES)
    done=read_completed(args.output) if args.resume else set()
    only_qids = None
    if args.only_qids:
        only_qids = {int(x.strip()) for x in args.only_qids.split(',') if x.strip()}
    items=[]
    with args.audit.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row=json.loads(line); qid=int(row['qid'])
            if only_qids is not None and qid not in only_qids: continue
            if qid in done: continue
            n_ans=int(row['n_ans'])
            reasoning,source=choose_reasoning(qid, submission.get(qid,''), raw, n_ans)
            template = STRICT_TEMPLATE if args.strict_slot_prompt else TEMPLATE
            prompt=template.format(
                question=question_text(questions[qid]),
                n_ans=n_ans,
                comma_count=max(n_ans - 1, 0),
                reasoning=reasoning[-args.reasoning_tail_chars:],
            )
            items.append({'question_id':qid, 'n_ans':n_ans, 'source':source, 'source_reasoning_chars':len(reasoning), 'prompt':prompt})
    print(f'[multi_redo] pending_items={len(items)} output={args.output}', flush=True)
    if not items: return
    llm,tokenizer=init_vllm(args)
    sampling=build_sampling(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a', encoding='utf-8') as out_f, tqdm(total=len(items), desc='multi_redo') as pbar:
        for i in range(0,len(items),args.batch_size):
            batch=items[i:i+args.batch_size]
            prompts=[render_prompt(tokenizer,item['prompt']) + '\n\\boxed{' for item in batch]
            outputs=llm.generate(prompts, sampling)
            for item,output in zip(batch,outputs):
                text=output.outputs[0].text.strip() if output.outputs else ''
                final='\\boxed{' + text
                if not final.rstrip().endswith('}'):
                    final=final.rstrip() + '}'
                out_f.write(json.dumps({**{k:item[k] for k in ('question_id','n_ans','source','source_reasoning_chars')}, 'finalizer_generations':[final]}, ensure_ascii=False)+'\n')
            out_f.flush(); pbar.update(len(batch))
    print(f'[multi_redo] done output={args.output}', flush=True)

if __name__ == '__main__':
    main()
