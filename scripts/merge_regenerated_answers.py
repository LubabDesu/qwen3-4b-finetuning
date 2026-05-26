#!/usr/bin/env python3
"""Vote over regenerated hard-row candidates and merge valid winners into a submission."""
from __future__ import annotations

import argparse, csv, json, math, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT/'scripts'))
from judger import Judger
BOX_RE=re.compile(r'\\boxed\s*\{')
BAD={'','none','null','n/a','na'}


def parse_args():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hard-rows',type=Path,default=REPO_ROOT/'hard_rows_all.csv')
    ap.add_argument('--private',type=Path,default=REPO_ROOT/'competition-data/private.jsonl')
    ap.add_argument('--draft',type=Path,default=REPO_ROOT/'artifacts/private_reasoning_paths/submission_draft_8_reasoning_restored.csv')
    ap.add_argument('--candidates',type=Path,default=REPO_ROOT/'artifacts/private_reasoning_paths/regenerated_candidates.jsonl')
    ap.add_argument('--output',type=Path,default=REPO_ROOT/'artifacts/private_reasoning_paths/submission_draft_9.csv')
    ap.add_argument('--summary',type=Path,default=REPO_ROOT/'artifacts/private_reasoning_paths/regenerated_vote_summary.csv')
    ap.add_argument('--failed',type=Path,default=REPO_ROOT/'artifacts/private_reasoning_paths/failed_regen_rows.csv')
    ap.add_argument('--strong-votes',type=int,default=3,help='Minimum regenerated candidates agreeing before replacing a valid old answer.')
    ap.add_argument('--only-ids',default='',help='Comma-separated IDs eligible for replacement. Empty means no allowlist.')
    ap.add_argument('--block-ids',default='',help='Comma-separated IDs never eligible for replacement.')
    return ap.parse_args()


def read_hard_ids(path):
    ids=[]
    if not path.exists(): raise SystemExit(f'hard rows file not found: {path}')
    with path.open(encoding='utf-8',newline='') as f:
        sample=f.read(4096); f.seek(0)
        has_header=csv.Sniffer().has_header(sample) if sample.strip() else False
        if has_header:
            for row in csv.DictReader(f):
                val=None
                for k in ('id','qid','question_id','row_id'):
                    if k in row and str(row[k]).strip(): val=row[k]; break
                if val is None: val=next((v for v in row.values() if str(v).strip().isdigit()), None)
                if val is not None: ids.append(int(val))
        else:
            for row in csv.reader(f):
                if row and str(row[0]).strip().isdigit(): ids.append(int(row[0]))
    return list(dict.fromkeys(ids))


def read_private(path):
    rows={}
    with path.open(encoding='utf-8') as f:
        for fallback,line in enumerate(f):
            if line.strip():
                r=json.loads(line); rows[int(r.get('id',r.get('question_id',fallback)))] = r
    return rows


def norm_text(s):
    s=unicodedata.normalize('NFKC',str(s or '')).strip()
    s=s.replace('\\boxed{','').replace('boxed{','')
    s=re.sub(r'[{}$]','',s)
    s=re.sub(r'\s+',' ',s).strip(' .,:;')
    return s


def split_top_level(s):
    parts=[]; buf=[]; depth=0
    for i,ch in enumerate(s or ''):
        if ch in '([{' and (i==0 or s[i-1] != '\\'): depth+=1
        elif ch in ')]}' and (i==0 or s[i-1] != '\\') and depth>0: depth-=1
        if ch in ',;' and depth==0:
            p=''.join(buf).strip()
            if p: parts.append(p)
            buf=[]
        else: buf.append(ch)
    p=''.join(buf).strip()
    if p: parts.append(p)
    return parts


def parse_float(s):
    t=str(s).strip().replace('%','').replace(',','')
    try: return float(t)
    except Exception: return None


def canon_part(p):
    p=norm_text(p).replace('%','')
    x=parse_float(p)
    if x is not None and math.isfinite(x): return f'{x:.10g}'
    return re.sub(r'\s*,\s*', ',', p.lower())


def option_value_to_letter(ans, opts):
    na=norm_text(ans).lower().replace('\\','')
    for i,opt in enumerate(opts or []):
        no=norm_text(opt).lower().replace('\\','')
        if na==no: return chr(ord('A')+i)
    return None


def has_corrupt_latex(s):
    t=str(s or '')
    if '\x08oxed' in t or ('boxed{' in t and '\\boxed{' not in t):
        return True
    if t.count('{') != t.count('}'):
        return True
    if re.search(r'\\(?:frac|dfrac|tfrac)(?!\s*\{)', t):
        return True
    if re.search(r'\\(?:frac|dfrac|tfrac)\s*\{[^{}]+\}\s*\{[^{}]+\}\s*[-+]?\d', t):
        return True
    if re.search(r'\\sqrt(?!\s*(?:\{|\[))', t):
        return True
    if re.search(r'(?<!\\)\\(?![A-Za-z{}\[\](),.+\-*/^_\s])', t):
        return True
    return False


def question_asks_explanation(row):
    q=str((row or {}).get('question','')).lower()
    return any(p in q for p in ['explain', 'justify', 'show your work', 'why'])


def looks_like_explanation(s, row=None):
    if question_asks_explanation(row):
        return False
    raw=str(s or '')
    t=norm_text(raw)
    if len(t) > 120 or '\n' in raw:
        return True
    if re.search(r'\b(because|since|therefore|p-value|option|we choose|the answer|we get|is equal to)\b', t, re.I):
        return True
    return len(re.findall(r'[A-Za-z]{3,}', t)) > 5




def parse_id_set(s):
    out=set()
    for part in str(s or '').replace(' ', '').split(','):
        if part:
            out.add(int(part))
    return out


def allows_multi_for_single_blank(question):
    q=str(question or '').lower()
    patterns=[
        'separate multiple answers by commas',
        'more than one answer',
        'find all',
        'list all',
        'all solutions',
        'all values',
        'select all',
        'which statements',
        'natural numbers',
        'integers between',
        'solutions to',
        'solve ',
        'solve for',
        'solutions of',
        'list the elements',
        'list the',
        'values of',
        'roots',
    ]
    return any(p in q for p in patterns)


def is_multi_value_question(row):
    q=(row.get('question') or '') if row else ''
    return q.count('[ANS]') > 1 or allows_multi_for_single_blank(q)


def answer_component_count(ans, row):
    text=norm_text(ans)
    opts=(row or {}).get('options')
    if isinstance(opts, list) and opts and re.fullmatch(r'[A-Ja-j]+', text):
        return len(text)
    return len(split_top_level(text)) if text else 0


def collapsed_multi_value(old_key, new_key, row):
    return (
        old_key is not None
        and new_key is not None
        and is_multi_value_question(row)
        and answer_component_count(old_key, row) > 1
        and answer_component_count(new_key, row) == 1
    )


def normalize_option_component(part, opts):
    letters=re.findall(r'(?<![A-Za-z])([A-J])(?![A-Za-z])', norm_text(part), re.I)
    if letters:
        return letters[0].upper()
    return option_value_to_letter(part, opts)

def normalize_answer(ans, row):
    ans=norm_text(ans)
    if ans.lower() in BAD: return None
    opts=row.get('options')
    question=row.get('question') or ''
    n=question.count('[ANS]')
    parts=split_top_level(ans)
    if isinstance(opts,list) and opts:
        if n>1:
            if len(parts)!=n: return None
            mapped=[normalize_option_component(p, opts) for p in parts]
            if any(not m for m in mapped): return None
            return ', '.join(mapped)
        tokens=re.findall(r'(?<![A-Za-z])([A-J])(?![A-Za-z])', ans, re.I)
        if tokens:
            letters=[]
            for t in tokens:
                u=t.upper()
                if u not in letters: letters.append(u)
            return ''.join(letters) if letters else None
        mapped=option_value_to_letter(ans, opts)
        return mapped
    if n>1:
        if len(parts)!=n: return None
        return ', '.join(canon_part(p) for p in parts)
    if len(parts)>1 and not allows_multi_for_single_blank(question):
        return None
    if len(parts)>1:
        return ', '.join(canon_part(p) for p in parts)
    return canon_part(ans)


def strict_normalize_answer(ans, row):
    if not str(ans or '').strip():
        return None
    if has_corrupt_latex(ans) or looks_like_explanation(ans, row):
        return None
    key=normalize_answer(ans, row)
    if not key or looks_like_explanation(key, row):
        return None
    return key




def clean_raw_answer(ans):
    t=str(ans or '').strip()
    t=re.sub(r'^\\boxed\s*\{(.*)\}$', r'\1', t, flags=re.S).strip()
    t=re.sub(r'\s*,\s*', ', ', t)
    return t


def format_final_answer(win):
    raw=clean_raw_answer(win.get('answer') or '')
    key=win.get('key') or raw
    parts=split_top_level(str(key or ''))
    if parts and all(re.fullmatch(r'[A-Ja-j]', norm_text(p)) for p in parts):
        return ', '.join(norm_text(p).upper() for p in parts)
    if re.fullmatch(r'[A-Ja-j]+', norm_text(key)):
        return norm_text(key).upper()
    return raw or key

def extract_answer(judger, text):
    try:
        a=judger.extract_ans(text) or ''
    except Exception:
        a=''
    if not a or a==text:
        try: a=judger.extract_boxed_answer(text) or ''
        except Exception: pass
    return a


def wait_count(s): return len(re.findall(r'\b(wait|hold on|maybe|not sure|confus|actually)\b', s or '', re.I))

def main():
    args=parse_args(); judger=Judger()
    hard=set(read_hard_ids(args.hard_rows)); private=read_private(args.private)
    only_ids=parse_id_set(args.only_ids)
    block_ids=parse_id_set(args.block_ids)
    draft_rows=[]; old_by_id={}
    with args.draft.open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            qid=int(row['id']); draft_rows.append(row); old_by_id[qid]=row['response'] or ''
    cands=defaultdict(list)
    for qid in hard:
        if qid in old_by_id:
            old_ans=extract_answer(judger, old_by_id[qid])
            key=strict_normalize_answer(old_ans, private[qid])
            if key:
                cands[qid].append({'source':'old','key':key,'answer':old_ans,'reasoning':old_by_id[qid],'temperature':99.0,'waits':wait_count(old_by_id[qid]),'valid':True})
    with args.candidates.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line); qid=int(r['question_id'])
            if qid not in hard or qid not in private: continue
            raw_ans=extract_answer(judger, r.get('extraction',''))
            key=strict_normalize_answer(raw_ans, private[qid])
            if not key: continue
            cands[qid].append({'source':'regen','key':key,'answer':raw_ans,'reasoning':r.get('reasoning',''),'temperature':float(r.get('temperature',99)), 'seed':r.get('seed'), 'sample_index':r.get('sample_index'), 'waits':wait_count(r.get('reasoning','')), 'valid':True})
    winners={}; summaries=[]; failed=[]
    for qid in sorted(hard):
        vals=cands.get(qid,[])
        if not vals:
            failed.append({'id':qid,'reason':'no_valid_candidate'}); continue
        old_ans=extract_answer(judger, old_by_id.get(qid,''))
        old_key=strict_normalize_answer(old_ans, private[qid]) if qid in private else None
        old_invalid=old_key is None
        counts=Counter(v['key'] for v in vals)
        top_count=max(counts.values())
        top_keys=[k for k,v in counts.items() if v==top_count]
        pool=[v for v in vals if v['key'] in top_keys]
        pool.sort(key=lambda v:(v['waits'], len(v.get('reasoning') or ''), v['temperature']))
        win=pool[0]
        regen_votes=sum(1 for v in vals if v['source']=='regen' and v['key']==win['key'])
        strong_agreement=regen_votes >= args.strong_votes
        id_allowed=(not only_ids or qid in only_ids) and qid not in block_ids
        collapse=collapsed_multi_value(old_key, win['key'], private[qid])
        accepted=id_allowed and win['source']=='regen' and (old_invalid or strong_agreement) and not collapse
        if accepted:
            winners[qid]=win
        else:
            if not id_allowed:
                reason='blocked_or_not_allowlisted'
            elif collapse:
                reason='collapsed_multi_value'
            else:
                reason='old_valid_and_no_strong_regen_vote' if win['source']=='regen' else 'old_answer_won'
            failed.append({'id':qid,'reason':reason})
        summaries.append({'id':qid,'winner':win['key'],'winner_source':win['source'],'accepted':accepted,'old_invalid':old_invalid,'collapsed_multi_value':collapse,'votes':top_count,'regen_votes':regen_votes,'num_valid_candidates':len(vals),'vote_counts':json.dumps(counts,sort_keys=True),'temperature':win.get('temperature'),'waits':win.get('waits')})
    out_rows=[]
    for row in draft_rows:
        qid=int(row['id']); resp=row['response'] or ''
        if qid in winners:
            ans=format_final_answer(winners[qid])
            resp=resp.rstrip()+f"\n\nFinal answer: \\boxed{{{ans}}}"
        out_rows.append({'id':str(qid),'response':resp})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','response'],quoting=csv.QUOTE_ALL); w.writeheader(); w.writerows(out_rows)
    with args.summary.open('w',encoding='utf-8',newline='') as f:
        fields=['id','winner','winner_source','accepted','old_invalid','collapsed_multi_value','votes','regen_votes','num_valid_candidates','vote_counts','temperature','waits']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summaries)
    with args.failed.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','reason']); w.writeheader(); w.writerows(failed)
    print(json.dumps({'hard_rows':len(hard),'accepted_regen_replacements':len(winners),'failed_or_unchanged':len(failed),'output':str(args.output),'summary':str(args.summary),'failed_file':str(args.failed)},indent=2))

if __name__=='__main__': main()
