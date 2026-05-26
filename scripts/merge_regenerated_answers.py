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
    if re.search(r'\\sqrt(?!\s*(?:\{|\[))', t):
        return True
    return False


def looks_like_explanation(s):
    t=norm_text(s)
    if len(t) > 120 or '\n' in str(s or ''):
        return True
    if re.search(r'\b(the answer|therefore|because|since|we get|is equal to|option)\b', t, re.I):
        return True
    return len(re.findall(r'[A-Za-z]{3,}', t)) > 5


def normalize_answer(ans, row):
    ans=norm_text(ans)
    if ans.lower() in BAD: return None
    opts=row.get('options')
    if isinstance(opts,list) and opts:
        tokens=re.findall(r'(?<![A-Za-z])([A-J])(?![A-Za-z])', ans, re.I)
        if tokens:
            letters=[]
            for t in tokens:
                u=t.upper()
                if u not in letters: letters.append(u)
            return ''.join(letters) if letters else None
        mapped=option_value_to_letter(ans, opts)
        return mapped
    n=(row.get('question') or '').count('[ANS]')
    if n>1:
        parts=split_top_level(ans)
        if len(parts)!=n: return None
        return ', '.join(canon_part(p) for p in parts)
    parts=split_top_level(ans)
    if len(parts)>1: return None
    return canon_part(ans)


def strict_normalize_answer(ans, row):
    if not str(ans or '').strip():
        return None
    if has_corrupt_latex(ans) or looks_like_explanation(ans):
        return None
    key=normalize_answer(ans, row)
    if not key or looks_like_explanation(key):
        return None
    return key


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
        accepted=win['source']=='regen' and (old_invalid or strong_agreement)
        if accepted:
            winners[qid]=win
        else:
            failed.append({'id':qid,'reason':'old_valid_and_no_strong_regen_vote' if win['source']=='regen' else 'old_answer_won'})
        summaries.append({'id':qid,'winner':win['key'],'winner_source':win['source'],'accepted':accepted,'old_invalid':old_invalid,'votes':top_count,'regen_votes':regen_votes,'num_valid_candidates':len(vals),'vote_counts':json.dumps(counts,sort_keys=True),'temperature':win.get('temperature'),'waits':win.get('waits')})
    out_rows=[]
    for row in draft_rows:
        qid=int(row['id']); resp=row['response'] or ''
        if qid in winners:
            ans=winners[qid]['key']
            resp=resp.rstrip()+f"\n\nFinal answer: \\boxed{{{ans}}}"
        out_rows.append({'id':str(qid),'response':resp})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','response'],quoting=csv.QUOTE_ALL); w.writeheader(); w.writerows(out_rows)
    with args.summary.open('w',encoding='utf-8',newline='') as f:
        fields=['id','winner','winner_source','accepted','old_invalid','votes','regen_votes','num_valid_candidates','vote_counts','temperature','waits']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summaries)
    with args.failed.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','reason']); w.writeheader(); w.writerows(failed)
    print(json.dumps({'hard_rows':len(hard),'accepted_regen_replacements':len(winners),'failed_or_unchanged':len(failed),'output':str(args.output),'summary':str(args.summary),'failed_file':str(args.failed)},indent=2))

if __name__=='__main__': main()
