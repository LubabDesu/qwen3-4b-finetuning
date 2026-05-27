#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT/'scripts'))
from judger import Judger

BOX_START_RE=re.compile(r'\\boxed\s*\{')


def find_matching_brace(text: str, open_idx: int) -> int | None:
    depth=0
    i=open_idx
    while i < len(text):
        ch=text[i]
        if ch=='{' and (i==0 or text[i-1] != '\\'):
            depth+=1
        elif ch=='}' and (i==0 or text[i-1] != '\\'):
            depth-=1
            if depth==0:
                return i
        i+=1
    return None


def extract_all_boxed(text: str) -> list[tuple[int,int,str]]:
    out=[]
    for m in BOX_START_RE.finditer(text):
        open_idx=text.find('{', m.start())
        if open_idx < 0:
            continue
        close_idx=find_matching_brace(text, open_idx)
        if close_idx is None:
            continue
        out.append((m.start(), close_idx+1, text[open_idx+1:close_idx]))
    return out


def strip_all_boxed(text: str) -> tuple[str,int]:
    boxes=extract_all_boxed(text)
    if not boxes:
        return text, 0
    parts=[]
    last=0
    for start,end,_ in boxes:
        parts.append(text[last:start])
        last=end
    parts.append(text[last:])
    cleaned=''.join(parts)
    leftover=BOX_START_RE.search(cleaned)
    if leftover:
        cleaned=cleaned[:leftover.start()].rstrip()
    cleaned=re.sub(r'(?im)^\s*Final answer\s*:\s*$', '', cleaned)
    cleaned=re.sub(r'(?im)^\s*(?:Answer|Final)\s*:\s*$', '', cleaned)
    cleaned=re.sub(r'\n{3,}', '\n\n', cleaned).rstrip()
    return cleaned, len(boxes)


def last_boxed_answer(text: str) -> str | None:
    boxes=extract_all_boxed(text)
    return boxes[-1][2].strip() if boxes else None


def has_appended_final(text: str) -> bool:
    return re.search(r'Final answer\s*:\s*\\boxed\s*\{', text[-1000:], re.I) is not None


def intended_answer(judger: Judger, response: str) -> str:
    if has_appended_final(response):
        ans=last_boxed_answer(response)
        if ans:
            return judger.normalize_answer(ans.strip())
    extracted=judger.extract_ans(response)
    if extracted and extracted != response:
        return extracted.strip()
    ans=last_boxed_answer(response)
    if ans:
        return judger.normalize_answer(ans.strip())
    return extracted.strip()


def canonicalize_response(judger: Judger, response: str) -> tuple[str,str,int,bool]:
    answer=intended_answer(judger, response)
    cleaned,n_boxes=strip_all_boxed(response)
    final=(cleaned.rstrip()+f"\n\nFinal answer: \\boxed{{{answer}}}").lstrip()
    got=judger.extract_ans(final).strip()
    ok=(got==answer.strip())
    return final, answer, n_boxes, ok


def main():
    ap=argparse.ArgumentParser(description='Strip all existing boxed answers and append exactly one canonical final box.')
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--log', type=Path, required=True)
    args=ap.parse_args()
    judger=Judger()
    rows=[]; logs=[]; failures=[]
    with args.input.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            qid=row['id']; resp=row['response'] or ''
            final,answer,n_boxes,ok=canonicalize_response(judger, resp)
            rows.append({'id':qid, 'response':final})
            log={'id':qid, 'intended_answer':answer, 'boxes_removed':n_boxes, 'verified':ok, 'extracted_after':judger.extract_ans(final)}
            logs.append(log)
            if not ok:
                failures.append(log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=['id','response'], quoting=csv.QUOTE_ALL)
        w.writeheader(); w.writerows(rows)
    with args.log.open('w', encoding='utf-8') as f:
        for r in logs:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print(json.dumps({'rows':len(rows),'verification_failures':len(failures),'output':str(args.output),'log':str(args.log)}, indent=2))
    if failures:
        print('first_failures='+json.dumps(failures[:10], ensure_ascii=False), file=sys.stderr)

if __name__=='__main__':
    main()
