"""MMBench-EN V11 circular scorer (LLaVA round, 2026-10-06).

Grouping rule identical to the audited Qwen-side protocol
(records_20261003 §11.4): the official V11 tsv holds 1292 base questions
(index < 1e6) plus 3584 rotation rows (index >= 1e6, base = index % 1e6).
A group is correct iff EVERY rotation row's prediction parses to the gt
option letter; any parse failure counts the group wrong (failed groups
are reported separately).  Output: circular accuracy over 1292 groups.
"""
import argparse
import json
import re


def extract_letter(text: str):
    m = re.search(r'\b([A-E])\b', text.strip())
    if m:
        return m.group(1)
    m = re.match(r'^([A-E])([\.\,\:\)].*)?$', text.strip())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--question-file', required=True)  # V11 tsv
    ap.add_argument('--result-file', required=True)    # predictions jsonl
    ap.add_argument('--out', default=None)             # write score json
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_table(args.question_file)
    gt = {}
    for _, row in df.iterrows():
        base = int(row['index']) % 1000000
        gt.setdefault(base, set()).add((int(row['index']), row['answer']))

    preds = {}
    for line in open(args.result_file):
        d = json.loads(line)
        preds[int(d['question_id'])] = d['text']

    groups = {}
    for base, rows in gt.items():
        letters = []
        for idx, ans in rows:
            if idx not in preds:
                letters.append(None)
            else:
                p = extract_letter(preds[idx])
                letters.append(1 if p == ans else 0)
        groups[base] = letters

    n = len(groups)
    correct = sum(1 for l in groups.values() if l and all(x == 1 for x in l))
    failed = sum(1 for l in groups.values() if None in l)
    res = {'n_groups': n, 'circular_correct': correct,
           'circular_acc': round(100 * correct / n, 4),
           'failed_groups_counted_0': failed,
           'result_file': args.result_file}
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
