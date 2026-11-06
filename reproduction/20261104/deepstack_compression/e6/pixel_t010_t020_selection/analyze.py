"""Descriptive matched DEV token-set overlap; no ground truth or scoring."""
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
B = D.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def panel(tag, plan):
    review_path = B / f'e1_independent_review/pixel_{tag}_dev384_result_review.json'
    review = read(review_path)
    formal_path = B / f'e6/pixelprune_{tag}_accept/verification001.json'
    if (sha(review_path) != plan['reviews_sha256'][str(review_path)] or review.get('status') != 'passed' or
        review.get('independent_review') is not True or review.get('questions') != 384 or
        review.get('GT_opened') is not False or review.get('formal_sha256') != sha(formal_path) or
        review.get('errors') != []):
        raise ValueError('independent review missing or altered')
    formal = read(formal_path)
    if formal.get('status') != 'passed' or formal.get('questions') != 384 or formal.get('GT_opened') is not False:
        raise ValueError('formal proof missing')
    records = {}
    obs = B / f'e6/pixelprune_{tag}_dev384/run_smoke001/observations'
    for item in formal['per_question']:
        task, qid = item['task'], str(item['qid'])
        p = obs / f'{task}_{qid}.json'
        if formal['files_sha256'].get(str(p)) != sha(p) or review['per_question_files_sha256'].get(str(p)) != sha(p):
            raise ValueError('observation changed')
        row = read(p)  # Contains predictions; only identity and selected indices are used.
        ids = row['selected_indices']
        if (row['task'] != task or str(row['qid']) != qid or row.get('status') != 'completed' or
            ids != sorted(set(ids)) or len(ids) != row['selected_count'] or
            not ids or any(type(i) is not int or not 0 <= i < 1024 for i in ids)):
            raise ValueError('invalid identity or index inventory')
        key = (task, qid, row['group_id'], row['image_sha256'], row['message_sha256'])
        if key in records:
            raise ValueError('duplicate image')
        records[key] = set(ids)
    if len(records) != 384:
        raise ValueError('incomplete panel')
    return records


def main():
    plan = read(D/'plan001.json')
    frozen = read(D/'source_freeze001.json')
    if frozen['plan_sha256'] != sha(D/'plan001.json') or frozen['script_sha256'] != sha(__file__):
        raise ValueError('plan or source changed')
    a, b = panel('t010', plan), panel('t020', plan)
    if set(a) != set(b):
        raise ValueError('matched panel mismatch')
    groups = defaultdict(list)
    for key in a:
        x, y = a[key], b[key]
        groups[key[0]].append((len(x)-len(y), y <= x, len(x & y)/len(x | y)))
    results = {}
    for task in plan['tasks']:
        rows = groups[task]
        if len(rows) != 128:
            raise ValueError('task denominator')
        diff = [r[0] for r in rows]
        results[task] = {'questions': 128, 'k010_minus_k020_min_median_max': [int(min(diff)), float(np.median(diff)), int(max(diff))],
                         'k010_less_than_k020': sum(v < 0 for v in diff),
                         't020_not_subset_of_t010': sum(not r[1] for r in rows),
                         'jaccard_min_median_max': [float(min(r[2] for r in rows)), float(np.median([r[2] for r in rows])), float(max(r[2] for r in rows))]}
    out = {'status': 'completed', 'plan_sha256': sha(D/'plan001.json'), 'script_sha256': sha(__file__),
           'GT_opened': False, 'scoring_performed': False, 'mixed_prediction_files_opened': True,
           'results': results, 'limitations': plan['limitations']}
    with (D/'result001.json').open('x') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


if __name__ == '__main__':
    main()
