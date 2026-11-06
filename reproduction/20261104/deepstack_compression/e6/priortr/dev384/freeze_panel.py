"""Freeze the complete existing DEV input panel for PriorTR; never open labels."""
import collections
import hashlib
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def lines(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    proof_path = BASE / 'e1/e1full001_completion.json'
    proof = json.loads(proof_path.read_text())
    if proof['status'] != 'passed' or proof['GT_opened'] is not False or proof['scoring_performed'] is not False:
        raise ValueError('E1 proof is not accepted without labels')
    inputs, panel = (proof['files'][key] for key in ('inputs', 'panel'))
    if any(sha(item['path']) != item['sha256'] for item in (inputs, panel)):
        raise ValueError('input hashes changed')
    rows = lines(inputs['path'])
    dev = {(x['task'], str(x['qid'])): x for x in lines(panel['path']) if x['split'] == 'DEV'}
    if len(rows) != 384 or len(dev) != 384 or len({(x['task'], str(x['qid'])) for x in rows}) != 384:
        raise ValueError('DEV inventory')
    keys = ('task', 'qid', 'group_id', 'image_path', 'image_sha256', 'message', 'message_sha256')
    sample = []
    for row in rows:
        key = (row['task'], str(row['qid']))
        if key not in dev or row['group_id'] != dev[key]['group_id'] or any(k in row for k in ('answer','answers','ground_truth','prediction')):
            raise ValueError('input not label-free or does not belong to DEV')
        sample.append({k: row[k] for k in keys})
    expected = {'DocVQA_VAL': 128, 'TextVQA_VAL': 128, 'AI2D_TEST': 128}
    if collections.Counter(x['task'] for x in sample) != expected or len({x['group_id'] for x in sample}) != 384:
        raise ValueError('task/group inventory')
    result = {
        'stage': 'complete_dev_competitor_panel', 'created_unix': time.time(),
        'selection': 'entire frozen E1 DEV input inventory in its existing order; no selection by answer or output',
        'design_after_existing_dev_results': True,
        'E1_proof_sha256': sha(proof_path), 'input_sha256': inputs['sha256'], 'panel_sha256': panel['sha256'],
        'freeze_source_sha256': sha(__file__), 'sample': sample,
        'physical_generations': 1152, 'arms': ['stock_native', 'author_default', 'priortr_k256'],
        'recipe': 'E1 inputs and generation settings, fixed PriorTR author environment and SDPA; default unchanged; k256 fixed keep_tokens=256',
        'limits': ['Cross-backend output comparisons do not establish numerical equivalence.',
                   'Instrumented generation is not a latency benchmark.',
                   'DEV results inform later selection and do not establish FINAL performance.',
                   'Previously run DEV32 outputs may overlap; this plan re-generates all 1152 arms.'],
        'GT_opened': False, 'prediction_opened': False, 'scoring_performed': False,
    }
    out = HERE / 'sample_001.json'
    with out.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps({'status': 'frozen', 'questions': len(sample), 'physical_generations': 1152}))

if __name__ == '__main__':
    main()
