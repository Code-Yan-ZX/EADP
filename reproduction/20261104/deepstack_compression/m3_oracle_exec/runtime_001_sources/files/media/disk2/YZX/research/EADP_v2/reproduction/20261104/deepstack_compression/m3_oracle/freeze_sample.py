"""Freeze the plan's 128 DEV oracle questions without reading answers.

This only prepares the deterministic sample. It does not read model outputs,
labels, or run a teacher. Full admission and actual GPU smoke remain separate.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
PANEL = BASE / 'data_manifest/panel_manifest.jsonl'
DATASET = BASE / 'data_manifest/manifest_summary.json'
VERIFY = BASE / 'data_manifest/verification.json'
VERIFY_TERMINAL = BASE / 'manifest_verify_supervision/terminal.json'
E1 = BASE / 'e1/e1full001_completion.json'
E1_TERMINAL = BASE / 'e1/monitorverifye1full001/terminal.json'
PLAN = BASE.parents[2] / 'docs/versions/v2/deepstack_compression_experiment_plan_20261104.md'
TASKS = (('DocVQA_VAL', 64), ('TextVQA_VAL', 32), ('AI2D_TEST', 32))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def group_order(group):
    return hashlib.sha256(('20261104|' + group).encode()).hexdigest(), group

def main():
    summary = json.loads(DATASET.read_text())
    verification = json.loads(VERIFY.read_text())
    e1 = json.loads(E1.read_text())
    if verification.get('status') != 'pass' or summary['outputs_sha256']['panel_manifest.jsonl'] != sha(PANEL):
        raise ValueError('unverified panel')
    for terminal in (VERIFY_TERMINAL, E1_TERMINAL):
        state = json.loads(terminal.read_text())
        if state.get('reason') != 'completed' or state.get('worker_returncode') != 0 or state.get('remaining_process_group') != []:
            raise ValueError('upstream verifier did not finish')
    if e1.get('status') != 'passed' or e1.get('natural_questions') != 384 or e1.get('GT_opened') is not False:
        raise ValueError('unverified E1 inputs')
    inputs = Path(e1['files']['inputs']['path'])
    if sha(inputs) != e1['files']['inputs']['sha256']:
        raise ValueError('E1 input file changed')
    rows = [json.loads(x) for x in PANEL.read_text().splitlines()]
    dev = [r for r in rows if r['split'] == 'DEV']
    source = [json.loads(x) for x in inputs.read_text().splitlines()]
    lookup = {(r['task'], str(r['qid'])): r for r in source}
    if len(dev) != 384 or len(source) != 384 or len(lookup) != 384:
        raise ValueError('bad DEV input inventory')
    chosen = []
    for task, size in TASKS:
        panel = [r for r in dev if r['task'] == task]
        if len(panel) != 128 or len({r['group_id'] for r in panel}) != 128:
            raise ValueError('bad task/group inventory')
        for row in sorted(panel, key=lambda r: group_order(r['group_id']))[:size]:
            item = lookup[(task, str(row['qid']))]
            if item['group_id'] != row['group_id']:
                raise ValueError('input group mismatch')
            chosen.append({k: item[k] for k in ('task', 'qid', 'group_id', 'image_path', 'image_sha256', 'message_sha256')})
    if len(chosen) != 128 or len({r['group_id'] for r in chosen}) != 128:
        raise ValueError('cross-task sample duplicate')
    payload = {'stage': 'sample_only', 'created_unix': time.time(), 'selection': 'first groups in original SHA256(20261104|group_id) order per task',
               'selected_after_e4_dev_scoring': True, 'selection_uses_scores': False, 'GT_opened': False, 'predictions_opened': False,
               'source_sha256': {str(p): sha(p) for p in (PLAN, PANEL, DATASET, VERIFY, VERIFY_TERMINAL, E1, E1_TERMINAL, inputs, Path(__file__))},
               'counts': dict(TASKS), 'examples': chosen,
               'limitation': 'sample freeze only; real model, teacher labels, GPU and oracle not executed'}
    path = OUT / 'sample_001.json'
    with path.open('x') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'status': 'sample_frozen', 'questions': len(chosen), 'sha256': sha(path)}))

if __name__ == '__main__':
    main()
