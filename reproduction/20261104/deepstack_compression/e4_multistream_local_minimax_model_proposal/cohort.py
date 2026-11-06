"""CPU-only cohort/selection verification; no model or scoring entrypoint."""
import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
I = BASE / 'e1_independent_review'
OLD = BASE / 'e4_multistream_local_minimax_v2'
ARMS = ('minimax', 'main_minimax')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def supervisor(directory, script, out=None):
    start, end = read(directory / 'started.json'), read(directory / 'terminal.json')
    cwd, cmd = Path(start['cwd']), start['command']
    actual = [(cwd / x).resolve() for x in cmd if Path(x).name == script.name]
    if (actual != [script.resolve()] or end.get('reason') != 'completed' or
        end.get('worker_returncode') != 0 or end.get('remaining_process_group') != [] or
        start.get('worker_pid') != end.get('worker_pid')):
        raise ValueError('supervisor/source mismatch')
    if out is not None and (cmd.count('--out') != 1 or (cwd / cmd[cmd.index('--out') + 1]).resolve() != out.resolve()):
        raise ValueError('supervisor output mismatch')


def validate_ids(ids):
    if (not isinstance(ids, list) or len(ids) != 256 or
        any(type(i) is not int or not 0 <= i < 1024 for i in ids) or
        sorted(set(ids)) != ids):
        raise ValueError('invalid sorted unique K256 indices')
    blocks = [(i // 32 // 2) * 16 + (i % 32 // 2) for i in ids]
    if sorted(blocks) != list(range(256)):
        raise ValueError('not exactly one per 2x2 block')


def construct(schedule, archive_inventory, feature_rows, formal, independent):
    identities = schedule['input_identities']
    archives = archive_inventory['archives']
    if any(len(x) != 32 for x in (identities, archives, feature_rows)):
        raise ValueError('32-item denominator')
    if Counter(x['task'] for x in identities) != {'DocVQA_VAL': 11, 'TextVQA_VAL': 11, 'AI2D_TEST': 10}:
        raise ValueError('task denominator')
    if (formal.get('status') != 'passed' or formal.get('images') != 32 or
        independent.get('status') != 'passed' or independent.get('images') != 32 or
        independent.get('independent_review') is not True):
        raise ValueError('component acceptance')
    result, seen = [], set()
    for index, (identity, archive, row) in enumerate(zip(identities, archives, feature_rows)):
        key = (identity['task'], str(identity['qid']))
        if key in seen:
            raise ValueError('duplicate task/qid')
        seen.add(key)
        if archive['index'] != index or row['index'] != index:
            raise ValueError('changed archive order')
        for field in ('task', 'qid', 'group_id'):
            if str(identity[field]) != str(archive[field]) or str(identity[field]) != str(row[field]):
                raise ValueError('image identity mismatch: ' + field)
        for field in ('image_sha256', 'message_sha256'):
            if identity[field] != archive[field]:
                raise ValueError('input identity mismatch: ' + field)
        a_hash = archive['sha256']
        path = archive['path']
        if row['archive_sha256'] != a_hash or formal['array_sha256'].get(path) != a_hash or independent['array_sha256'].get(path) != a_hash:
            raise ValueError('archive hash not accepted')
        ids = {}
        for arm in ARMS:
            selected = row['arms'][arm]['ids']
            validate_ids(selected)
            ids[arm] = selected
        result.append({'index': index, 'task': key[0], 'qid': key[1],
                       'group_id': identity['group_id'], 'image_sha256': identity['image_sha256'],
                       'message_sha256': identity['message_sha256'],
                       'processor': identity['processor'], 'archive_path': path,
                       'archive_sha256': a_hash, 'ids': ids})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.resolve() != (HERE / 'cohort001.json').resolve() or args.out.exists():
        raise ValueError('fixed new output required')
    files = {'schedule': BASE/'e4_c_microbench/schedule001.json', 'inventory': OLD/'inventory001.json',
             'rows': OLD/'rows001.jsonl', 'feature_result': OLD/'result001.json',
             'feature_plan': OLD/'plan001.json', 'feature_review': I/'local_minimax_v2_result_review.json',
             'reviewer': I/'review_local_minimax_v2_result.py',
             'formal': BASE/'e4_c_microbench_full/full_001_verification.json',
             'independent': I/'component_full001_review.json'}
    result, review, inv = read(files['feature_result']), read(files['feature_review']), read(files['inventory'])
    if (result.get('status') != 'passed' or result.get('images') != 32 or result.get('errors') != [] or
        review.get('status') != 'passed' or review.get('independent_review') is not True or
        review.get('images') != 32 or review.get('errors') != [] or
        result.get('rows_sha256') != sha(files['rows']) or result.get('plan_sha256') != sha(files['feature_plan']) or
        review.get('rows_sha256') != sha(files['rows']) or review.get('result_sha256') != sha(files['feature_result']) or
        review.get('plan_sha256') != sha(files['feature_plan']) or review.get('reviewer_sha256') != sha(files['reviewer']) or
        inv.get('schedule_sha256') != sha(files['schedule']) or inv.get('formal_sha256') != sha(files['formal']) or
        inv.get('independent_review_sha256') != sha(files['independent'])):
        raise ValueError('upstream proof identity')
    supervisor(OLD/'monitor_run001', OLD/'run.py', files['feature_result'])
    supervisor(I/'monitor_local_minimax_v2_result', files['reviewer'], files['feature_review'])
    for key, name in [('run_started_sha256','started.json'), ('run_terminal_sha256','terminal.json')]:
        if review.get(key) != sha(OLD/'monitor_run001'/name):
            raise ValueError('reviewer/run binding')
    with files['rows'].open() as stream:
        rows = [json.loads(line) for line in stream]
    cohort = construct(read(files['schedule']), inv, rows, read(files['formal']), read(files['independent']))
    payload = {'status': 'passed', 'scope': 'CPU input and selected-ID preparation only, not model/GPU admission',
               'images': 32, 'jobs': 64, 'task_counts': dict(Counter(r['task'] for r in cohort)),
               'GT_opened': False, 'GPU_used': False, 'model_executed': False,
               'real_feature_archives_opened': False, 'source_sha256': {str(p): sha(p) for p in files.values()},
               'cohort': cohort, 'errors': []}
    with args.out.open('x') as out:
        json.dump(payload, out, indent=2); out.write('\n'); out.flush(); os.fsync(out.fileno())


if __name__ == '__main__':
    main()
