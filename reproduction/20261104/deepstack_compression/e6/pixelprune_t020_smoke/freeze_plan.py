"""Freeze four exploratory threshold-0.2 anchors after CPU independent review."""
import hashlib
import json
import os
from pathlib import Path

B = Path(__file__).resolve().parents[2]
D = B / 'e6/pixelprune_t020_smoke'
R = B / 'e1_independent_review'
C = B / 'e6/pixel_low_threshold_cpu'
S = B / 'e6/pixelprune_threshold32'
P = B / 'e6/pixelprune_dev384'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


review_path = R / 'pixel_low_threshold_cpu_review.json'
reviewer = R / 'review_pixel_low_threshold_cpu.py'
review_mon = R / 'monitor_pixel_low_threshold_cpu'
review = read(review_path)
started = read(review_mon / 'started.json')
terminal = read(review_mon / 'terminal.json')
if (review.get('status') != 'passed' or review.get('independent_review') is not True or
    review.get('case_count') != 96 or review.get('ordered_matches') != 96 or
    review.get('GT_opened') is not False or review.get('predictions_opened') is not False or
    review.get('errors') != [] or review.get('reviewer_sha256') != sha(reviewer) or
    review.get('result_sha256') != sha(C / 'result001.json') or
    review.get('plan_sha256') != sha(C / 'plan001.json') or
    terminal.get('reason') != 'completed' or terminal.get('worker_returncode') != 0 or
    terminal.get('remaining_process_group') != [] or
    started.get('worker_pid') != terminal.get('worker_pid')):
    raise ValueError('independent CPU calibration incomplete')
test_started = read(D / 'monitor_test002/started.json')
test_terminal = read(D / 'monitor_test002/terminal.json')
if (test_terminal['reason'] != 'completed' or test_terminal['worker_returncode'] != 0 or
    test_terminal['remaining_process_group'] != [] or
    test_started['worker_pid'] != test_terminal['worker_pid']):
    raise ValueError('CPU smoke tests incomplete')
samples = read(S / 'plan001.json')['samples']
rows = [x for x in read(C / 'result001.json')['rows'] if x['threshold'] == 0.2]
assert len(rows) == len(samples) == 32 and all(x['ordered_equal'] for x in rows)
chosen = [next(x for x in rows if x['identity']['task'] == task)
          for task in ('DocVQA_VAL', 'TextVQA_VAL', 'AI2D_TEST')]
used = {x['index'] for x in chosen}
chosen.append(min((x for x in rows if x['index'] not in used), key=lambda x: (x['count'], x['index'])))
original = read(P / 'plan001.json')
by_key = {(x['task'], str(x['qid'])): x for x in original['anchors']}
anchors = []
for row in chosen:
    i = row['index']
    sample = samples[i]
    identity = sample['identity']
    key = (identity['task'], str(identity['qid']))
    a = dict(by_key[key])
    if a['group_id'] != sample['group_id'] or sha(sample['capture_path']) != sample['capture_sha256']:
        raise ValueError('calibration identity changed')
    a.update(cpu_index=i, cpu_K=row['count'], cpu_selected_indices=row['selected_indices'])
    anchors.append(a)
if len({x['group_id'] for x in anchors}) != 4:
    raise ValueError('duplicate groups')
source = dict(original['sha256'])
files = [D / 'worker.py', D / 'launch.py', D / 'test_worker.py', D / 'freeze_plan.py',
         P / 'plan001.json', S / 'plan001.json', C / 'plan001.json', C / 'result001.json',
         review_path, reviewer, review_mon / 'started.json', review_mon / 'terminal.json',
         D / 'monitor_test002/started.json', D / 'monitor_test002/terminal.json',
         D / 'monitor_test002/stderr.log']
files += [Path(x) for x in read(C / 'plan001.json')['source_sha256']]
files += [Path(samples[x['index']]['capture_path']) for x in chosen]
for f in files:
    source[str(f.resolve())] = sha(f)
plan = {'stage': 'posthoc threshold 0.2 four-DEV natural diagnostic; no scoring',
        'anchors': anchors, 'threshold': 0.2, 'method': 'pred_2d', 'metric': 'max',
        'anchored': True, 'vit_layer': 0, 'generation': original['generation'],
        'budget_source_sha256': original['budget_source_sha256'],
        'source_commit': original['source_commit'], 'calibration_review_sha256': sha(review_path),
        'sha256': source, 'stale_seconds': 1200, 'GT_opened': False,
        'predictions_opened_for_identity': True,
        'selection_rule': 'first calibration row per task plus minimum-K distinct remaining image; posthoc',
        'limitations': ['Threshold and anchors selected after prior DEV work and CPU K calibration.',
                        'Four chosen images are not representative.',
                        'Dynamic K is not equal compute; no quality or speed claim.',
                        'No full high-dimensional visual/DS/KV arrays archived.']}
with (D / 'plan001.json').open('x') as f:
    json.dump(plan, f, indent=2)
    f.write('\n')
    f.flush()
    os.fsync(f.fileno())
print('frozen', [(x['task'], x['qid'], x['cpu_K']) for x in anchors], sha(D / 'plan001.json'))
