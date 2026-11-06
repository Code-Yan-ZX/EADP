"""Freeze an exploratory GPU selector diagnostic after independent CPU review."""
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
B = D.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


cpu = B / 'e6/pixel_exact_zero_cpu'
review = B / 'e1_independent_review/pixel_exact_zero_cpu_review.json'
reviewer = B / 'e1_independent_review/review_pixel_exact_zero_cpu.py'
review_mon = B / 'e1_independent_review/monitor_pixel_exact_zero_cpu'
old = B / 'e6/pixelprune_threshold32'
samples = read(old / 'plan001.json')['samples']
result = read(cpu / 'result001.json')
proof = read(review)
terminal = read(review_mon / 'terminal.json')
assert (len(samples) == 32 and result['matches'] == 32 and
        result['synthetic_matches'] == 56 and proof['status'] == 'passed' and
        terminal['reason'] == 'completed' and terminal['worker_returncode'] == 0 and
        terminal['remaining_process_group'] == [])
anchors = []
for i, sample in enumerate(samples):
    row = result['images'][i]
    assert row['index'] == i and row['ordered_equal'] and sha(sample['capture_path']) == sample['capture_sha256']
    ident = sample['identity']
    anchors.append({'task': ident['task'], 'qid': str(ident['qid']),
                    'group_id': sample['group_id'], 'capture_path': sample['capture_path'],
                    'expected_indices': row['reference']})
assert len({x['group_id'] for x in anchors}) == 32
files = [*D.glob('*.py'), B / 'initial_plan.json', B / 'monitor/supervise.py',
         B / 'e4_c_latency/timing_launch.py', cpu / 'plan001.json', cpu / 'result001.json',
         cpu / 'monitor_check001/started.json', cpu / 'monitor_check001/terminal.json',
         review, reviewer, review_mon / 'started.json', review_mon / 'terminal.json',
         *[Path(x['capture_path']) for x in anchors]]
files += [Path(x) for x in read(cpu / 'plan001.json')['source_sha256']]
root = B.parents[2]
ap = root / 'reproduction/20261028/qwen/gpu_formula_facility_panel_run02'
files += [ap / 'run_plan.json', ap / 'run_panel.py', root / 'reproduction/20261028/qwen/container_runtime.py']
plan = {'scope': 'posthoc 32 existing DEV images: author vs zero-specialized GPU selector',
        'threshold': 0.0, 'anchors': anchors,
        'source_sha256': {str(x.resolve()): sha(x) for x in sorted(set(files), key=str)},
        'budget_source_sha256': sha(B / 'initial_plan.json'),
        'model_loaded': False, 'GT_opened': False, 'new_predictions_opened': False,
        'limitations': ['CPU review precedes GPU; no extrapolation to all inputs or devices.',
                        '32 images are the same previously inspected DEV images.',
                        'One synchronized wall sample per arm/image; no model or end-to-end performance conclusion.',
                        'Numerical zero equivalence is a conditional argument, especially under device flush-to-zero.']}
with (D / 'plan001.json').open('x') as f:
    json.dump(plan, f, indent=2)
    f.write('\n')
print('sources', len(plan['source_sha256']), 'anchors', len(anchors))
