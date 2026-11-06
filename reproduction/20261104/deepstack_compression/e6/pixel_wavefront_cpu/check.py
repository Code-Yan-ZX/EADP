"""CPU-only synthetic and archived-input comparison; no model, labels or CUDA."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from candidate import select

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
PIXEL = BASE / 'e6/external_sources/pixelprune_8d347eb'
OLD = BASE / 'e6/pixelprune_threshold32'
sys.path.insert(0, str(PIXEL))
from pixelprune.methods.pred_2d import Pred2DSelector
from pixelprune.patches.qwen3_vl_hf import normalize_pixel_values_for_selector


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def synthetic():
    gen = torch.Generator(device='cpu').manual_seed(20261105)
    cases = 0
    for h, w in ((1, 1), (1, 7), (7, 1), (2, 3), (3, 2), (5, 5), (8, 9)):
        for dtype in (torch.float32, torch.bfloat16):
            for threshold in (0.4, 0.45, 0.5):
                values = torch.rand((h * w, 5), generator=gen).to(dtype)
                for x in (values, torch.zeros_like(values), torch.ones_like(values)):
                    original = Pred2DSelector(method='max', threshold=threshold, anchored=True)
                    expected = original._select_2d_loco_anchored(x, h, w, x.device)
                    assert torch.equal(expected, select(x, h, w, threshold)), (h, w, dtype, threshold)
                    cases += 1
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU only')
    torch.set_num_threads(1)
    plan = read(HERE / 'plan001.json')
    if any(sha(path) != expected for path, expected in plan['source_sha256'].items()):
        raise ValueError('source or evidence changed')
    count = synthetic()
    old = read(OLD / 'plan001.json')
    rows = []
    for i, sample in enumerate(old['samples']):
        capture = Path(sample['capture_path'])
        if sha(capture) != sample['capture_sha256']:
            raise ValueError('capture changed')
        processor = torch.load(capture, map_location='cpu', weights_only=True)['actual']['processor']
        grid = processor['image_grid_thw'].tolist()
        if len(grid) != 1 or grid[0] != [1, 64, 64]:
            raise ValueError('unexpected processor grid')
        values = normalize_pixel_values_for_selector(processor['pixel_values'])
        tokens = values.reshape(1024, -1)
        journal = [json.loads(line) for line in (OLD / f'run001/image_{i:03d}/journal.jsonl').read_text().splitlines()]
        for threshold in plan['thresholds']:
            references = [x for x in journal if x.get('event') == 'completed' and x['threshold'] == threshold]
            if len(references) != 1:
                raise ValueError('missing original reference')
            before = time.monotonic_ns()
            actual = select(tokens, 32, 32, threshold).tolist()
            elapsed = time.monotonic_ns() - before
            expected = references[0]['selected_indices']
            rows.append({'index': i, 'identity': sample['identity'], 'threshold': threshold,
                         'selected_count': len(actual), 'selected_indices': actual,
                         'ordered_equal': actual == expected, 'wall_ns_unpaired': elapsed})
    result = {'status': 'completed', 'plan_sha256': sha(HERE / 'plan001.json'),
              'synthetic_cases': count, 'images': len(old['samples']), 'comparisons': len(rows),
              'matches': sum(x['ordered_equal'] for x in rows), 'rows': rows,
              'GT_opened': False, 'predictions_opened': False, 'gpu_used': False,
              'model_loaded': False, 'cuda_initialized': torch.cuda.is_initialized(),
              'limitations': ['Finite CPU BF16 inputs only; no claim of GPU numerical or model identity.',
                              'Wall times are unpaired and include no preprocessor or model; no speed claim.',
                              'The prior image set and thresholds were selected after existing DEV work.']}
    with args.out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


if __name__ == '__main__':
    main()
