"""Frozen CPU-only check of zero-threshold selector, no labels or model."""
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from candidate import select

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE / 'e6/external_sources/pixelprune_8d347eb'))
from pixelprune.methods.pred_2d import Pred2DSelector
from pixelprune.patches.qwen3_vl_hf import normalize_pixel_values_for_selector


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU only')
    torch.set_num_threads(1)
    plan = json.loads((HERE / 'plan001.json').read_text())
    for name, digest in plan['source_sha256'].items():
        if sha(name) != digest:
            raise ValueError('source changed: ' + name)
    author = Pred2DSelector(method='max', threshold=0.0, anchored=True)
    gen = torch.Generator(device='cpu').manual_seed(20261105)
    cases = []
    for h, w in ((1, 1), (1, 7), (7, 1), (2, 3), (3, 2), (5, 5), (8, 9)):
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randint(0, 4, (h * w, 5), generator=gen).to(dtype)
            sub = torch.zeros_like(x)
            sub[::2] = torch.finfo(dtype).tiny / 2
            for label, values in [('quantized', x), ('zeros', torch.zeros_like(x)),
                                  ('ones', torch.ones_like(x)), ('subnormal', sub)]:
                ref = author._select_2d_loco_anchored(values, h, w, values.device)
                got = select(values, h, w)
                cases.append({'grid': [h, w], 'dtype': str(dtype), 'case': label,
                              'reference': ref.tolist(), 'candidate': got.tolist(),
                              'ordered_equal': torch.equal(ref, got)})
    old = json.loads((BASE / 'e6/pixelprune_threshold32/plan001.json').read_text())
    rows = []
    for i, sample in enumerate(old['samples']):
        capture = Path(sample['capture_path'])
        if sha(capture) != sample['capture_sha256']:
            raise ValueError('capture changed')
        proc = torch.load(capture, map_location='cpu', weights_only=True)['actual']['processor']
        if proc['image_grid_thw'].tolist() != [[1, 64, 64]]:
            raise ValueError('unexpected processor grid')
        x = normalize_pixel_values_for_selector(proc['pixel_values']).reshape(1024, -1)
        if not torch.isfinite(x).all().item():
            raise ValueError('nonfinite real input')
        ref = author._select_2d_loco_anchored(x, 32, 32, x.device)
        got = select(x, 32, 32)
        rows.append({'index': i, 'identity': sample['identity'],
                     'reference': ref.tolist(), 'candidate': got.tolist(),
                     'ordered_equal': torch.equal(ref, got)})
    result = {'status': 'completed', 'plan_sha256': sha(HERE / 'plan001.json'),
              'synthetic_cases': cases, 'images': rows,
              'matches': sum(x['ordered_equal'] for x in rows),
              'synthetic_matches': sum(x['ordered_equal'] for x in cases),
              'cuda_initialized': torch.cuda.is_initialized(), 'GT_opened': False,
              'predictions_opened': False, 'model_loaded': False,
              'limitations': ['CPU only; finite actual BF16 processor inputs.',
                              'Device-specific underflow may invalidate the mathematical precondition.',
                              'Prior 32 DEV images reused; no GPU, model, or speed conclusion.']}
    with (HERE / 'result001.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


if __name__ == '__main__':
    main()
