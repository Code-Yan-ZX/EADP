"""CPU-only, no-label inspection of three lower positive thresholds."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

D = Path(__file__).resolve().parent
B = D.parents[1]
P = B / 'e6/external_sources/pixelprune_8d347eb'
sys.path.insert(0, str(P))
from pixelprune.methods.pred_2d import Pred2DSelector
from pixelprune.patches.qwen3_vl_hf import normalize_pixel_values_for_selector


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU only')
    torch.set_num_threads(1)
    plan = json.loads((D / 'plan001.json').read_text())
    for path, expected in plan['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('source changed: ' + path)
    candidate_file = B / 'e6/pixel_wavefront_cpu/candidate.py'
    spec = importlib.util.spec_from_file_location('wavefront_low_threshold', candidate_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = json.loads((B / 'e6/pixelprune_threshold32/plan001.json').read_text())['samples']
    rows = []
    for i, sample in enumerate(samples):
        if sha(sample['capture_path']) != sample['capture_sha256']:
            raise ValueError('capture changed')
        proc = torch.load(sample['capture_path'], map_location='cpu', weights_only=True)['actual']['processor']
        if proc['image_grid_thw'].tolist() != [[1, 64, 64]]:
            raise ValueError('unexpected grid')
        tokens = normalize_pixel_values_for_selector(proc['pixel_values']).reshape(1024, -1)
        if not torch.isfinite(tokens).all().item():
            raise ValueError('nonfinite input')
        for t in plan['thresholds']:
            author = Pred2DSelector(method='max', threshold=t, anchored=True)
            ref = author._select_2d_loco(tokens, 32, 32, tokens.device)
            candidate = module.select(tokens, 32, 32, t)
            rows.append({'index': i, 'identity': sample['identity'], 'threshold': t,
                         'selected_indices': ref.tolist(), 'candidate_indices': candidate.tolist(),
                         'count': len(ref), 'ordered_equal': torch.equal(ref, candidate)})
    result = {'status': 'completed', 'plan_sha256': sha(D / 'plan001.json'), 'rows': rows,
              'comparisons': len(rows), 'matches': sum(x['ordered_equal'] for x in rows),
              'GT_opened': False, 'predictions_opened': False, 'cuda_initialized': False,
              'limitations': ['Reuses 32 posthoc DEV inputs, no labels or quality.',
                              'CPU identity only; no GPU or full-model equivalence.',
                              'K does not measure actual time or equal compute.']}
    with (D / 'result001.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


if __name__ == '__main__':
    main()
