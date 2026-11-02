#!/usr/bin/env python3
"""Small, declared synthetic boundary checks; no dataset score or GPU use."""
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
rows = []
for label, scores in [('constant', [0.25] * 9), ('nonconstant', [0., .001, .01])]:
    x = torch.tensor(scores)
    span = x.max() - x.min()
    paper = (x - x.min()) / (span + 1e-6)
    source = (x - x.min() + 1e-6) / (span + 1e-6)
    rows.append({'case': label, 'input': scores, 'paper_Eq10': paper.tolist(),
                 'source_encode_images': source.tolist(), 'max_abs_difference': float((paper - source).abs().max())})
entropy = torch.arange(9, dtype=torch.float32)
q = .2
threshold = torch.quantile(entropy, q)
result = {'scope': 'Synthetic arithmetic counterexamples only. Not evidence that finished GPU runs encounter degenerate scores or that this closes any benchmark gap.',
          'minmax_cases': rows,
          'Eq6_quantile_vs_floor_example': {'entropies': entropy.tolist(), 'q': q,
              'linear_quantile_threshold': float(threshold), 'paper_threshold_kept_count': int((entropy <= threshold).sum()),
              'source_floor_topk_kept_count': max(1, int(q * len(entropy))),
              'tie_note': 'Eq6 threshold retains all ties at the threshold; source topk always takes a fixed count. Paper does not specify sample quantile interpolation.'},
          'cuda_initialized': torch.cuda.is_initialized()}
(ROOT / 'cpu_formula_boundary_checks.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
