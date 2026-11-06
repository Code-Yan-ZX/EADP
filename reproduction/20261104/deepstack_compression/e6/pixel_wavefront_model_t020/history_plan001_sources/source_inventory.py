"""Deterministic source/evidence inventory for the model identity preparation."""
import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];S=B/'e6/pixelprune_t020_smoke';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def stable(p):
 """Preserve absolute mount aliases; resolve() differs between host/container."""
 return Path(os.path.abspath(os.fspath(p)))
def sources():
 old=json.loads((S/'plan001.json').read_text());proof=json.loads((I/'pixel_t020_result_review.json').read_text())
 paths={stable(x) for x in old['sha256']};paths.update(stable(x) for x in proof['files_sha256'])
 component=json.loads((I/'pixel_wavefront_t020_gpu32_result_review.json').read_text());paths.update(stable(x) for x in component['evidence_sha256'])
 paths.update(stable(p) for p in D.glob('*.py'));paths.add(stable(D/'PLAN_DRAFT.md'))
 paths.update(stable(B/x) for x in ('initial_plan.json','monitor/supervise.py','e6/pixel_wavefront_cpu/candidate.py','e6/pixelprune_t020_smoke/plan001.json','e1_independent_review/pixel_t020_result_review.json','e1_independent_review/review_pixel_t020_result.py','e1_independent_review/monitor_pixel_t020_result/started.json','e1_independent_review/monitor_pixel_t020_result/terminal.json','e1_independent_review/pixel_wavefront_t020_gpu32_result_review.json','e1_independent_review/review_pixel_wavefront_t020_gpu32_result.py','e1_independent_review/monitor_pixel_wavefront_t020_gpu32_result/started.json','e1_independent_review/monitor_pixel_wavefront_t020_gpu32_result/terminal.json'))
 return sorted(paths,key=str)
def source_hashes():return {str(p):sha(p) for p in sources()}
