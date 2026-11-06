"""Execution source closure; preserve container path aliases."""
import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def sources():
 prior=json.loads((I/'pixel_wavefront_model_t020_preflight_result_review.json').read_text());paths={Path(os.path.abspath(x)) for x in prior['source_sha256']}
 paths.update(Path(os.path.abspath(p)) for p in D.glob('*.py'));paths.add(D/'PREFLIGHT_REVIEWER_PROVENANCE.md');paths.add(D/'preflight_plan001.json');paths.update(p for p in (D/'history_plan001_sources').iterdir() if p.is_file())
 paths.update(I/x for x in ('pixel_wavefront_model_t020_preflight_result_review.json','review_pixel_wavefront_model_t020_preflight_result.py','monitor_pixel_wavefront_model_t020_preflight_result/started.json','monitor_pixel_wavefront_model_t020_preflight_result/terminal.json'))
 return sorted(paths,key=str)
def hashes():return {str(p):sha(p) for p in sources()}
