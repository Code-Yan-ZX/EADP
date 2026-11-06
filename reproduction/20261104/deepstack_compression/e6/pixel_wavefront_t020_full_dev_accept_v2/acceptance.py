"""Bind post-run verifier revision and preserve the failed first attempt."""
import hashlib
import json
from pathlib import Path
V=Path(__file__).resolve().parent;D=V.parent/'pixel_wavefront_t020_full_dev'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def check():
    p=json.loads((V/'acceptance_plan002.json').read_text())
    if p.get('run_plan_sha256')!=sha(D/'plan002.json') or any(sha(k)!=v for k,v in p['source_sha256'].items()):
        raise ValueError('acceptance sources or original run plan changed')
    if any(sha(k)!=v for k,v in p['prior_evidence_sha256'].items()):raise ValueError('original/failed evidence changed')
    return p

def approval():
    import baseline
    base=D.parents[1];review=base/'e1_independent_review';proof=review/'pixel_wavefront_full_accept_v2_prelaunch_review.json'
    reviewer=review/'review_pixel_wavefront_full_accept_v2_prelaunch.py';mon=review/'monitor_pixel_wavefront_full_accept_v2_prelaunch'
    baseline.supervised(mon,reviewer,proof);baseline.supervised(V/'monitor_cpu002',V/'cpu_validation.py')
    p=json.loads(proof.read_text());cpu=json.loads((V/'cpu_proof002.json').read_text())
    expected={'acceptance_plan_sha256':V/'acceptance_plan002.json','reviewer_sha256':reviewer,'cpu_proof_sha256':V/'cpu_proof002.json',
              'cpu_started_sha256':V/'monitor_cpu002/started.json','cpu_terminal_sha256':V/'monitor_cpu002/terminal.json',
              'guard_smoke_sha256':V/'guard_smoke002.json'}
    if (p.get('status')!='passed' or p.get('independent_review') is not True or p.get('errors')!=[]
        or any(p.get(k)!=sha(v) for k,v in expected.items()) or cpu.get('status')!='passed'
        or cpu.get('acceptance_plan_sha256')!=sha(V/'acceptance_plan002.json')):
        raise ValueError('post-run acceptance independent review missing')
