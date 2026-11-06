"""CPU-only host preparation proof; never authorizes real sampling."""
import importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;R=BASE/'e1_independent_review'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
upstream=load('eadp_host_smoke_cpu_gate',BASE/'e4_c_memory_smoke/cpu_gate.py');rt=upstream.rt
PROOF=R/'e4c_memory_smoke_preflight001_review.json';REVIEWER=R/'review_e4c_memory_smoke_preflight001.py';MON=R/'monitor_e4c_memory_smoke_preflight001'
NAMES=('PLAN.md','host_plan.json','host_contract.py','test_contract.py','host_admission.py','runtime_gate.py','test_runtime_gate.py','cpu_check.py')

def sources():
    actual=upstream.sources();actual.update({str((HERE/n).resolve()):rt.sha(HERE/n) for n in NAMES});return actual

def admit_cpu():
    previous=upstream.admit_cpu();rt.supervisor(MON,REVIEWER)
    r=json.loads(PROOF.read_text());files=r.get('files_sha256',{})
    d=BASE/'e4_c_memory_smoke'
    required={str(p.resolve()) for p in (d/'container_preflight001/launch.json',d/'container_preflight001/exit.json',
              d/'container_preflight001/worker_preflight.json',d/'monitor_preflight001/started.json',
              d/'monitor_preflight001/terminal.json',R/'e4c_memory_smoke_worker_preflight_review.json',
              R/'e4c_memory_runtime_container001_scope_correction.json')}
    if (r.get('status')!='passed' or r.get('independent_review') is not True
        or r.get('scope')!='C-ON worker CPU preflight, no actual sampling' or r.get('gpu_authorized') is not False
        or r.get('GT_opened') is not False or r.get('scoring_performed') is not False
        or r.get('existing_DEV_predictions_opened') is not True or r.get('new_predictions_generated') is not False
        or r.get('schedule_jobs')!=72 or r.get('source_count')!=119 or r.get('errors')!=[]
        or r.get('reviewer_sha256')!=rt.sha(REVIEWER) or set(files)!=required
        or any(rt.sha(path)!=digest for path,digest in files.items())):
        raise ValueError('C-ON CPU preflight independent proof mismatch')
    return {'scope':'host CPU preparation only','gpu_authorized':False,'source_sha256':sources(),
            'worker_preflight_review_sha256':rt.sha(PROOF),'nvml_review_sha256':previous['nvml_review_sha256']}

if __name__=='__main__':raise SystemExit('locked: CPU proof cannot authorize GPU')
