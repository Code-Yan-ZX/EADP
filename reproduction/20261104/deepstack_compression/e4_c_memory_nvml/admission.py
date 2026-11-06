"""CPU-only source gate for real sampler preparations; no GPU permission."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/'e4_c_memory_real'))
import gate as prior
R=BASE/'e1_independent_review';PROOF=R/'e4c_memory_real_preflight001_review.json'
REVIEWER=R/'review_e4c_memory_real_preflight001.py';MON=R/'monitor_e4c_memory_real_preflight001'
NAMES=('PLAN.md','nvml_source.py','real_sampler.py','verify_metadata.py','admission.py','cpu_check.py','test_nvml_source.py')

def sources():
    actual=prior.sources();actual.update({str((HERE/n).resolve()):prior.runtime.sha(HERE/n) for n in NAMES});return actual

def admit_cpu():
    prior.admit_cpu();rt=prior.runtime;rt.supervisor(MON,REVIEWER)
    proof=json.loads(PROOF.read_text())
    if (proof.get('status')!='passed' or proof.get('independent_review') is not True
        or proof.get('scope')!='real NVML worker CPU preflight, no actual sampling'
        or proof.get('gpu_authorized') is not False or proof.get('GT_opened') is not False
        or proof.get('existing_DEV_predictions_opened') is not True or proof.get('new_predictions_generated') is not False
        or proof.get('schedule_jobs')!=72 or proof.get('source_count')!=100 or proof.get('errors')!=[]
        or proof.get('reviewer_sha256')!=rt.sha(REVIEWER)):
        raise ValueError('prior preflight review mismatch')
    files=proof.get('files_sha256',{})
    if len(files)!=7 or any(rt.sha(path)!=digest for path,digest in files.items()):raise ValueError('prior evidence changed')
    return {'scope':'CPU-only real sampler preparation','gpu_authorized':False,
            'source_sha256':sources(),'prior_review_sha256':rt.sha(PROOF)}

if __name__=='__main__':raise SystemExit('locked: CPU admission cannot authorize GPU')
