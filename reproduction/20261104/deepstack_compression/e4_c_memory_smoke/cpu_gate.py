"""CPU-only evidence chain; deliberately cannot grant GPU authority."""
import importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;R=BASE/'e1_independent_review'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
upstream=load('eadp_smoke_nvml_admission',BASE/'e4_c_memory_nvml/admission.py');rt=upstream.prior.runtime
PROOF=R/'e4c_memory_nvml_cpu006_review.json';REVIEWER=R/'review_e4c_memory_nvml_cpu006.py';MON=R/'monitor_e4c_memory_nvml_cpu006'
CPU=BASE/'e4_c_memory_nvml/cpu_check006/verification.json';CPUMON=BASE/'e4_c_memory_nvml/monitor_cpu006'
NAMES=('PLAN.md','preflight_plan.json','worker_contract.py','worker.py','wiring.py','cpu_gate.py',
       'preflight_gate.py','preflight_launch.py','preflight_entry.py','cpu_check.py','test_worker_contract.py','test_wiring.py')

def sources():
    x=upstream.sources();x.update({str((HERE/n).resolve()):rt.sha(HERE/n) for n in NAMES});return x
def admit_cpu():
    prior=upstream.admit_cpu();rt.supervisor(MON,REVIEWER,PROOF);review=json.loads(PROOF.read_text());cpu=json.loads(CPU.read_text())
    expected={'status':'passed','independent_review':True,'scope':'NVML device/process sampler CPU-only review',
              'gpu_authorized':False,'GT_opened':False,'scoring_performed':False,'errors':[],
              'source_sha256':upstream.sources(),'cpu_sha256':rt.sha(CPU),'cpu_started_sha256':rt.sha(CPUMON/'started.json'),
              'cpu_terminal_sha256':rt.sha(CPUMON/'terminal.json'),'prior_review_sha256':prior['prior_review_sha256'],
              'reviewer_sha256':rt.sha(REVIEWER)}
    if any(review.get(k)!=v for k,v in expected.items()) or cpu.get('source_sha256_before')!=upstream.sources() or cpu.get('source_sha256_after')!=upstream.sources():
        raise ValueError('NVML CPU independent proof mismatch')
    return {'scope':'C-ON worker CPU preflight only','gpu_authorized':False,'source_sha256':sources(),
            'nvml_review_sha256':rt.sha(PROOF),'prior_review_sha256':prior['prior_review_sha256']}
if __name__=='__main__':raise SystemExit('locked: CPU gate cannot authorize GPU')
