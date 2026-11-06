"""Future reviewed+root-scheduled C-ON GPU gate; missing proofs fail closed."""
import json,time
from pathlib import Path
import host_admission,host_contract
HERE=host_admission.HERE;BASE=host_admission.BASE;R=host_admission.R;rt=host_admission.rt
CPU=HERE/'cpu_check005/verification.json';CPUMON=HERE/'monitor_cpu005'
REVIEW=R/'e4c_memory_host_prelaunch_review.json';REVIEWER=R/'review_e4c_memory_host_prelaunch.py';MON=R/'monitor_e4c_memory_host_prelaunch'
ALLOCATION=HERE/'allocation001.json'

def admit():
    upstream=host_admission.admit_cpu();rt.supervisor(CPUMON,HERE/'cpu_check.py')
    cpu=json.loads(CPU.read_text());actual=host_admission.sources()
    if (cpu.get('status')!='passed' or cpu.get('gpu_used') is not False or cpu.get('gpu_authorized') is not False
        or cpu.get('source_sha256_before')!=actual or cpu.get('source_sha256_after')!=actual):raise ValueError('host CPU proof mismatch')
    rt.supervisor(MON,REVIEWER,REVIEW);review=json.loads(REVIEW.read_text())
    expected={'status':'passed','independent_review':True,'scope':'C-ON memory host prelaunch CPU review',
              'gpu_authorized':False,'GT_opened':False,'scoring_performed':False,'errors':[],
              'source_sha256':actual,'cpu_sha256':rt.sha(CPU),'cpu_started_sha256':rt.sha(CPUMON/'started.json'),
              'cpu_terminal_sha256':rt.sha(CPUMON/'terminal.json'),'worker_preflight_review_sha256':upstream['worker_preflight_review_sha256'],
              'reviewer_sha256':rt.sha(REVIEWER)}
    if any(review.get(k)!=v for k,v in expected.items()):raise ValueError('host independent review mismatch')
    plan=json.loads((HERE/'host_plan.json').read_text());budget_path=(BASE/plan['budget_source']).resolve()
    if rt.sha(budget_path)!=plan['budget_source_sha256']:raise ValueError('user budget source changed')
    budget=json.loads(budget_path.read_text());allocation=json.loads(ALLOCATION.read_text())
    if allocation.get('prelaunch_review_sha256')!=rt.sha(REVIEW):raise ValueError('allocation review mismatch')
    host_contract.validate(plan,allocation,time.time(),budget)
    return {'scope':'C-ON isolated memory smoke','gpu_authorized':True,'stage':'smoke','mode':'C-ON',
            'source_sha256':actual,'allocation_sha256':rt.sha(ALLOCATION),'allocation':allocation,
            'prelaunch_review_sha256':rt.sha(REVIEW)}

if __name__=='__main__':raise SystemExit('locked: allocation and reviewed launcher required')
