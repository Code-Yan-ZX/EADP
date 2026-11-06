"""Admit only C-ON no-GPU worker preflight after independent review."""
import json
import cpu_gate
HERE=cpu_gate.HERE;R=cpu_gate.R;rt=cpu_gate.rt
CPU=HERE/'cpu_check002/verification.json';CPUMON=HERE/'monitor_cpu002'
REVIEW=R/'e4c_memory_smoke_worker_preflight_review.json'
REVIEWER=R/'review_e4c_memory_smoke_worker_preflight.py'
MON=R/'monitor_e4c_memory_smoke_worker_preflight'
def admit():
    upstream=cpu_gate.admit_cpu();rt.supervisor(CPUMON,HERE/'cpu_check.py');cpu=json.loads(CPU.read_text());actual=cpu_gate.sources()
    if (cpu.get('status')!='passed' or cpu.get('gpu_used') is not False or cpu.get('gpu_authorized') is not False
        or cpu.get('source_sha256_before')!=actual or cpu.get('source_sha256_after')!=actual):raise ValueError('smoke CPU source proof mismatch')
    rt.supervisor(MON,REVIEWER,REVIEW);r=json.loads(REVIEW.read_text())
    expected={'status':'passed','independent_review':True,'scope':'C-ON memory worker CPU preflight only',
              'gpu_authorized':False,'GT_opened':False,'scoring_performed':False,'errors':[],
              'source_sha256':actual,'cpu_sha256':rt.sha(CPU),'cpu_started_sha256':rt.sha(CPUMON/'started.json'),
              'cpu_terminal_sha256':rt.sha(CPUMON/'terminal.json'),'nvml_review_sha256':upstream['nvml_review_sha256'],
              'reviewer_sha256':rt.sha(REVIEWER)}
    if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('smoke independent preflight review mismatch')
    return {'scope':'C-ON memory worker CPU preflight only','gpu_authorized':False,'source_sha256':actual,
            'review_sha256':rt.sha(REVIEW),'nvml_review_sha256':upstream['nvml_review_sha256']}
if __name__=='__main__':raise SystemExit('locked: no GPU authority')
