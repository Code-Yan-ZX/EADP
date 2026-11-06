"""Fail closed pending CPU-only independent review; no GPU authority."""
import json
import gate

R=gate.R;HERE=gate.HERE;rt=gate.runtime
CPU=HERE/'cpu_check004/verification.json';CPUMON=HERE/'monitor_cpu004'
REVIEW=R/'e4c_memory_real_worker_preflight_review.json'
REVIEWER=R/'review_e4c_memory_real_worker_preflight.py'
REVIEWMON=R/'monitor_e4c_memory_real_worker_preflight'

def admit():
    upstream=gate.admit_cpu();rt.supervisor(CPUMON,HERE/'cpu_check.py')
    cpu=json.loads(CPU.read_text());actual=gate.sources()
    if (cpu.get('status')!='passed' or cpu.get('gpu_used') is not False or cpu.get('gpu_authorized') is not False
        or cpu.get('source_sha256_before')!=actual or cpu.get('source_sha256_after')!=actual):raise ValueError('CPU source proof mismatch')
    rt.supervisor(REVIEWMON,REVIEWER,REVIEW);review=json.loads(REVIEW.read_text())
    expected={'status':'passed','independent_review':True,'scope':'real NVML CPU worker preflight only',
              'gpu_authorized':False,'GT_opened':False,'scoring_performed':False,'errors':[],
              'source_sha256':actual,'cpu_sha256':rt.sha(CPU),'cpu_started_sha256':rt.sha(CPUMON/'started.json'),
              'cpu_terminal_sha256':rt.sha(CPUMON/'terminal.json'),'fake_review_sha256':upstream['fake_review_sha256'],
              'reviewer_sha256':rt.sha(REVIEWER)}
    if any(review.get(k)!=v for k,v in expected.items()):raise ValueError('independent CPU preflight review mismatch')
    return {'scope':'real NVML CPU worker preflight only','gpu_authorized':False,
            'source_sha256':actual,'cpu_sha256':rt.sha(CPU),'review_sha256':rt.sha(REVIEW),
            'fake_review_sha256':upstream['fake_review_sha256']}

if __name__=='__main__':raise SystemExit('locked: CPU preflight gate cannot authorize GPU')
