"""CPU-only preparation gate; deliberately cannot authorize GPU execution."""
import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/'e4_c_memory_runtime'))
import runtime_admission as runtime
R=BASE/'e1_independent_review'
FAKE=R/'e4c_memory_fake_handshake001_review.json'
FAKE_REVIEWER=R/'review_e4c_memory_fake_handshake001.py'
FAKE_MON=R/'monitor_e4c_memory_fake_handshake001'
FAKE_RUN=BASE/'e4_c_memory_handshake/run_001'
FAKE_RUN_MON=BASE/'e4_c_memory_handshake/monitor_run001'
NAMES=('PLAN.md','preflight_plan.json','guard.py','controller.py','worker.py','gate.py',
       'preflight_gate.py','preflight_launch.py','preflight_entry.py','cpu_check.py','test_guard.py','test_worker.py')

def sources():
    source=runtime.source_hashes()
    source.update({str((HERE/x).resolve()):runtime.sha(HERE/x) for x in NAMES})
    return source

def admit_cpu():
    upstream=runtime.admit()
    runtime.supervisor(FAKE_MON,FAKE_REVIEWER)
    proof=json.loads(FAKE.read_text())
    if (proof.get('status')!='passed' or proof.get('independent_review') is not True
        or proof.get('scope')!='fake sampler, no GPU measurement' or proof.get('gpu_authorized') is not False
        or proof.get('GT_opened') is not False or proof.get('errors')!=[]
        or proof.get('samples_verified')!=14 or proof.get('fabricated_sample_bytes')!=1048576
        or proof.get('reviewer_sha256')!=runtime.sha(FAKE_REVIEWER)):
        raise ValueError('fake handshake review mismatch')
    evidence={'started_sha256':FAKE_RUN_MON/'started.json','terminal_sha256':FAKE_RUN_MON/'terminal.json',
              'sampler_started_sha256':FAKE_RUN/'sampler_monitor/started.json',
              'sampler_terminal_sha256':FAKE_RUN/'sampler_monitor/terminal.json'}
    if any(proof.get(k)!=runtime.sha(p) for k,p in evidence.items()):raise ValueError('fake handshake supervisor evidence changed')
    files=proof.get('files_sha256',{})
    required={'launch.json','exit.json','expected.json','worker_ready.json','release.json','stopped.json','stopped_inspect.json','samples.jsonl'}
    if set(files)!=required or any(runtime.sha(FAKE_RUN/p)!=h for p,h in files.items()):raise ValueError('fake handshake evidence changed')
    actual=sources()
    return {'scope':'CPU preparation only; GPU remains locked','gpu_authorized':False,
            'source_sha256':actual,'runtime_review_sha256':upstream['review_sha256'],
            'fake_review_sha256':runtime.sha(FAKE)}

if __name__=='__main__':raise SystemExit('locked: this gate cannot authorize GPU')
