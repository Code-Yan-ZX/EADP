"""Fail-closed local approvals for the exploratory E2/E3 smoke draft."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;E4=BASE/'e4';CONTRACT=BASE/'e2_e3_contract'
APPROVAL=HERE/'approval.json'
SOURCES=[HERE/x for x in ('gate.py','local_imports.py','launch.py','preflight.py','smoke_worker.py','intervention_runner.py','cache_fingerprints.py','verify_interventions.py','verify_smoke.py','accept.py','test_verify_interventions.py','test_cache_fingerprints.py','test_gate.py','test_preflight.py','test_accept.py','cpu_checks.py')]
SOURCES += [CONTRACT/x for x in ('model_adapter.py','interventions.py','test_model_adapter.py','test_interventions.py')]
SOURCES += [E4/x for x in ('smoke_contract.py','budget_adapter.py','decode_observer.py','off_adapter.py','skip_deepstack.py','identity_records.py','identity_worker.py')]
UPSTREAM_PROOF=BASE/'e4_native/cpu_check_002/verification.json'
SOURCES += [Path(x) for x in json.loads(UPSTREAM_PROOF.read_text())['source_sha256_after']]
SOURCES += [UPSTREAM_PROOF,BASE/'initial_plan.json']
SOURCES=list(dict.fromkeys(p.resolve() for p in SOURCES))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def check_approval(preflight=False):
    if not APPROVAL.is_file():raise ValueError('independent approval is absent')
    a=json.loads(APPROVAL.read_text())
    if set(a)!={'cpu_proof','review_proof'}:raise ValueError('approval fields')
    cpu=Path(a['cpu_proof']).resolve();review=Path(a['review_proof']).resolve()
    c=json.loads(cpu.read_text());r=json.loads(review.read_text())
    if c.get('status')!='passed' or c.get('returncode')!=0 or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('cpu_proof_sha256')!=sha(cpu):raise ValueError('CPU/review proof')
    for path in SOURCES:
        h=sha(path);key=str(path.resolve())
        if c.get('source_sha256_before',{}).get(key)!=h or c.get('source_sha256_after',{}).get(key)!=h or r.get('source_sha256',{}).get(key)!=h:raise ValueError('unreviewed source '+key)
    if not preflight:
        fp=HERE/'container_full_preflight_002.json'
        if not fp.is_file() or r.get('full_preflight_sha256')!=sha(fp):raise ValueError('container preflight absent')
        f=json.loads(fp.read_text())
        started=HERE/'monitor_preflight002/started.json';terminal=HERE/'monitor_preflight002/terminal.json'
        if not started.is_file() or not terminal.is_file() or r.get('preflight_started_sha256')!=sha(started) or r.get('preflight_terminal_sha256')!=sha(terminal):raise ValueError('preflight supervisor proof absent')
        s=json.loads(started.read_text());t=json.loads(terminal.read_text());cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
        if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or
            t.get('worker_pid')!=s.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name=='preflight.py']!=[(HERE/'preflight.py').resolve()]):raise ValueError('preflight supervisor failed')
        if (f.get('status')!='passed' or f.get('container_running') is not False or f.get('container_exit_code')!=0 or f.get('returncode')!=0 or
            f.get('timed_out') is not False or f.get('worker_sha256')!=sha(HERE/'smoke_worker.py') or f.get('host_source_sha256')!=sha(HERE/'preflight.py') or
            f.get('launch_sha256')!=sha(HERE/'preflight_002/launch.json') or f.get('stdout_sha256')!=sha(HERE/'preflight_002/stdout.log') or
            f.get('stderr_sha256')!=sha(HERE/'preflight_002/stderr.log')):raise ValueError('container preflight failed')
    return {'approval_sha256':sha(APPROVAL),'cpu_sha256':sha(cpu),'review_sha256':sha(review)}
