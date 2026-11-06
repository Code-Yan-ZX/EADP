"""Fail-closed admission for a one-image, no-GT gradient feasibility run."""
import importlib.util
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;E4=ROOT/'e4';V2=ROOT/'e4_panel_v2';ORACLE=ROOT/'m3_oracle'
sys.path[:0]=[str(E4),str(V2)]
from panel_v2_imports import load_local
from local_imports import local
sha=local('contract').sha
APPROVAL=HERE/'approval.json'
LOCAL_SOURCES=('plan.json','contract.py','core.py','gate.py','worker.py','launcher.py','verify.py','preflight.py','local_imports.py',
               'cpu_checks.py','cpu_tests.py','review_oracle.py','test_contract.py','test_core.py','test_gate.py','test_launcher.py')

def _read(p):return json.loads(Path(p).read_text())

def _supervisor(started,terminal,script,output=None):
    s=_read(started);t=_read(terminal);cwd=Path(s.get('cwd','/'));cmd=s.get('command',[])
    if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or
        not isinstance(s.get('worker_pid'),int) or s['worker_pid']<=0 or t.get('worker_pid')!=s.get('worker_pid') or
        [(cwd/x).resolve() for x in cmd if Path(x).name==Path(script).name]!=[Path(script).resolve()]):
        raise ValueError('supervised evidence not completed: '+str(terminal))
    if output is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=Path(output).resolve()):
        raise ValueError('wrong supervised output')

def upstream(*,host=False):
    sample=ORACLE/'sample_001.json';sample_review=ORACLE/'sample_001_independent_review.json'
    sr=_read(sample_review)
    if (sr.get('status')!='passed' or sr.get('questions_verified')!=128 or sr.get('errors')!=[] or
        sr.get('sample_sha256')!=sha(sample) or sr.get('reviewer_sha256')!=sha(ORACLE/'review_sample.py') or
        sr.get('GT_opened') is not False or sr.get('predictions_opened') is not False):raise ValueError('M3 sample review failed')
    _supervisor(ORACLE/'monitor_review_sample001/started.json',ORACLE/'monitor_review_sample001/terminal.json',ORACLE/'review_sample.py')
    proof=HERE/'oracle_review_001.json';r=_read(proof)
    if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('GT_opened') is not False or
        r.get('real_model_loaded') is not False or r.get('gpu_used') is not False or r.get('reviewer_sha256')!=sha(HERE/'review_oracle.py') or
        r.get('upstream_proof_sha256')!=sha(ORACLE/'cpu_validation001.json')):raise ValueError('independent oracle review failed')
    _supervisor(HERE/'monitor_oracle_review001/started.json',HERE/'monitor_oracle_review001/terminal.json',HERE/'review_oracle.py')
    for path,digest in r['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('oracle source changed: '+path)
    if host:
        key='run_cpu_checks';path=(E4/'run_cpu_checks.py').resolve()
        if key in sys.modules:
            if Path(sys.modules[key].__file__).resolve()!=path:raise ValueError('conflicting E4 CPU dependency')
        else:
            spec=importlib.util.spec_from_file_location(key,path);m=importlib.util.module_from_spec(spec);sys.modules[key]=m;spec.loader.exec_module(m)
    native=load_local('native_panel_gate').check_native(inspect_live=host)
    return {'sample_sha256':sha(sample),'sample_review_sha256':sha(sample_review),'oracle_review_sha256':sha(proof),
            'native_proof_sha256':native['proof_sha256'],'native_review_sha256':native['review_sha256']}

def _review_binding(b,r,preflight_only):
    stage='preflight' if preflight_only else 'final'
    expected=ROOT/'e1_independent_review'/('review_m3gpu002_'+stage+'.py')
    reviewer=Path(r['reviewer_path']);mon=ROOT/'e1_independent_review'/('monitor_m3gpu002_'+stage)
    if (Path(b['review_started']).resolve()!=(mon/'started.json').resolve() or Path(b['review_terminal']).resolve()!=(mon/'terminal.json').resolve()):raise ValueError('wrong independent review supervisor')
    if reviewer.resolve()!=expected.resolve() or Path(b['review_proof']).resolve()!=(HERE/('independent_review_002_'+stage+'.json')).resolve():raise ValueError('wrong independent reviewer/proof')
    _supervisor(b['review_started'],b['review_terminal'],reviewer, b['review_proof'])
    if r.get('reviewer_sha256')!=sha(reviewer):raise ValueError('independent review source not bound')

def _check_preflight_evidence(p):
    d=_read(p);out=HERE/'preflight_001';launch=out/'launch.json';receipt=out/'host_upstream.json'
    expected={'preflight_sha256':HERE/'preflight.py','launch_sha256':launch,'receipt_sha256':receipt,
              'stdout_sha256':out/'stdout.log','stderr_sha256':out/'stderr.log','worker_sha256':HERE/'worker.py'}
    if any(d.get(k)!=sha(v) for k,v in expected.items()):raise ValueError('preflight evidence changed')
    l=_read(launch);cmd=l.get('command',[]);rl=l.get('host_upstream',{});result=d.get('result',{})
    if (rl!={'path':str(receipt.resolve()),'sha256':sha(receipt)} or _read(receipt)!=upstream(host=False) or
        cmd.count(str(HERE/'worker.py'))!=1 or cmd.count('--preflight-only')!=1 or cmd.count('--execute')!=1 or
        cmd.count('--out')!=1 or cmd[cmd.index('--out')+1]!=str(out) or
        cmd.count('--name')!=1 or cmd[cmd.index('--name')+1]!='eadp-m3-preflight-001' or
        any('/dev/nvidia' in str(x) for x in cmd) or 'CUDA_VISIBLE_DEVICES=' not in cmd or
        d.get('status')!='passed' or d.get('returncode')!=0 or d.get('container_running') is not False or
        d.get('container_exit_code')!=0 or d.get('timed_out') is not False or not d.get('container_id') or
        result.get('status')!='preflight_passed' or result.get('cuda_initialized') is not False or
        result.get('model_loaded') is not False or result.get('GT_opened') is not False or result.get('predictions') is not False or
        result.get('target_ids')!=[100,200,300]):raise ValueError('M3 container preflight failed')

def check_approval(*,preflight_only=False):
    if not APPROVAL.is_file():raise ValueError('M3 GPU smoke locked: independent approval missing')
    b=_read(APPROVAL)
    if set(b)!={'cpu_proof','cpu_started','cpu_terminal','review_proof','review_started','review_terminal'}:raise ValueError('incomplete M3 approval bundle')
    cpu=Path(b['cpu_proof']);c=_read(cpu);r=_read(b['review_proof'])
    _review_binding(b,r,preflight_only)
    _supervisor(b['cpu_started'],b['cpu_terminal'],HERE/'cpu_checks.py',cpu.parent)
    if (c.get('status')!='passed' or c.get('returncode')!=0 or c.get('timeout') is not False or
        r.get('status')!='passed' or r.get('independent_review') is not True or r.get('cpu_proof_sha256')!=sha(cpu) or
        r.get('cpu_started_sha256')!=sha(b['cpu_started']) or r.get('cpu_terminal_sha256')!=sha(b['cpu_terminal'])):
        raise ValueError('M3 CPU/independent review invalid')
    for path,digest in c['source_sha256_after'].items():
        if c['source_sha256_before'].get(path)!=digest or r.get('source_sha256',{}).get(path)!=digest or sha(path)!=digest:
            raise ValueError('M3 reviewed dependency changed: '+path)
    if c.get('source_sha256_before')!=c.get('source_sha256_after') or any(
        c['source_sha256_after'].get(str(HERE/x))!=sha(HERE/x) for x in LOCAL_SOURCES):
        raise ValueError('M3 complete reviewed source inventory missing')
    for path,digest in _read(HERE/'oracle_review_001.json')['source_sha256'].items():
        if c['source_sha256_after'].get(path)!=digest:raise ValueError('M3 oracle source not in reviewed bundle')
    native=upstream(host=False)
    if (r.get('oracle_review_sha256')!=native['oracle_review_sha256'] or
        r.get('native_review_sha256')!=native['native_review_sha256'] or r.get('sample_review_sha256')!=native['sample_review_sha256']):
        raise ValueError('M3 review binds different upstream evidence')
    if not preflight_only:
        p=HERE/'container_preflight_001.json';s=HERE/'monitor_preflight001/started.json';t=HERE/'monitor_preflight001/terminal.json'
        if r.get('preflight_sha256')!=sha(p) or r.get('preflight_started_sha256')!=sha(s) or r.get('preflight_terminal_sha256')!=sha(t):
            raise ValueError('M3 full preflight proof not reviewed')
        _supervisor(s,t,HERE/'preflight.py')
        _check_preflight_evidence(p)
    return {'approval_sha256':sha(APPROVAL),'cpu_sha256':sha(cpu),'review_sha256':sha(b['review_proof']),'upstream':native}
