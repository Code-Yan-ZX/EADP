"""Fail-closed B admission; host additionally checks the live panel container."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;E4=ROOT/'e4';ANALYSIS=ROOT/'e4_analysis'
sys.path.insert(1,str(E4))
from identity_worker import sha

APPROVAL=HERE/'b_approval.json'
PANEL_FIELDS=('run','panel_proof','verify_started','verify_terminal','run_started','run_terminal',
              'e1_acceptance','e1_started','e1_terminal','review_proof','review_started','review_terminal')
SOURCES=('README.md','b_gate.py','b_imports.py','b_worker.py','b_launch.py','b_verify.py','b_preflight.py','b_import_regression.py','budget_plan.json',
         'arm.py','off_capture.py','pair_contract.py','verify_records.py','cpu_checks.py','test_arm.py',
         'test_b_launch.py','test_b_verify.py','test_b_gate.py','test_off_capture.py','test_pair_contract.py','test_verify_records.py')

def _bind(name,path):
    path=path.resolve()
    if name in sys.modules:
        module=sys.modules[name]
        if Path(module.__file__).resolve()!=path:raise ValueError('B dependency module identity mismatch: '+name)
        return module
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module

def _analysis():
    # The downstream audited E4 code uses these unqualified names.  Bind each
    # to its exact source, and reject any previously cached conflicting module.
    sys.path.insert(0,str(E4))
    e1=ROOT/'e1_analysis'
    for name,path in (('run_cpu_checks',E4/'run_cpu_checks.py'),('completion',e1/'completion.py'),
                      ('metrics',e1/'metrics.py'),('paired_stats',e1/'paired_stats.py'),('score',e1/'score.py'),
                      ('paired',ANALYSIS/'paired.py'),('panel_gate',E4/'panel_gate.py'),('smoke_launch',E4/'smoke_launch.py')):
        _bind(name,path)
    key='eadp_b_panel_scoring_gate';path=ANALYSIS/'score_e4.py'
    if key in sys.modules:
        module=sys.modules[key]
        if Path(module.__file__).resolve()!=path.resolve():raise ValueError('B panel gate module identity mismatch')
        return module
    spec=importlib.util.spec_from_file_location(key,path);module=importlib.util.module_from_spec(spec);sys.modules[key]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[key]
        raise
    return module

def _read(path):return json.loads(Path(path).read_text())

def check_approval(*,preflight_only=False):
    if not APPROVAL.is_file():raise ValueError('E4-B locked: approval bundle absent')
    bundle=_read(APPROVAL)
    if set(bundle)!={'cpu_proof','cpu_started','cpu_terminal','b_review_proof','panel'} or set(bundle['panel'])!=set(PANEL_FIELDS):
        raise ValueError('E4-B approval bundle incomplete')
    cpu=Path(bundle['cpu_proof']);review=Path(bundle['b_review_proof']);c=_read(cpu);r=_read(review)
    if (c.get('status')!='passed' or c.get('returncode')!=0 or c.get('timeout') is not False or
        r.get('status')!='passed' or r.get('independent_review') is not True or r.get('cpu_proof_sha256')!=sha(cpu)):
        raise ValueError('E4-B source review failed')
    for name in SOURCES:
        path=HERE/name;digest=sha(path);key=str(path)
        if (c.get('source_sha256_before',{}).get(key)!=digest or c.get('source_sha256_after',{}).get(key)!=digest or
            r.get('source_sha256',{}).get(key)!=digest):raise ValueError('E4-B source not reviewed: '+key)
    for key,digest in c['source_sha256_after'].items():
        if c['source_sha256_before'].get(key)!=digest or r.get('source_sha256',{}).get(key)!=digest or sha(key)!=digest:
            raise ValueError('E4-B dependency changed: '+key)
    ip=HERE/'b_import_preflight_002.json'
    if not ip.is_file() or r.get('import_preflight_sha256')!=sha(ip):raise ValueError('E4-B container import preflight missing')
    iv=_read(ip)
    if (iv.get('status')!='passed' or iv.get('returncode')!=0 or iv.get('container_running') is not False or
        iv.get('container_exit_code')!=0 or iv.get('timed_out') is not False or
        iv.get('worker_sha256')!=sha(HERE/'b_worker.py') or iv.get('gate_sha256')!=sha(HERE/'b_gate.py') or
        iv.get('regression_sha256')!=sha(HERE/'b_import_regression.py') or
        iv.get('result',{}).get('cuda_initialized') is not False):raise ValueError('E4-B container import preflight failed')
    pp=bundle['panel'];p=Path(pp['panel_proof']);panel=_read(p);score=_analysis()
    score.validate_supervisor(Path(bundle['cpu_started']),Path(bundle['cpu_terminal']),HERE/'cpu_checks.py',cpu.parent)
    if (r.get('cpu_started_sha256')!=sha(bundle['cpu_started']) or r.get('cpu_terminal_sha256')!=sha(bundle['cpu_terminal'])):
        raise ValueError('E4-B CPU supervision not reviewed')
    score.validate_supervisor(Path(pp['verify_started']),Path(pp['verify_terminal']),ROOT/'e4_panel_v2/panel_verify.py',p)
    if (panel.get('status')!='passed' or panel.get('stage')!='full' or panel.get('n_questions')!=384 or
        panel.get('all_arms_validated')!=3072 or panel.get('GT_opened') is not False or panel.get('scoring_performed') is not False or
        panel.get('verifier_sha256')!=sha(ROOT/'e4_panel_v2/panel_verify.py')):raise ValueError('E4 panel acceptance missing')
    score.validate_independent_review(Path(pp['review_proof']),Path(pp['review_started']),Path(pp['review_terminal']),
                                      p,Path(pp['verify_started']),Path(pp['verify_terminal']))
    if r.get('panel_proof_sha256')!=sha(p) or r.get('panel_review_sha256')!=sha(pp['review_proof']):
        raise ValueError('E4-B review bound to a different panel')
    required=[Path(pp[x]) for x in ('e1_acceptance','e1_started','e1_terminal','run_started','run_terminal')]
    run=Path(pp['run']).resolve();required += [run/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','records.jsonl','generated.jsonl')]
    files=panel.get('files_sha256',{})
    if any(files.get(str(x.resolve()))!=sha(x) for x in required):raise ValueError('E4 accepted panel artifact missing or changed')
    for path,digest in files.items():
        if sha(path)!=digest:raise ValueError('E4 panel artifact changed')
    if not preflight_only:
        pf=HERE/'b_container_preflight_002.json'
        if not pf.is_file() or r.get('preflight_sha256')!=sha(pf):raise ValueError('E4-B full preflight review missing')
        d=_read(pf)
        ps=HERE/'monitor_bpreflight002/started.json';pt=HERE/'monitor_bpreflight002/terminal.json'
        if (not ps.is_file() or not pt.is_file() or r.get('preflight_started_sha256')!=sha(ps) or
            r.get('preflight_terminal_sha256')!=sha(pt)):raise ValueError('E4-B preflight supervisor proof missing')
        started=_read(ps);terminal=_read(pt);cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
        if (terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or
            terminal.get('worker_pid')!=started.get('worker_pid') or started.get('timeout_seconds')!=900 or
            [(cwd/x).resolve() for x in cmd if Path(x).name=='b_preflight.py']!=[(HERE/'b_preflight.py').resolve()]):
            raise ValueError('E4-B preflight supervision failed')
        if (d.get('status')!='passed' or d.get('returncode')!=0 or d.get('container_running') is not False or
            d.get('container_exit_code')!=0 or d.get('timed_out') is not False or d.get('worker_sha256')!=sha(HERE/'b_worker.py') or
            d.get('host_script_sha256')!=sha(HERE/'b_preflight.py') or d.get('result',{}).get('cuda_initialized') is not False):
            raise ValueError('E4-B container preflight failed')
    return {'approval_sha256':sha(APPROVAL),'cpu_sha256':sha(cpu),'b_review_sha256':sha(review),
            'panel_proof_sha256':sha(p),'panel_review_sha256':sha(pp['review_proof'])}

def host_validate_panel():
    """Run the full no-GT gate on host, including live Docker inspect."""
    bundle=_read(APPROVAL);score=_analysis();args=SimpleNamespace(**{k:Path(v) for k,v in bundle['panel'].items()})
    score.gate(args)
    return {'panel_proof_sha256':sha(args.panel_proof),'panel_review_sha256':sha(args.review_proof),
            'panel_gate_source_sha256':sha(ANALYSIS/'score_e4.py'),'host_live_docker_inspection':True}
