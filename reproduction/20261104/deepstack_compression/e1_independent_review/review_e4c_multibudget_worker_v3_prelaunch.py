"""Independent prelaunch review of the 32-image multibudget timing pilot."""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parent
B = R.parent
D = B/'e4_c_multibudget_worker_v3'
A = B/'e4_c_multibudget_worker_accept_v3'
F = B/'e4_c_multibudget_references_accept_v3'
P = D/'run_preflight001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def terminal(folder,script):
    s,t=read(folder/'started.json'),read(folder/'terminal.json')
    if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or
        t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or
        (Path(s['cwd'])/s['command'][1]).resolve()!=script.resolve()):
        raise ValueError('supervisor mismatch '+str(folder))
    return s


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve()
    if out!=R/'e4c_multibudget_worker_v3_prelaunch_review.json':raise ValueError('unplanned output')
    plan_path=D/'runtime_plan001.json'
    p,c=read(plan_path),read(D/'execution_cpu001.json')
    sources={name:sha(name) for name in p['source_sha256']}
    if (len(sources)!=245 or sources!=p['source_sha256'] or sources!=c['source_sha256'] or
        p.get('stage')!='post-hoc 32-image 4-budget natural/fixed32 timing pilot' or p.get('jobs')!=1520 or
        p.get('GPU_authorized_by_plan') is not False or p.get('GT_opened') is not False or
        c.get('status')!='passed' or c.get('test_returncode')!=0 or 'Ran 21 tests' not in c.get('test_output','') or
        c.get('plan_sha256')!=sha(plan_path) or c.get('GPU_authorized') is not False):
        raise ValueError('frozen timing sources or CPU proof')
    terminal(D/'monitor_execution_cpu001',D/'freeze_execution.py')
    temp,tempcode,tempmon=(R/'e4c_multibudget_worker_v3_preflight_review.json',
        R/'review_e4c_multibudget_worker_v3_preflight.py',R/'monitor_e4c_multibudget_worker_v3_preflight')
    tr=read(temp);terminal(tempmon,tempcode)
    if (tr.get('status')!='passed' or tr.get('independent_review') is not True or
        tr.get('scope')!='multibudget_worker_v3_cpu_preflight_only' or tr.get('reviewer_sha256')!=sha(tempcode) or
        tr.get('plan_sha256')!=sha(plan_path) or tr.get('source_sha256')!=sources or tr.get('errors')!=[]):
        raise ValueError('temporary review mismatch')
    pm=D/'monitor_preflight001';st=terminal(pm,D/'launch.py')
    launch,ex,worker,proc=map(read,(P/'launch.json',P/'exit.json',P/'worker_preflight.json',P/'processor_preflight.json'))
    if (st.get('timeout_seconds')!=900 or st.get('grace_seconds')!=90 or '--preflight-only' not in st['command'] or
        st.get('worker_pid')!=launch['controller']['pid'] or launch.get('approval_bundle') is not None or
        launch.get('runtime_plan_sha256')!=sha(plan_path) or launch.get('worker_sha256')!=sha(D/'worker.py') or
        launch.get('launcher_sha256')!=sha(D/'launch.py') or ex.get('controller')!=launch['controller'] or
        ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or
        ex.get('interrupted') is not None or ex.get('container_stopped_verified') is not True or
        ex.get('client_exited_while_container_running') is not False):raise ValueError('no-GPU preflight exit')
    cmd=launch['command']
    if '--gpus' in cmd or '--device' in cmd or 'CUDA_VISIBLE_DEVICES=' not in cmd or '--rm' in cmd:
        raise ValueError('unexpected GPU or ephemeral container')
    cid=ex['container_identity']['container_id']
    doc=json.loads(subprocess.run(['docker','inspect',cid],check=True,capture_output=True,text=True,timeout=15).stdout)[0]
    state,host=doc['State'],doc['HostConfig']
    if (doc['Id']!=cid or doc['Image']!=ex['container_identity']['image'] or state['Running'] or state['Pid']!=0 or
        state['ExitCode']!=0 or state['OOMKilled'] or host['NetworkMode']!='none' or host['Devices'] or
        host['DeviceRequests'] or 'CUDA_VISIBLE_DEVICES=' not in doc['Config']['Env'] or
        doc['Config']['Cmd']!=[str(D/'worker.py'),'--out',str(P),'--preflight-only']):
        raise ValueError('actual Docker state or devices')
    if (worker.get('status')!='passed' or worker.get('plan_sha256')!=sha(plan_path) or
        worker.get('processor_preflight_sha256')!=sha(P/'processor_preflight.json') or
        worker.get('inputs')!=62 or worker.get('jobs')!=1520 or worker.get('cuda_initialized') is not False or
        worker.get('model_loaded') is not False or worker.get('GT_opened') is not False or
        proc.get('status')!='passed' or proc.get('inputs_checked')!=62 or len(proc.get('processor_inputs',[]))!=62 or
        proc.get('references_checked')!=248 or proc.get('jobs')!=1520 or proc.get('weight_payloads_hashed')!=4 or
        len(proc.get('weight_stats',{}))!=4 or proc.get('cuda_initialized') is not False or proc.get('model_loaded') is not False or
        proc.get('GT_opened') is not False or proc.get('GPU_authorized') is not False):
        raise ValueError('worker/processor preflight mismatch')
    formal,freview=F/'verification001.json',R/'e4c_multibudget_references_v3_result_review.json'
    fm,rm=F/'monitor_verify001',R/'monitor_e4c_multibudget_references_v3_result'
    terminal(fm,F/'verify.py');terminal(rm,R/'review_e4c_multibudget_references_v3_result.py')
    f,r=read(formal),read(freview)
    if (f.get('status')!='passed' or f.get('inputs')!=62 or f.get('generations')!=248 or
        f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('timing_claims') is not False or
        f.get('errors')!=[] or r.get('status')!='passed' or r.get('independent_review') is not True or
        r.get('inputs')!=62 or r.get('generations')!=248 or r.get('formal_sha256')!=sha(formal) or
        r.get('reviewer_sha256')!=sha(R/'review_e4c_multibudget_references_v3_result.py') or r.get('errors')!=[]):
        raise ValueError('reference evidence mismatch')
    spec=importlib.util.spec_from_file_location('timing_review_fixed_reference_gate',D/'fixed_reference_gate.py')
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    if len(gate.load())!=248:raise ValueError('reference inventory mismatch')
    ap=A/'verification_plan001.json';accept=read(ap);terminal(A/'monitor_freeze001',A/'freeze.py')
    if (accept.get('status')!='prepared' or accept.get('run_id')!='001' or accept.get('GPU_authorized') is not False or
        'Ran 3 tests' not in accept.get('tests','') or any(sha(path)!=digest for path,digest in accept['source_sha256'].items())):
        raise ValueError('formal timing acceptance plan mismatch')
    analysis_dir=B/'e4_c_multibudget_expansion'
    analysis_plan=analysis_dir/'ANALYSIS_PLAN.md'
    analysis_freeze=analysis_dir/'analysis_plan_freeze001.json'
    analysis_monitor=analysis_dir/'monitor_freeze_analysis001'
    af=read(analysis_freeze);terminal(analysis_monitor,analysis_dir/'freeze_analysis_plan.py')
    if (af.get('status')!='frozen' or af.get('analysis_plan_sha256')!=sha(analysis_plan) or
        af.get('pilot_plan_sha256')!=sha(analysis_dir/'pilot_plan001.json') or
        af.get('freeze_source_sha256')!=sha(analysis_dir/'freeze_analysis_plan.py') or
        af.get('n_images')!=32 or af.get('measured_jobs')!=1280 or af.get('warmup_jobs')!=240 or
        af.get('GT_opened') is not False or af.get('timing_results_opened') is not False):
        raise ValueError('analysis plan freeze mismatch')
    ac=B/'e4_c_multibudget_analysis'
    analysis_review=ac/'independent_prelaunch_review.json'
    analysis_reviewer=R/'review_multibudget_analysis_prelaunch.py'
    analysis_review_monitor=R/'monitor_multibudget_analysis_prelaunch'
    ar=read(analysis_review);terminal(analysis_review_monitor,analysis_reviewer)
    if (ar.get('status')!='passed' or ar.get('independent_review') is not True or
        ar.get('scope')!='multibudget_analysis_prelaunch' or ar.get('GT_opened') is not False or
        ar.get('timing_results_opened') is not False or ar.get('errors')!=[] or
        ar.get('reviewer_sha256')!=sha(analysis_reviewer)):
        raise ValueError('analysis independent prelaunch missing')
    agspec=importlib.util.spec_from_file_location('timing_review_analysis_gate',ac/'analysis_gate.py')
    ag=importlib.util.module_from_spec(agspec);agspec.loader.exec_module(ag)
    ag.prelaunch()
    env=B/'e4_c_multibudget_environment';ep=env/'plan001.json';e=read(ep)
    es,et=read(env/'monitor_cpu001/started.json'),read(env/'monitor_cpu001/terminal.json')
    if (et.get('reason')!='completed' or et.get('worker_returncode')!=0 or et.get('remaining_process_group')!=[] or
        es.get('worker_pid')!=et.get('worker_pid') or es['command'][1:4]!=['-m','unittest','discover'] or
        'test_validate.py' not in es['command']):
        raise ValueError('host observation test supervisor mismatch')
    terminal(env/'monitor_freeze001',env/'freeze_plan.py')
    if (e.get('GT_opened') is not False or e.get('scoring_performed') is not False or
        e.get('GPU_authorized') is not False or any(sha(path)!=digest for path,digest in e['source_sha256'].items()) or
        'Ran 2 tests' not in (env/'monitor_cpu001/stderr.log').read_text() or
        e['command'].count('--duration')!=1 or e['command'][e['command'].index('--duration')+1]!='7500' or
        e['command'].count('--interval')!=1 or e['command'][e['command'].index('--interval')+1]!='60'):
        raise ValueError('host observation plan mismatch')
    env_review=R/'e4c_multibudget_environment_prelaunch_review.json'
    env_reviewer=R/'review_e4c_multibudget_environment_prelaunch.py'
    env_review_monitor=R/'monitor_e4c_multibudget_environment_prelaunch'
    er=read(env_review);terminal(env_review_monitor,env_reviewer)
    if (er.get('status')!='passed' or er.get('independent_review') is not True or
        er.get('plan_sha256')!=sha(ep) or er.get('source_sha256')!=e['source_sha256'] or
        er.get('reviewer_sha256')!=sha(env_reviewer) or er.get('errors')!=[]):
        raise ValueError('host observation independent review mismatch')
    fields={'plan_sha256':plan_path,'execution_cpu_sha256':D/'execution_cpu001.json',
      'execution_cpu_started_sha256':D/'monitor_execution_cpu001/started.json','execution_cpu_terminal_sha256':D/'monitor_execution_cpu001/terminal.json',
      'temp_review_sha256':temp,'temp_review_started_sha256':tempmon/'started.json','temp_review_terminal_sha256':tempmon/'terminal.json',
      'worker_preflight_sha256':P/'worker_preflight.json','processor_preflight_sha256':P/'processor_preflight.json',
      'preflight_launch_sha256':P/'launch.json','preflight_exit_sha256':P/'exit.json',
      'preflight_started_sha256':pm/'started.json','preflight_terminal_sha256':pm/'terminal.json',
      'reference_formal_sha256':formal,'reference_formal_started_sha256':fm/'started.json','reference_formal_terminal_sha256':fm/'terminal.json',
      'reference_review_sha256':freview,'reference_review_started_sha256':rm/'started.json','reference_review_terminal_sha256':rm/'terminal.json',
      'acceptance_plan_sha256':ap,'acceptance_freeze_started_sha256':A/'monitor_freeze001/started.json',
      'acceptance_freeze_terminal_sha256':A/'monitor_freeze001/terminal.json','reviewer_sha256':Path(__file__),
      'analysis_plan_sha256':analysis_plan,'analysis_freeze_sha256':analysis_freeze,
      'analysis_freeze_started_sha256':analysis_monitor/'started.json','analysis_freeze_terminal_sha256':analysis_monitor/'terminal.json',
      'analysis_code_plan_sha256':ac/'analysis_code_plan004.json','analysis_cpu_proof_sha256':ac/'cpu_proof004.json',
      'analysis_cpu_started_sha256':ac/'monitor_cpu004/started.json','analysis_cpu_terminal_sha256':ac/'monitor_cpu004/terminal.json',
      'analysis_review_sha256':analysis_review,'analysis_review_started_sha256':analysis_review_monitor/'started.json',
      'analysis_review_terminal_sha256':analysis_review_monitor/'terminal.json','analysis_reviewer_sha256':analysis_reviewer,
      'environment_plan_sha256':ep,'environment_freeze_started_sha256':env/'monitor_freeze001/started.json',
      'environment_freeze_terminal_sha256':env/'monitor_freeze001/terminal.json',
      'environment_review_sha256':env_review,'environment_review_started_sha256':env_review_monitor/'started.json',
      'environment_review_terminal_sha256':env_review_monitor/'terminal.json','environment_reviewer_sha256':env_reviewer}
    result={'status':'passed','independent_review':True,'scope':'multibudget_worker_v3_gpu_prelaunch',
      'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'source_sha256':sources,'errors':[],
      'container_id':cid,'limitations':['Post-hoc 32-image DEV timing pilot; no generalization or quality claim.',
        'First-token and completion boundaries intentionally synchronize CUDA; completion excludes output fsync and decoding.',
        'All raw samples and health window overlaps must be retained and independently reviewed after completion.',
        'External host processes and monitoring can perturb timing; record and disclose observed activity.',
        'Existing DEV predictions read for input and same-arm identity; no new GT.']}
    result.update({key:sha(path) for key,path in fields.items()})
    with out.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


if __name__=='__main__':main()
