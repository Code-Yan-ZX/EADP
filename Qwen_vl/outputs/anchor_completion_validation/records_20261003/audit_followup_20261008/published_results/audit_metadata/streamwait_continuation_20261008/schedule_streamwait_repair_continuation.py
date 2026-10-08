"""Independent frozen continuation; prepare does not register GPU/publish scope."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent
ROOT=Path('/media/disk2/YZX/research/EADP_amp')
DIAG=OUT.parent/'next_gap_diagnosis_20261008'
DEPENDENCY_PLAN=DIAG/'stability/next_text_full_streamwait.controller.plan.json'
DEPENDENCY_STATE=DIAG/'stability/next_text_full_streamwait.controller.state.json'
DEPENDENCY_MARKER=DIAG/'stability/next_text_full_streamwait.controller.finished.json'
WORKER=OUT/'streamwait_repair_worker.py'
PLAN=OUT/'streamwait_repair_continuation.plan.json'
STATE=OUT/'streamwait_repair_continuation.state.json'

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded

FROZEN=module(DIAG/'schedule_next_full_controls.py','prior_frozen_control_helpers')
read,sha,atomic,now,process=FROZEN.read,FROZEN.sha,FROZEN.atomic,FROZEN.now,FROZEN.process

GROUP_SPECS=[
 ('next_textvqa_K64_streamwait','textvqa',64,'pair',['next_text_E64','next_text_AZ64']),
 ('next_textvqa_K128_streamwait','textvqa',128,'pair',['next_text_E128','next_text_AZ128']),
 ('next_sqa_K32_streamwait','sqa',32,'pair',['next_sqa_E32','next_sqa_AZ32']),
 ('next_sqa_K64_streamwait','sqa',64,'pair',['next_sqa_E64','next_sqa_AZ64']),
 ('next_sqa_K128_streamwait','sqa',128,'pair',['next_sqa_E128','next_sqa_AZ128']),
 ('next_textvqa_FULL_streamwait','textvqa',0,'FULL',['next_text_FULL']),
 ('next_sqa_FULL_streamwait','sqa',0,'FULL',['next_sqa_FULL']),
]

def worker_path(plan):
    path=Path(plan['worker_path']).resolve()
    if path.parent!=OUT or path.name not in ('streamwait_repair_worker.py','legacy_streamwait_worker.py'):
        raise ValueError('Unapproved continuation worker path')
    return path

def verify(plan):
    if sha(worker_path(plan))!=plan['worker_sha256']:raise ValueError('Frozen continuation worker drift')
    for filename,expected in plan['source_and_input_sha256'].items():
        if sha(filename)!=expected:raise ValueError('Frozen continuation source/input drift: '+filename)
    if sha(__file__)!=plan['controller_sha256']:raise ValueError('Frozen continuation controller drift')

def protected(job):
    # Prepared immutable protocols and image manifests are permitted. GPU and
    # score artifacts are always preserved, including empty prior log files.
    for name,path in job['output_paths'].items():
        if name in ('control_protocol','protocol','manifest','image_manifest'):continue
        if Path(path).exists():raise FileExistsError('Preserve existing continuation artifact: '+str(path))

def dependencies():
    state=read(DEPENDENCY_STATE);expected={'EADP_beta2','AZ_beta2','EADP_beta1'}
    marker=read(DEPENDENCY_MARKER) if DEPENDENCY_MARKER.is_file() else {}
    job_pids=[job.get('pid') for job in state.get('jobs',{}).values()]
    ready=(state.get('status')=='complete' and set(state.get('registered_methods',[]))==expected
           and state.get('child_pids')==[] and not process(state.get('supervisor_pid'))['alive']
           and all(not process(pid)['alive'] for pid in job_pids)
           and all(state.get('jobs',{}).get(arm,{}).get('status')=='complete' for arm in expected)
           and marker.get('success') is True and marker.get('complete') is True
           and set(marker.get('completed',{}))==expected and FROZEN.prerequisites()['ready'])
    return dict(ready=ready,prior_controller_status=state.get('status'),prior_supervisor_pid=state.get('supervisor_pid'),
                prior_child_pids=state.get('child_pids'),all_three_registered_and_scored=all(state.get('jobs',{}).get(arm,{}).get('status')=='complete' for arm in expected))

def validate_dependency():
    if not dependencies()['ready']:raise ValueError('Existing three controls/main queues have not completed and exited')
    plan=read(DEPENDENCY_PLAN)
    reports={arm:FROZEN.validate_arm(plan['jobs'][arm],plan) for arm in ('EADP_beta2','AZ_beta2','EADP_beta1')}
    return dict(success=True,validated_utc=now(),marker_sha256=sha(DEPENDENCY_MARKER),reports=reports)

def launch_gate(job,gpu,active,progress,dependency_ready):
    result=FROZEN.gate(gpu,active,progress,dependency_ready)
    reasons=list(result['reasons'])
    if job['exclusive_gpu']:
        if active or result['contexts']:reasons.append('FULL_requires_true_exclusive_GPU_no_running_or_loading_model')
    required=job['min_free_memory_mib']
    if gpu['free_mib']<required and 'insufficient_free_GPU_memory' not in reasons:reasons.append('insufficient_free_GPU_memory')
    result.update(ready=not reasons,reasons=reasons,minimum_free_mib=required,exclusive_gpu=job['exclusive_gpu'])
    return result

def has_progress(job):
    # ScienceQA can carry runtime counts in prediction metadata instead of a
    # separate runtime stream. Neither an empty file nor worker.state counts.
    return any(FROZEN.first_json(path) for key in ('prediction','runtime')
               if (path:=job['output_paths'].get(key)))

def fairness(group,jobs):
    selected=[jobs[key] for key in group['job_ids']]
    for job in selected:
        if job['parameters'].get('beta')!=2. or job['parameters'].get('alpha')!=.5:
            raise ValueError('Continuation must preserve beta2/alpha.5')
        if job['parameters'].get('visual_token_num')!=group['budget']:raise ValueError('Frozen paired budget differs')
        if job['n']!=(5000 if group['task']=='textvqa' else 2017):raise ValueError('Frozen task denominator differs')
    if group['kind']=='pair':
        values=[]
        for job in selected:
            parameter=dict(job['parameters']);parameter.pop('anchorzip',None);values.append(parameter)
        if values[0]!=values[1] or selected[0]['n']!=selected[1]['n']:raise ValueError('Paired native input/generation parameters differ')

def prepare():
    worker=module(WORKER,'new_continuation_prepare_only')
    specs=worker.job_specs();jobs={};sources={};groups=[]
    for group_id,task,budget,kind,job_ids in GROUP_SPECS:
        group=dict(group_id=group_id,model='next',task=task,budget=budget,protocol_label='caller_stream_wait_only_beta2',
                   kind=kind,arms=['EADP','AnchorZip'] if kind=='pair' else ['FULL'],job_ids=job_ids)
        for job_id in job_ids:
            spec=specs[job_id];protocol,paths=worker.prepare(job_id,attempt=1)
            paths={key:str(value) for key,value in paths.items()}
            protocol_path=paths.get('control_protocol',paths.get('protocol'))
            if not protocol_path:raise ValueError('Worker has no immutable control protocol path')
            mapping=protocol.get('source_and_input_sha256')
            if not isinstance(mapping,dict):raise ValueError('Worker has no frozen source/input mapping')
            for filename,expected in mapping.items():
                if filename in sources and sources[filename]!=expected:raise ValueError('Conflicting frozen source')
                sources[filename]=expected
            sources[protocol_path]=sha(protocol_path)
            for path_key,hash_key in [('image_manifest','image_manifest_sha256'),('manifest','manifest_sha256')]:
                if protocol.get(path_key):sources[protocol[path_key]]=protocol[hash_key]
            job=dict(job_id=job_id,key=spec.get('key',job_id),model='next',task=task,budget=budget,
                     method=group['arms'][job_ids.index(job_id)],n=5000 if task=='textvqa' else 2017,attempt=1,
                     control_protocol=protocol_path,control_protocol_sha256=sha(protocol_path),parameters=protocol['parameters'],
                     output_paths=paths,generate_command=protocol['generate_command'],
                     min_free_memory_mib=32000 if kind=='FULL' else 22500,exclusive_gpu=kind=='FULL')
            protected(job);jobs[job_id]=job
        fairness(group,jobs);groups.append(group)
    sources[str(WORKER)]=sha(WORKER);sources[str(DIAG/'schedule_next_full_controls.py')]=sha(DIAG/'schedule_next_full_controls.py')
    sources[str(DIAG/'schedule_next_gap_panel.py')]=sha(DIAG/'schedule_next_gap_panel.py')
    sources[str(DEPENDENCY_PLAN)]=sha(DEPENDENCY_PLAN)
    for filename,expected in read(DEPENDENCY_PLAN)['source_and_input_sha256'].items():
        if filename in sources and sources[filename]!=expected:raise ValueError('Prior/new frozen source differs')
        sources[filename]=expected
    plan=dict(prepared_utc=now(),groups=groups,jobs=jobs,source_and_input_sha256=sources,controller_sha256=sha(__file__),
              worker_sha256=sha(WORKER),worker_path=str(WORKER),worker=str(WORKER),environment=FROZEN.UTIL.environment(),n_jobs=12,n_groups=7,
              n_predictions=sum(job['n'] for job in jobs.values()),dependency_controller_plan=str(DEPENDENCY_PLAN),
              dependency_controller_state=str(DEPENDENCY_STATE),dependency_marker=str(DEPENDENCY_MARKER),
              authorization='User authorized the affected NeXT experiments after the existing three full controls; identical beta2/generation with only the caller stream wait repair',
              scheduling='Ordered paired groups, two owned loading reservations maximum; FULL true exclusive with free>=32000MiB')
    verify(plan)
    if PLAN.exists() or STATE.exists():raise FileExistsError('Preserve existing continuation registration')
    atomic(PLAN,plan);atomic(STATE,dict(status='prepared',started_utc=None,registered_jobs=[],publication_expected_groups=[],
                                      plan=str(PLAN),plan_sha256=sha(PLAN),child_pids=[],jobs={},groups={},updated_utc=now()))
    print(json.dumps(dict(CPU_only=True,GPU_started=False,plan=str(PLAN),state=str(STATE),n_jobs=12,n_groups=7,n_predictions=plan['n_predictions'],
                         run_command=[sys.executable,str(Path(__file__).resolve()),'--run','--plan',str(PLAN)],dependency=dependencies()),indent=2))

def validate_job(job,plan):
    verify(plan)
    if sha(job['control_protocol'])!=job['control_protocol_sha256']:raise ValueError('Frozen job protocol differs')
    worker=module(worker_path(plan),'new_continuation_finished_checks')
    verified=worker.validate_finished(read(job['control_protocol']))
    finished=verified['finished']
    if finished.get('success') is not True or finished.get('n')!=job['n'] or finished.get('generation_exit_code')!=0:
        raise ValueError('Continuation official score hard gate did not complete')
    return verified

def run(plan_path):
    plan=read(plan_path);state=read(STATE)
    lock=(OUT/'streamwait_repair_continuation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if state['status']!='prepared' or state['plan_sha256']!=sha(plan_path):raise ValueError('Preserve previous continuation registration/state')
    verify(plan);FROZEN.UTIL.environment()
    ordered=[job_id for group in plan['groups'] for job_id in group['job_ids']]
    state.update(status='waiting_for_existing_controls',started_utc=now(),supervisor_pid=os.getpid(),registered_jobs=ordered,
                 publication_expected_groups=[group['group_id'] for group in plan['groups']],
                 jobs={key:dict(value,status='pending') for key,value in plan['jobs'].items()},
                 groups={group['group_id']:dict(group,status='pending') for group in plan['groups']},updated_utc=now())
    atomic(STATE,state);active={}
    def interrupted(signum,frame):raise KeyboardInterrupt('Continuation supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while not dependencies()['ready']:
            state.update(status='waiting_for_existing_controls',dependency=dependencies(),child_pids=[],updated_utc=now());atomic(STATE,state);time.sleep(1)
        verify(plan);state['dependency_validation']=validate_dependency();atomic(STATE,state)
        for group in plan['groups']:
            group_id=group['group_id'];state['current_group']=group_id;state['groups'][group_id]['status']='running'
            while True:
                for job_id,entry in list(active.items()):
                    returned=entry['child'].poll()
                    if returned is None:continue
                    entry['log'].close();del active[job_id]
                    if returned:
                        state['jobs'][job_id].update(status='failed',returncode=returned,error='Worker exited '+str(returned),finished_utc=now())
                    else:
                        verify(plan)
                        try:
                            validation=validate_job(plan['jobs'][job_id],plan)
                            state['jobs'][job_id].update(status='complete',returncode=0,validation=validation,finished_utc=now())
                        except Exception as error:state['jobs'][job_id].update(status='failed',returncode=0,error=str(error),finished_utc=now())
                statuses=[state['jobs'][job_id]['status'] for job_id in group['job_ids']]
                if not active and all(status in ('complete','failed') for status in statuses):
                    if any(status=='failed' for status in statuses):raise RuntimeError('Continuation group failed; completed partner and partial evidence preserved: '+group_id)
                    state['groups'][group_id].update(status='complete',finished_utc=now());atomic(STATE,state);break
                pending=[job_id for job_id in group['job_ids'] if state['jobs'][job_id]['status']=='pending']
                ready_dependency=dependencies()['ready']
                if not ready_dependency:raise RuntimeError('Existing controls completion identity changed')
                gpu=FROZEN.UTIL.gpu_snapshot();pids={entry['child'].pid for entry in active.values()}
                progress={entry['child'].pid:has_progress(plan['jobs'][job_id]) for job_id,entry in active.items()}
                if active and len([entry for entry in gpu['entries'] if not (entry['memory_mib']==0 and entry.get('alive') and entry['command'].startswith('/usr/local/bin/ollama runner '))])>2:
                    raise RuntimeError('External third GPU context appeared; terminate only owned continuation children')
                job=plan['jobs'][pending[0]] if pending else plan['jobs'][group['job_ids'][0]]
                ready=launch_gate(job,gpu,pids,progress,ready_dependency)
                state.update(status='running' if active else 'waiting_for_GPU_slot',gpu_gate=ready,child_pids=sorted(pids),updated_utc=now())
                for job_id,entry in active.items():state['jobs'][job_id].update(status='generating' if progress[entry['child'].pid] else 'loading_reserved')
                atomic(STATE,state)
                if pending and ready['ready']:
                    job_id=pending[0];job=plan['jobs'][job_id];verify(plan);protected(job);FROZEN.UTIL.environment()
                    fresh=launch_gate(job,FROZEN.UTIL.gpu_snapshot(),pids,progress,dependencies()['ready'])
                    if not fresh['ready']:time.sleep(1);continue
                    log=Path(job['output_paths']['log']).open('x')
                    try:
                        child=subprocess.Popen(job['generate_command'],cwd=ROOT/'LLaVA',env=dict(os.environ,**plan['environment']),
                                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    except BaseException:log.close();raise
                    active[job_id]=dict(child=child,log=log)
                    state['jobs'][job_id].update(status='loading_reserved',pid=child.pid,process_identity=process(child.pid),started_utc=now(),generation_launch_gate=fresh)
                    state.update(child_pids=sorted(entry['child'].pid for entry in active.values()),updated_utc=now());atomic(STATE,state)
                time.sleep(1)
        state.update(status='complete',child_pids=[],finished_utc=now(),updated_utc=now());atomic(STATE,state)
        marker=OUT/'streamwait_repair_continuation.finished.json'
        if marker.exists():raise FileExistsError('Preserve previous continuation success marker')
        atomic(marker,dict(success=True,complete=True,n_jobs=12,n_groups=7,registered_jobs=ordered,
                           publication_expected_groups=state['publication_expected_groups'],plan_sha256=sha(plan_path),finished_utc=now()))
    except BaseException as error:
        for job_id,entry in active.items():
            FROZEN.UTIL.stop_owned(entry['child']);entry['log'].close()
            state['jobs'][job_id].update(status='interrupted_preserved',error=str(error),finished_utc=now())
        state.update(status='failed',error=str(error),child_pids=[],updated_utc=now());atomic(STATE,state);raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--plan',type=Path,default=PLAN);args=parser.parse_args()
    if args.run:run(args.plan)
    else:prepare()

if __name__=='__main__':main()
