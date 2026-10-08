"""Independent fixed legacy continuation; only --run registers the 32 jobs."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import json

OUT=Path(__file__).resolve().parent
ROOT=Path('/media/disk2/YZX/research/EADP_amp')
WORKER=OUT/'legacy_streamwait_worker.py'
INVENTORY=OUT/'legacy_affected_inventory.json'
PLAN=OUT/'legacy_streamwait_continuation.plan.json'
STATE=OUT/'legacy_streamwait_continuation.state.json'
MARKER=OUT/'legacy_streamwait_continuation.finished.json'
DEPENDENCY_PLAN=OUT/'streamwait_repair_continuation.plan.json'
DEPENDENCY_STATE=OUT/'streamwait_repair_continuation.state.json'
DEPENDENCY_MARKER=OUT/'streamwait_repair_continuation.finished.json'
TASK_ORDER=('mme','pope','mmben','mmbcn','vizwiz','gqa')
COUNTS=dict(mme=2374,pope=8910,mmben=4876,mmbcn=4876,vizwiz=4319,gqa=12578)
# Root archived the never-registered attempt1 preparation; keep every old
# protocol/output path intact and use only independent attempt2 paths.
ATTEMPT=2

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded

PRIOR=module(OUT/'schedule_streamwait_repair_continuation.py','frozen_phase1_helpers')
FROZEN=PRIOR.FROZEN
read,sha,atomic,now,process=PRIOR.read,PRIOR.sha,PRIOR.atomic,PRIOR.now,PRIOR.process

def group_specs():
    """Exact reviewed inventory, task barriers, and honest unpaired audit cells."""
    groups=[]
    for task in TASK_ORDER:
        if task in ('mme','pope','gqa'):
            groups.append(dict(group_id=f'legacy_v15_{task}_K128_streamwait',model='v15',task=task,budget=128,kind='pair',
                               arms=['EADP','AnchorZip'],job_ids=[f'v15_{task}_EGATHER',f'v15_{task}_LRMAIN025']))
            if task!='pope':
                groups.append(dict(group_id=f'legacy_v15_{task}_AZ_K32_streamwait',model='v15',task=task,budget=32,kind='single_audit',
                                   arms=['AnchorZip'],job_ids=[f'v15_{task}_LRMAIN00625']))
            # Low-budget MME cells are first among individual audit arms.
            groups.append(dict(group_id=f'legacy_next_{task}_AZ_K32_streamwait',model='next',task=task,budget=32,kind='single_audit',
                               arms=['AnchorZip'],job_ids=[f'next_{task}_LRMAIN00625']))
            groups.append(dict(group_id=f'legacy_v15_{task}_AZ_K64_streamwait',model='v15',task=task,budget=64,kind='single_audit',
                               arms=['AnchorZip'],job_ids=[f'v15_{task}_LRMAIN0125']))
            for budget,suffix in ((64,'0125'),(128,'025')):
                groups.append(dict(group_id=f'legacy_next_{task}_AZ_K{budget}_streamwait',model='next',task=task,budget=budget,kind='single_audit',
                                   arms=['AnchorZip'],job_ids=[f'next_{task}_LRMAIN{suffix}']))
            groups.append(dict(group_id=f'legacy_v15_{task}_FULL_streamwait',model='v15',task=task,budget=0,kind='FULL',
                               arms=['FULL'],job_ids=[f'v15_{task}_FULL']))
        else:
            for budget,suffix in ((32,'00625'),(64,'0125'),(128,'025')):
                groups.append(dict(group_id=f'legacy_next_{task}_AZ_K{budget}_streamwait',model='next',task=task,budget=budget,kind='single_audit',
                                   arms=['AnchorZip'],job_ids=[f'next_{task}_LRMAIN{suffix}']))
    return groups

GROUP_SPECS=group_specs()
FIXED_IDS=tuple(key for group in GROUP_SPECS for key in group['job_ids'])

def worker_path(plan):
    path=Path(plan['worker_path']).resolve()
    if path!=WORKER:raise ValueError('Only the independent fixed legacy worker is approved')
    return path

def verify(plan):
    if sha(worker_path(plan))!=plan['worker_sha256']:raise ValueError('Frozen legacy worker drift')
    for filename,expected in plan['source_and_input_sha256'].items():
        if sha(filename)!=expected:raise ValueError('Frozen legacy source/input drift: '+filename)
    if sha(__file__)!=plan['controller_sha256']:raise ValueError('Frozen legacy controller drift')

def dependency_gate(state,marker,original_ready,is_alive=process):
    expected=set(read(DEPENDENCY_PLAN)['jobs'])
    pids=[job.get('pid') for job in state.get('jobs',{}).values()]
    return (state.get('status')=='complete' and set(state.get('registered_jobs',[]))==expected
            and state.get('child_pids')==[] and not is_alive(state.get('supervisor_pid'))['alive']
            and all(not is_alive(pid)['alive'] for pid in pids)
            and all(state.get('jobs',{}).get(key,{}).get('status')=='complete' for key in expected)
            and marker.get('success') is True and marker.get('complete') is True
            and marker.get('n_jobs')==12 and marker.get('n_groups')==7
            and set(marker.get('registered_jobs',[]))==expected
            and marker.get('plan_sha256')==sha(DEPENDENCY_PLAN) and original_ready)

def dependencies():
    state=read(DEPENDENCY_STATE);marker=read(DEPENDENCY_MARKER) if DEPENDENCY_MARKER.is_file() else {}
    return dict(ready=dependency_gate(state,marker,PRIOR.dependencies()['ready']),
                phase1_status=state.get('status'),phase1_supervisor_pid=state.get('supervisor_pid'),
                phase1_child_pids=state.get('child_pids'),phase1_finished_marker=str(DEPENDENCY_MARKER))

def validate_dependency():
    if not dependencies()['ready']:raise ValueError('Phase1/original controls not complete and exited')
    plan=read(DEPENDENCY_PLAN);PRIOR.verify(plan)
    reports={key:PRIOR.validate_job(job,plan) for key,job in plan['jobs'].items()}
    return dict(success=True,validated_utc=now(),marker_sha256=sha(DEPENDENCY_MARKER),reports=reports)

def launch_gate(job,gpu,active,progress,dependency_ready,active_jobs=()):
    result=FROZEN.gate(gpu,active,progress,dependency_ready);reasons=list(result['reasons'])
    if any(other['exclusive_gpu'] for other in active_jobs):reasons.append('running_exclusive_job_blocks_other_model')
    if job['exclusive_gpu'] and (active or result['contexts']):reasons.append('exclusive_job_requires_no_model_or_loading_reservation')
    if gpu['free_mib']<job['min_free_memory_mib'] and 'insufficient_free_GPU_memory' not in reasons:reasons.append('insufficient_free_GPU_memory')
    result.update(ready=not reasons,reasons=reasons,minimum_free_mib=job['min_free_memory_mib'],exclusive_gpu=job['exclusive_gpu'])
    return result

def fairness(group,jobs):
    selected=[jobs[key] for key in group['job_ids']]
    for job in selected:
        params=job['parameters']
        if params.get('beta')!=2. or params.get('alpha')!=.5 or params.get('visual_token_num')!=group['budget']:
            raise ValueError('Legacy method parameters changed')
        if job['model']!=group['model'] or job['task']!=group['task'] or job['n']!=COUNTS[group['task']]:
            raise ValueError('Legacy fixed native dataset identity differs')
    if group['kind']=='pair':
        common=[]
        for job in selected:
            params=dict(job['parameters']);params.pop('anchorzip',None);common.append(params)
        if len(common)!=2 or common[0]!=common[1]:raise ValueError('Legacy E/AZ paired input/generation differs')
    elif len(selected)!=1:raise ValueError('Single audit/FULL group must have one arm')

def prepare():
    if PLAN.exists() or STATE.exists():raise FileExistsError('Preserve existing legacy registration')
    worker=module(WORKER,'legacy_prepare_only');specs=worker.job_specs()
    if set(specs)!=set(FIXED_IDS):raise ValueError('Worker is not the exact reviewed legacy32 inventory')
    inventory=read(INVENTORY)
    if {job['jobid'] for job in inventory['jobs']}!=set(FIXED_IDS):raise ValueError('Reviewed fixed legacy inventory differs')
    jobs={};sources={};groups=[]
    for original in GROUP_SPECS:
        group=dict(original,protocol_label='legacy_native_caller_stream_wait_only_beta2')
        for position,key in enumerate(group['job_ids']):
            spec=specs[key];protocol,paths=worker.prepare(key,attempt=ATTEMPT)
            paths={name:str(path) for name,path in paths.items()};protocol_path=paths['control_protocol']
            if spec['model']!=group['model'] or spec['task']!=group['task'] or spec['n']!=COUNTS[group['task']]:
                raise ValueError('Legacy worker native task/model/count differs')
            for filename,expected in protocol['source_and_input_sha256'].items():
                if filename in sources and sources[filename]!=expected:raise ValueError('Conflicting legacy frozen source')
                sources[filename]=expected
            sources[protocol_path]=sha(protocol_path)
            if protocol.get('image_manifest'):sources[protocol['image_manifest']]=protocol['image_manifest_sha256']
            if protocol.get('identity_manifest'):sources[protocol['identity_manifest']]=protocol['identity_manifest_sha256']
            jobs[key]=dict(job_id=key,key=key,model=group['model'],task=group['task'],budget=group['budget'],
                           method=group['arms'][position],n=spec['n'],attempt=ATTEMPT,parameters=protocol['parameters'],
                           control_protocol=protocol_path,control_protocol_sha256=sha(protocol_path),output_paths=paths,
                           generate_command=protocol['generate_command'],min_free_memory_mib=spec['min_free_memory_mib'],
                           exclusive_gpu=spec['exclusive_gpu'])
            if jobs[key]['min_free_memory_mib']<22500:raise ValueError('Unsafe legacy minimum GPU memory')
            if jobs[key]['model']=='next' and group['kind']=='FULL' and (not jobs[key]['exclusive_gpu'] or jobs[key]['min_free_memory_mib']<32000):
                raise ValueError('NeXT FULL requires true exclusive GPU and >=32000MiB')
            PRIOR.protected(jobs[key])
        fairness(group,jobs);groups.append(group)
    for path in [WORKER,INVENTORY,OUT/'schedule_streamwait_repair_continuation.py',DEPENDENCY_PLAN]:
        sources[str(path)]=sha(path)
    for filename,expected in read(DEPENDENCY_PLAN)['source_and_input_sha256'].items():
        if filename in sources and sources[filename]!=expected:raise ValueError('Phase1/legacy frozen source differs')
        sources[filename]=expected
    plan=dict(prepared_utc=now(),groups=groups,jobs=jobs,source_and_input_sha256=sources,controller_sha256=sha(__file__),
              worker_path=str(WORKER),worker_sha256=sha(WORKER),environment=FROZEN.UTIL.environment(),
              n_jobs=32,n_groups=29,n_predictions=sum(job['n'] for job in jobs.values()),
              dependency_controller_plan=str(DEPENDENCY_PLAN),dependency_controller_state=str(DEPENDENCY_STATE),
              dependency_marker=str(DEPENDENCY_MARKER),task_order=list(TASK_ORDER),
              authorization='User authorized all affected existing experiments; exact legacy32 inventory plus only caller stream waits; unpaired audit cells are not E/AZ comparisons',
              scheduling='All MME first, then POPE/MMB_EN/MMB_CN/VizWiz/GQA; task barriers and <=2 loading/running reservations; worker exclusive/minfree preserved')
    if plan['n_predictions']!=224199:raise ValueError('Fixed legacy prediction denominator differs')
    verify(plan);atomic(PLAN,plan)
    atomic(STATE,dict(status='prepared',started_utc=None,registered_jobs=[],publication_expected_groups=[],plan=str(PLAN),
                      plan_sha256=sha(PLAN),child_pids=[],jobs={},groups={},updated_utc=now()))
    print(json.dumps(dict(CPU_only=True,GPU_started=False,plan=str(PLAN),state=str(STATE),n_jobs=32,n_groups=29,
                         n_predictions=plan['n_predictions'],run_command=[sys.executable,str(Path(__file__).resolve()),'--run','--plan',str(PLAN)],dependency=dependencies()),indent=2))

def validate_job(job,plan):
    verify(plan)
    if sha(job['control_protocol'])!=job['control_protocol_sha256']:raise ValueError('Frozen legacy protocol differs')
    worker=module(worker_path(plan),'legacy_finished_checks');verified=worker.validate_finished(read(job['control_protocol']))
    finished=verified['finished']
    if (finished.get('success') is not True or finished.get('complete') is not True or finished.get('n')!=job['n']
        or finished.get('generation_exit_code')!=0 or finished.get('job_id')!=job['job_id']):
        raise ValueError('Legacy official score hard gate failed')
    return verified

def run(plan_path):
    plan=read(plan_path);state=read(STATE);lock=(OUT/'legacy_streamwait_continuation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if state['status']!='prepared' or state['plan_sha256']!=sha(plan_path):raise ValueError('Preserve prior legacy registration/state')
    verify(plan);FROZEN.UTIL.environment()
    ordered=[key for group in plan['groups'] for key in group['job_ids']]
    state.update(status='waiting_for_phase1',started_utc=now(),supervisor_pid=os.getpid(),registered_jobs=ordered,
                 publication_expected_groups=[group['group_id'] for group in plan['groups']],
                 jobs={key:dict(job,status='pending') for key,job in plan['jobs'].items()},
                 groups={group['group_id']:dict(group,status='pending') for group in plan['groups']},updated_utc=now())
    atomic(STATE,state);active={}
    def interrupted(signum,frame):raise KeyboardInterrupt('Legacy continuation supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while not dependencies()['ready']:
            state.update(status='waiting_for_phase1',dependency=dependencies(),child_pids=[],updated_utc=now());atomic(STATE,state);time.sleep(1)
        verify(plan);state['dependency_validation']=validate_dependency();atomic(STATE,state)
        for task in plan['task_order']:
            group_batch=[group for group in plan['groups'] if group['task']==task]
            task_ids=[key for group in group_batch for key in group['job_ids']];state['current_task']=task
            while True:
                for key,entry in list(active.items()):
                    returned=entry['child'].poll()
                    if returned is None:continue
                    entry['log'].close();del active[key]
                    if returned:
                        state['jobs'][key].update(status='failed',returncode=returned,error='Worker exited '+str(returned),finished_utc=now())
                        raise RuntimeError('Legacy worker failed, partial preserved: '+key+' exit='+str(returned))
                    try:validation=validate_job(plan['jobs'][key],plan)
                    except Exception as error:
                        state['jobs'][key].update(status='failed',returncode=0,error=str(error),finished_utc=now());raise
                    state['jobs'][key].update(status='complete',returncode=0,validation=validation,finished_utc=now())
                for group in group_batch:
                    statuses=[state['jobs'][key]['status'] for key in group['job_ids']]
                    target=state['groups'][group['group_id']]
                    if all(status=='complete' for status in statuses) and target['status']!='complete':target.update(status='complete',finished_utc=now())
                    elif any(status!='pending' for status in statuses) and target['status']=='pending':target.update(status='running')
                pending=[key for key in task_ids if state['jobs'][key]['status']=='pending']
                if not pending and not active:atomic(STATE,state);break
                if not dependencies()['ready']:raise RuntimeError('Phase1 completion/process identity changed')
                gpu=FROZEN.UTIL.gpu_snapshot();pids={entry['child'].pid for entry in active.values()}
                progress={entry['child'].pid:PRIOR.has_progress(plan['jobs'][key]) for key,entry in active.items()}
                contexts=[entry for entry in gpu['entries'] if not (entry['memory_mib']==0 and entry.get('alive') and entry['command'].startswith('/usr/local/bin/ollama runner '))]
                if active and len(contexts)>2:raise RuntimeError('External third context appeared; stop only owned legacy workers')
                active_jobs=[plan['jobs'][key] for key in active]
                job=plan['jobs'][pending[0]] if pending else plan['jobs'][task_ids[0]]
                gate=launch_gate(job,gpu,pids,progress,True,active_jobs)
                state.update(status='running' if active else 'waiting_for_GPU_slot',gpu_gate=gate,child_pids=sorted(pids),updated_utc=now())
                for key,entry in active.items():state['jobs'][key]['status']='generating' if progress[entry['child'].pid] else 'loading_reserved'
                atomic(STATE,state)
                if pending and gate['ready']:
                    key=pending[0];job=plan['jobs'][key];verify(plan);PRIOR.protected(job);FROZEN.UTIL.environment()
                    fresh=launch_gate(job,FROZEN.UTIL.gpu_snapshot(),pids,progress,dependencies()['ready'],active_jobs)
                    if not fresh['ready']:time.sleep(1);continue
                    log=Path(job['output_paths']['log']).open('x')
                    try:child=subprocess.Popen(job['generate_command'],cwd=ROOT/'LLaVA',env=dict(os.environ,**plan['environment']),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    except BaseException:log.close();raise
                    active[key]=dict(child=child,log=log)
                    state['jobs'][key].update(status='loading_reserved',pid=child.pid,process_identity=process(child.pid),started_utc=now(),generation_launch_gate=fresh)
                    state.update(child_pids=sorted(entry['child'].pid for entry in active.values()),updated_utc=now());atomic(STATE,state)
                time.sleep(1)
        state.update(status='complete',child_pids=[],finished_utc=now(),updated_utc=now());atomic(STATE,state)
        if MARKER.exists():raise FileExistsError('Preserve previous legacy success marker')
        atomic(MARKER,dict(success=True,complete=True,n_jobs=32,n_groups=29,registered_jobs=ordered,
                           publication_expected_groups=state['publication_expected_groups'],plan_sha256=sha(plan_path),finished_utc=now()))
    except BaseException as error:
        for key,entry in active.items():
            FROZEN.UTIL.stop_owned(entry['child']);entry['log'].close()
            state['jobs'][key].update(status='interrupted_preserved',error=str(error),finished_utc=now())
        state.update(status='failed',error=str(error),child_pids=[],updated_utc=now());atomic(STATE,state);raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--plan',type=Path,default=PLAN);args=parser.parse_args()
    if args.run:run(args.plan)
    else:prepare()

if __name__=='__main__':main()

