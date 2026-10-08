"""CPU prepare; --run waits for a spare slot and yields to main TextVQA FULL."""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path('/media/disk2/YZX/research/EADP_amp')
OUT=Path(__file__).resolve().parent
BATCH=ROOT/'Qwen_vl/outputs/audit_followup_20261008/rerun_batch'
PLAN=OUT/'stream_panel_scheduler.plan.json'
STATE=OUT/'stream_panel_scheduler.state.json'
MIN_FREE=22500
NEXT_MODEL='/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    temporary=Path(str(path)+'.tmp');temporary.write_text(json.dumps(value,indent=2));os.replace(temporary,path)

def process(pid):
    if not pid:return dict(alive=False,pid=pid,command='',ppid=None,start_ticks=None)
    try:
        base=Path(f'/proc/{pid}');stat=(base/'stat').read_text().rsplit(')',1)[1].split()
        return dict(alive=stat[0]!='Z',pid=pid,command=(base/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'),
                    ppid=int(stat[1]),start_ticks=int(stat[19]))
    except OSError:return dict(alive=False,pid=pid,command='',ppid=None,start_ticks=None)

def environment():
    bad=[key for key,value in os.environ.items() if key.startswith('USE_LLAVA_ARCH_') and value.lower() not in ('','0','false')]
    if bad:raise ValueError('Unexpected architecture environment keys: '+','.join(sorted(bad)))
    values=dict(PYTHONPATH=str(ROOT/'LLaVA'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for key in ('CUDA_VISIBLE_DEVICES','PYTORCH_CUDA_ALLOC_CONF'):
        if key in os.environ:values[key]=os.environ[key]
    return values

def gpu_snapshot():
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    entries=[]
    for line in lines:
        pid,memory=[value.strip() for value in line.split(',')];identity=process(int(pid))
        entries.append(dict(identity,memory_mib=int(memory)))
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(free)!=1:raise ValueError('Diagnostic registration requires the single project GPU')
    return dict(entries=entries,free_mib=int(free[0]))

def lane_snapshot():
    lanes=[]
    for name in ('lane1_v15','lane2_next'):
        state=read(BATCH/(name+'_state.json'));plan=read(BATCH/(name+'_plan.json'))
        done={entry['key'] for entry in state['completed']};expected={entry['key'] for entry in plan['jobs']}
        supervisor=process(state.get('supervisor_pid'));child=process(state.get('child_pid'))
        complete=state['status']=='complete' and done==expected and state.get('child_pid') is None and not supervisor['alive'] and not child['alive']
        lanes.append(dict(status=state['status'],complete=complete,completed_count=len(done),expected_count=len(expected),
                          supervisor=supervisor,child=child,task=state.get('current_job',{}).get('task'),arm=state.get('current_job',{}).get('arm')))
    return lanes

def full_started(lane):return lane['task']=='textvqa' and lane['arm']=='FULL'

def gate(lanes,gpu,after_all=False):
    reasons=[];ignored=[];resident=[]
    for entry in gpu['entries']:
        if entry['memory_mib']==0 and entry.get('alive') and entry['command'].startswith('/usr/local/bin/ollama runner '):ignored.append(entry)
        else:resident.append(entry)
    if not lanes[0]['complete']:reasons.append('v15_queue_or_supervisor_not_finished')
    if after_all:
        if not lanes[1]['complete']:reasons.append('waiting_for_both_main_queues')
        if resident:reasons.append('GPU_contexts_remain_after_main_queues')
    else:
        if lanes[1]['task']!='sqa' or lanes[1]['status']!='generating':reasons.append('lane2_not_generating_SQA')
        child=lanes[1]['child'];supervisor=lanes[1]['supervisor']
        authorized=[entry for entry in resident if entry['pid']==child.get('pid') and entry.get('alive') and entry['memory_mib']>0
                    and '--model-path '+NEXT_MODEL in entry['command'] and entry.get('ppid')==supervisor.get('pid')]
        if len(resident)!=1 or len(authorized)!=1:reasons.append('require_only_authorized_lane2_NeXT_context')
    if len(resident)>1:reasons.append('no_third_GPU_process')
    if gpu['free_mib']<MIN_FREE:reasons.append('insufficient_free_GPU_memory')
    return dict(ready=not reasons,reasons=reasons,ignored_zero_memory_ollama=ignored,resident_contexts=resident,free_mib=gpu['free_mib'],required_free_mib=MIN_FREE)

def verify_frozen(plan):
    for filename,expected in plan['source_and_input_sha256'].items():
        if sha(filename)!=expected:raise ValueError('Frozen source/input drift: '+filename)
    if sha(__file__)!=plan['scheduler_sha256']:raise ValueError('Frozen scheduler source drift')

def preserve_outputs(attempt):
    for name in ('records','result','state','log'):
        if Path(attempt['output_paths'][name]).exists():raise FileExistsError('Preserve existing diagnostic output: '+attempt['output_paths'][name])

def prepare(protocol_path):
    protocol=read(protocol_path);worker=Path(protocol['generate_command'][1])
    if sha(worker)!=protocol['worker_sha256']:raise ValueError('Worker SHA differs from prepared protocol')
    if protocol['n']!=128 or protocol['method']!='EADP':raise ValueError('Expected the fixed EADP128-question diagnosis')
    if not protocol['AST_validation']['restoration_ast_identical']:raise ValueError('Worker math AST identity not verified')
    sources=dict(protocol['source_and_input_sha256']);sources[str(worker)]=protocol['worker_sha256']
    sources[protocol['manifest']]=protocol['manifest_sha256']
    for lane in ('lane1_v15','lane2_next'):
        path=BATCH/(lane+'_plan.json');sources[str(path)]=sha(path)
        for filename,expected in read(path)['source_sha256'].items():
            if filename in sources and sources[filename]!=expected:raise ValueError('Conflicting frozen source identity')
            sources[filename]=expected
    command=list(protocol['generate_command'])
    if '--no-index-trace' not in command:command.append('--no-index-trace')
    attempts=[]
    stem=Path(protocol['output_paths']['protocol']).name.removesuffix('.protocol.json')
    for number in (1,2):
        argv=list(command);argv[argv.index('--attempt')+1]=str(number)
        paths=dict(protocol['output_paths'])
        if number==2:paths={name:str(Path(value).with_name(Path(value).name.replace(stem,stem+'_attempt2',1))) for name,value in paths.items()}
        attempt=dict(attempt=number,command=argv,output_paths=paths)
        preserve_outputs(attempt);attempts.append(attempt)
    plan=dict(registered_utc=now(),authorization='User authorized an additional mechanism diagnosis while preserving the main queues',
              worker_protocol_at_registration=protocol,worker_source=str(worker),source_and_input_sha256=sources,
              scheduler_sha256=sha(__file__),attempts=attempts,environment=environment(),minimum_free_mib=MIN_FREE,
              no_index_trace=True,poll_seconds=1,termination_policy='Terminate only the owned diagnosis child at lane2 TextVQA FULL; retry in fresh attempt2 after both main queues exit')
    verify_frozen(plan)
    if PLAN.exists() or STATE.exists():raise FileExistsError('Preserve existing scheduler registration/state')
    atomic(PLAN,plan);atomic(STATE,dict(status='prepared',registered_utc=now(),plan=str(PLAN),plan_sha256=sha(PLAN),child_pid=None,attempts=[]))
    print(json.dumps(dict(prepared=True,GPU_started=False,plan=str(PLAN),state=str(STATE),run_command=[sys.executable,str(Path(__file__).resolve()),'--run','--plan',str(PLAN)],initial_gate=gate(lane_snapshot(),gpu_snapshot())),indent=2))

def stop_owned(child):
    if child.poll() is not None:return
    if os.getpgid(child.pid)!=child.pid:raise RuntimeError('Owned worker process group identity changed')
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)

def validate_success(attempt,plan):
    paths=attempt['output_paths'];result=read(paths['result']);protocol=read(paths['protocol']);manifest=read(paths['manifest'])
    if result.get('success') is not True or result.get('complete') is not True or result['summary']['n']!=128 or result['summary']['n_generations']!=512:raise ValueError('Diagnostic result not complete')
    if protocol['worker_sha256']!=plan['source_and_input_sha256'][plan['worker_source']] or protocol.get('index_trace') is not False:raise ValueError('Runtime worker/index observation differs')
    if protocol['parameters']!=plan['worker_protocol_at_registration']['parameters'] or protocol['AST_validation']!=plan['worker_protocol_at_registration']['AST_validation']:raise ValueError('Runtime parameter/math identity differs')
    for name,key in [('protocol','protocol_sha256'),('manifest','manifest_sha256'),('records','records_sha256')]:
        if sha(paths[name])!=result[key]:raise ValueError('Diagnostic artifact SHA differs: '+name)
    if result['manifest_sha256']!=plan['worker_protocol_at_registration']['manifest_sha256']:raise ValueError('Fixed panel manifest changed')
    rows=[json.loads(line) for line in Path(paths['records']).open()]
    if len(rows)!=128:raise ValueError('Missing panel rows')
    for row,sample in zip(rows,manifest['samples']):
        for key in ('question_id','image','prompt','panel_position','dataset_position','prompt_sha256','image_sha256'):
            if row[key]!=sample[key]:raise ValueError('Fixed sample identity changed: '+key)
        if set(row['arms'])!={'as_is_1','wait_1','as_is_2','wait_2'}:raise ValueError('Missing paired repetitions')
        if any(not str(arm['text']).strip() or arm['text']=='FAILED' or not isinstance(arm['actual_visual_tokens_retained'],int) or arm['actual_visual_tokens_retained']<=0 for arm in row['arms'].values()):raise ValueError('Empty prediction/runtime count')
    verify_frozen(plan)
    return dict(success=True,attempt=attempt['attempt'],n=128,n_generations=512,result=paths['result'],result_sha256=sha(paths['result']))

def run(plan_path):
    plan=read(plan_path);state=read(STATE)
    if state['plan_sha256']!=sha(plan_path):raise ValueError('Registration plan changed')
    lock=(OUT/'stream_panel_scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if state['status']!='prepared':raise ValueError('Preserve prior scheduler state; resume requires explicit review')
    verify_frozen(plan);environment()
    state.update(supervisor_pid=os.getpid(),status='waiting_for_v15_slot',started_utc=now());atomic(STATE,state)
    child=None
    try:
        for attempt in plan['attempts']:
            number=attempt['attempt']
            while True:
                lanes=lane_snapshot();gpu=gpu_snapshot();ready=gate(lanes,gpu,after_all=number==2)
                if any(lane['status']=='failed' for lane in lanes):raise RuntimeError('A main queue failed; diagnostic launch halted')
                if number==1 and full_started(lanes[1]):
                    state['attempts'].append(dict(attempt=1,status='skipped_window',reason='main_TextVQA_FULL_started_before_launch',finished_utc=now()));atomic(STATE,state)
                    break
                state.update(status='waiting_for_v15_slot' if number==1 else 'waiting_for_both_main_queues',gpu_gate=ready,child_pid=None,updated_utc=now());atomic(STATE,state)
                if ready['ready']:break
                time.sleep(1)
            if number==1 and full_started(lanes[1]):continue
            verify_frozen(plan);preserve_outputs(attempt)
            # Recheck immediately before launch; the worker also checks its slot.
            lanes=lane_snapshot();ready=gate(lanes,gpu_snapshot(),after_all=number==2)
            if not ready['ready']:raise RuntimeError('Launch gate changed before worker start; no GPU started')
            env=dict(os.environ,**plan['environment']);environment()
            with Path(attempt['output_paths']['log']).open('x') as log:
                child=subprocess.Popen(attempt['command'],cwd=ROOT/'LLaVA',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                owned=dict(attempt=number,status='generating',pid=child.pid,process_identity=process(child.pid),started_utc=now(),output_paths=attempt['output_paths'],command=attempt['command'])
                state.update(status='generating',child_pid=child.pid,current_attempt=owned,gpu_gate=ready,updated_utc=now());atomic(STATE,state)
                interrupted=False
                while child.poll() is None:
                    lanes=lane_snapshot()
                    if number==1 and full_started(lanes[1]):
                        stop_owned(child);interrupted=True;break
                    time.sleep(1)
                returned=child.wait();child=None
            owned.update(status='interrupted_for_main_TextVQA_FULL' if interrupted else 'finished',returncode=returned,finished_utc=now())
            state['attempts'].append(owned);state.update(child_pid=None,current_attempt=None,updated_utc=now());atomic(STATE,state)
            if interrupted:continue
            if returned:raise RuntimeError('Diagnostic worker failed: '+str(returned))
            evidence=validate_success(attempt,plan)
            state.update(status='complete',validation=evidence,finished_utc=now(),child_pid=None);atomic(STATE,state)
            marker=OUT/'stream_panel_scheduler.finished.json'
            if marker.exists():raise FileExistsError('Preserve prior finished marker')
            atomic(marker,evidence);return
        raise RuntimeError('No completed diagnostic attempt')
    except BaseException as error:
        if child is not None:stop_owned(child)
        state.update(status='failed',error=str(error),child_pid=None,updated_utc=now());atomic(STATE,state);raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--plan',type=Path,default=PLAN)
    parser.add_argument('--worker-protocol',type=Path,default=OUT/'stability/next_text_K32_EADP_stream_panel128.protocol.json')
    args=parser.parse_args()
    if args.run:run(args.plan)
    else:prepare(args.worker_protocol)

if __name__=='__main__':main()
