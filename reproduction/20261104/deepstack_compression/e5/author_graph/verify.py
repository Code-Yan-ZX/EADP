"""CPU-only post-run evidence verifier; never performs GPU work or scoring."""
import argparse,hashlib,json,math,os,subprocess
from pathlib import Path
import torch
import preflight as cpu

HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
RUN=HERE/'run_smoke001';OBS=RUN/'observations';MON=HERE/'monitor_smoke001'
KS=(1,16,128,256,512);NAMES=('real_A','changed_B','real_A_again','tie','zero','seeded_random')

def sha(path):return cpu.sha(path)
def read(path):return json.loads(Path(path).read_text())
def finite_nonnegative(value):return type(value) in (int,float) and math.isfinite(value) and value>=0
def check():
    plan=read(HERE/'plan001.json');launch=read(RUN/'launch.json');exit=read(RUN/'exit.json')
    started=read(MON/'started.json');terminal=read(MON/'terminal.json')
    if (launch.get('plan_sha256')!=sha(HERE/'plan001.json') or launch.get('stage')!='smoke'
        or started.get('worker_pid')!=launch.get('controller',{}).get('pid')
        or exit.get('controller')!=launch.get('controller') or terminal.get('worker_pid')!=started.get('worker_pid')
        or terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[]
        or exit.get('returncode')!=0 or exit.get('container_exit_code')!=0 or exit.get('container_stopped_verified') is not True
        or exit.get('timed_out') is not False or exit.get('interrupted') is not None or exit.get('failure_type') is not None
        or exit.get('client_exited_while_container_running') is not False or exit.get('gpu_pids_after')!=[]):raise ValueError('run did not exit cleanly')
    identity=exit['container_identity'];actual=json.loads(subprocess.run(['docker','inspect','--type','container',identity['container_id']],
                  check=True,capture_output=True,text=True,timeout=15).stdout)
    if len(actual)!=1 or actual[0]['Id']!=identity['container_id'] or actual[0]['State']['Running'] is not False or actual[0]['State']['Pid']!=0 or actual[0]['State']['ExitCode']!=0:
        raise ValueError('container has not stopped successfully')
    obj=actual[0];cmd=launch['command'];cfg=obj['Config'];host=obj['HostConfig']
    if (obj['Image']!=identity.get('image') or obj['Name']!='/'+launch['container_name']
        or cfg.get('Entrypoint')!=[cmd[cmd.index('--entrypoint')+1]]
        or cfg.get('Cmd')!=[str(HERE/'worker.py'),'--out',str(OBS)]
        or cfg.get('User')!=cmd[cmd.index('--user')+1] or cfg.get('WorkingDir')!=cmd[cmd.index('--workdir')+1]
        or host.get('NetworkMode')!='none' or host.get('PidMode') not in ('',None)):
        raise ValueError('actual container launch profile differs')
    required_devices={'/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'}
    if {d.get('PathOnHost') for d in host.get('Devices',[])}!=required_devices:raise ValueError('GPU devices differ')
    completed=read(OBS/'worker_complete.json')
    if (completed.get('status')!='attempts_recorded' or completed.get('GT_opened') is not False
        or completed.get('scoring_performed') is not False or
        completed.get('cases')!=[{'k':k,'returncode':0,'timed_out':False} for k in KS]):
        raise ValueError('K attempts incomplete')
    if sha(cpu.CASE)!=cpu.CASE_SHA or sha(cpu.PROOF)!=cpu.PROOF_SHA:raise ValueError('source evidence changed')
    archived=torch.load(cpu.CASE,map_location='cpu',weights_only=True)['ordered'][0].tolist()
    if len(archived)!=512:raise ValueError('archive order malformed')
    files=[HERE/'plan001.json',RUN/'launch.json',RUN/'exit.json',OBS/'worker_complete.json',OBS/'progress.jsonl',
           MON/'started.json',MON/'terminal.json',HERE/'preflight001.json',HERE/'independent_prelaunch_review.json']
    count=0
    for k in KS:
        path=OBS/f'k_{k:03d}.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        if len(rows)!=8 or rows[0].get('event')!='capture_completed' or rows[-1]!={'event':'k_completed','k':k,'matched_all_inputs':True,'comparisons':6}:
            raise ValueError('capture or replay inventory mismatch K='+str(k))
        first=rows[0]
        if first.get('k')!=k or not finite_nonnegative(first.get('setup_wall_ms')):raise ValueError('setup time missing')
        for metric in ('memory_before','memory_after'):
            m=first.get(metric,{})
            if set(m)!={'allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes'} or any(type(v) is not int or v<0 for v in m.values()):raise ValueError('allocator stats malformed')
        if tuple(r.get('input') for r in rows[1:-1])!=NAMES:raise ValueError('replay schedule mismatch')
        for row in rows[1:-1]:
            author=row.get('author_order');graph=row.get('graph_order')
            if (row.get('event')!='replay_compared' or row.get('k')!=k or row.get('matched') is not True
                or row.get('repeat_same_input_equal') is not True or author!=graph or not isinstance(author,list) or len(author)!=1
                or not isinstance(author[0],list) or len(author[0])!=k or len(set(author[0]))!=k
                or any(type(x) is not int or not 0<=x<1024 for x in author[0])
                or row.get('author_quota')!=[k] or row.get('graph_quota')!=[k]
                or any(not finite_nonnegative(row.get(x)) for x in ('input_copy_plus_replay_device_ms','host_wall_ms','graph_only_device_ms'))):
                raise ValueError('selection, quota or timing mismatch K='+str(k))
            if row['input'] in ('real_A','real_A_again') and author[0]!=archived[:k]:raise ValueError('archived author prefix mismatch')
            count+=1
        files += [path,OBS/f'k_{k:03d}.stdout.log',OBS/f'k_{k:03d}.stderr.log']
    if torch.cuda.is_initialized():raise ValueError('verifier unexpectedly initialized CUDA')
    return {'status':'passed','ks':list(KS),'comparisons':count,'all_ordered_ids_equal':True,'GT_opened':False,
            'scoring_performed':False,'cuda_initialized':False,'files_sha256':{str(x.resolve()):sha(x) for x in files},
            'verifier_sha256':sha(__file__),'errors':[],
            'limitations':['one archived real image plus synthetic and changed inputs; not a model or accuracy result',
                'allocator statistics are process allocator quantities, not device-wide peak memory',
                'event times are exploratory single measurements, not end-to-end TTFT or a robust benchmark']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.resolve()!=HERE/'smoke001_verification.json':raise ValueError('unexpected output')
    try:r=check()
    except Exception as exc:r={'status':'failed','error_type':type(exc).__name__,'errors':[str(exc)],'verifier_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False}
    with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if r['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
