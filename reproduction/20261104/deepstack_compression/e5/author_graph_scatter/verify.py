"""CPU-only evidence audit; capture failure and identity failure stay distinct."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import torch
import worker

HERE=Path(__file__).resolve().parent;RUN=HERE/'run_smoke001';OBS=RUN/'observations';MON=HERE/'monitor_smoke001'
KS=(1,16,128,256,512);NAMES=('real_A','changed_B','real_A_again','tie','zero','seeded_random')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return worker.sha(p)

def check():
    plan=read(HERE/'plan001.json');launch=read(RUN/'launch.json');exit=read(RUN/'exit.json')
    s=read(MON/'started.json');t=read(MON/'terminal.json');complete=read(OBS/'worker_complete.json')
    if (launch.get('plan_sha256')!=sha(HERE/'plan001.json') or launch.get('stage')!='smoke'
        or s.get('worker_pid')!=launch.get('controller',{}).get('pid') or exit.get('controller')!=launch.get('controller')
        or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]
        or t.get('worker_pid')!=s.get('worker_pid') or exit.get('returncode')!=0 or exit.get('container_exit_code')!=0
        or exit.get('container_stopped_verified') is not True or exit.get('timed_out') is not False
        or exit.get('interrupted') is not None or exit.get('failure_type') is not None
        or exit.get('client_exited_while_container_running') is not False or exit.get('gpu_pids_after')!=[]):
        raise ValueError('outer run did not exit cleanly')
    identity=exit['container_identity'];objects=json.loads(subprocess.run(['docker','inspect','--type','container',identity['container_id']],
                  check=True,capture_output=True,text=True,timeout=15).stdout)
    if (len(objects)!=1 or objects[0]['Id']!=identity['container_id'] or objects[0]['State']['Running'] is not False
        or objects[0]['State']['Pid']!=0 or objects[0]['State']['ExitCode']!=0):raise ValueError('container not stopped')
    obj=objects[0];cmd=launch['command'];cfg=obj['Config'];host=obj['HostConfig']
    if (obj['Image']!=identity.get('image') or obj['Name']!='/'+launch['container_name']
        or cfg.get('Entrypoint')!=[cmd[cmd.index('--entrypoint')+1]]
        or cfg.get('Cmd')!=[str(HERE/'worker.py'),'--out',str(OBS)]
        or cfg.get('User')!=cmd[cmd.index('--user')+1] or cfg.get('WorkingDir')!=cmd[cmd.index('--workdir')+1]
        or host.get('NetworkMode')!='none' or host.get('PidMode') not in ('',None)):
        raise ValueError('container profile mismatch')
    if {d.get('PathOnHost') for d in host.get('Devices',[])}!={'/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'}:
        raise ValueError('unexpected GPU devices')
    attempts=complete.get('cases')
    if (complete.get('status')!='attempts_recorded' or complete.get('GT_opened') is not False
        or complete.get('scoring_performed') is not False or not isinstance(attempts,list)
        or [x.get('case') for x in attempts]!=list(worker.CASES)):
        raise ValueError('attempt inventory incomplete')
    case_archive=torch.load(worker.cpu.CASE,map_location='cpu',weights_only=True)['ordered'][0].tolist()
    files=[HERE/'plan001.json',RUN/'launch.json',RUN/'exit.json',OBS/'worker_complete.json',OBS/'progress.jsonl',
           MON/'started.json',MON/'terminal.json',HERE/'preflight001.json',HERE/'independent_prelaunch_review.json']
    cases=[]
    for attempt in attempts:
        name=attempt['case'];path=OBS/(name+'.jsonl');stdout=OBS/(name+'.stdout.log');stderr=OBS/(name+'.stderr.log')
        files.extend([stdout,stderr]);rows=[]
        if path.exists():files.append(path);rows=[json.loads(x) for x in path.read_text().splitlines()]
        failures=[x for x in rows if x.get('event')=='failed'];comparisons=[x for x in rows if x.get('event')=='replay_compared']
        mismatches=[x.get('input') for x in comparisons if x.get('matched') is not True or x.get('repeat_same_input_equal') is not True]
        if type(attempt.get('returncode')) is not int or type(attempt.get('timed_out')) is not bool:raise ValueError('invalid child outcome')
        if attempt['returncode']==0:
            if attempt['timed_out'] or failures:raise ValueError('inconsistent successful child')
            if name.startswith('k_'):
                k=int(name[2:]);done=[x for x in rows if x.get('event')=='k_completed']
                if (tuple(x.get('input') for x in comparisons)!=NAMES or len(done)!=1
                    or done[0].get('matched_all_inputs') is not True or done[0].get('comparisons')!=6 or mismatches):
                    raise ValueError('successful full capture lacks identity evidence')
                for x in comparisons:
                    order=x.get('author_order')
                    if (order!=x.get('graph_order') or not isinstance(order,list) or len(order)!=1
                        or len(order[0])!=k or len(set(order[0]))!=k or any(type(i) is not int or not 0<=i<1024 for i in order[0])
                        or x.get('author_quota')!=[k] or x.get('graph_quota')!=[k]):raise ValueError('selection or quota malformed')
                    if x['input'] in ('real_A','real_A_again') and order[0]!=case_archive[:k]:raise ValueError('real author order differs from archive')
            else:
                primitive=[x for x in rows if x.get('event')=='primitive_compared']
                if len(primitive)!=1 or primitive[0].get('primitive')!=name or primitive[0].get('matched') is not True:
                    raise ValueError('successful primitive lacks evidence')
        elif not attempt['timed_out'] and not failures and not stderr.stat().st_size:
            raise ValueError('failed process lacks failure evidence')
        outcome='completed' if attempt['returncode']==0 else 'timeout' if attempt['timed_out'] else 'failed'
        cases.append({'case':name,'outcome':outcome,'returncode':attempt['returncode'],'timed_out':attempt['timed_out'],
                      'identity_mismatches':mismatches,'failure_types':[x.get('error_type') for x in failures],
                      'captured_replay_comparisons':len(comparisons)})
    if torch.cuda.is_initialized():raise ValueError('verifier initialized CUDA')
    full={x['case']:x for x in cases if x['case'].startswith('k_')}
    return {'status':'passed','scope':'complete diagnostic evidence only; not an automatic method success',
            'cases':cases,'full_k_identity_all_passed':all(full[f'k_{k:03d}']['outcome']=='completed' for k in KS),
            'GT_opened':False,'scoring_performed':False,'cuda_initialized':False,
            'files_sha256':{str(x.resolve()):sha(x) for x in files},'verifier_sha256':sha(__file__),'errors':[],
            'limitations':['capture/OOM/timeout classified as failure, not numerical mismatch',
                'only one archived real image and synthetic variations; no model integration, accuracy or TTFT claim',
                'PyTorch allocator fraction and statistics do not bound device-wide memory']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.resolve()!=HERE/'smoke001_verification.json':raise ValueError('unexpected output')
    try:r=check()
    except Exception as exc:r={'status':'failed','error_type':type(exc).__name__,'errors':[str(exc)],'verifier_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False}
    with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if r['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
