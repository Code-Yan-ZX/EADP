"""CPU-only post-run evidence audit; no score, CUDA, or model inference."""
import argparse,json,math,os,subprocess
from pathlib import Path
import torch
import inventory,worker

HERE=Path(__file__).resolve().parent;RUN=HERE/'run_smoke001';OBS=RUN/'observations';MON=HERE/'monitor_smoke001'
if Path(inventory.__file__).resolve()!=HERE/'inventory.py' or Path(worker.__file__).resolve()!=HERE/'worker.py':
    raise ValueError('wrong verifier dependency import')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return inventory.sha(p)
def number(x):return type(x) in (int,float) and math.isfinite(x) and x>=0

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
        or cfg.get('Entrypoint')!=[cmd[cmd.index('--entrypoint')+1]] or cfg.get('Cmd')!=[str(HERE/'worker.py'),'--out',str(OBS)]
        or cfg.get('User')!=cmd[cmd.index('--user')+1] or cfg.get('WorkingDir')!=cmd[cmd.index('--workdir')+1]
        or host.get('NetworkMode')!='none' or host.get('PidMode') not in ('',None)):
        raise ValueError('container profile differs')
    if {d.get('PathOnHost') for d in host.get('Devices',[])}!={'/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'}:
        raise ValueError('GPU devices differ')
    entries=inventory.inventory(hash_files=False);inventory.require_scatter_review()
    if entries!=plan.get('case_manifest'):raise ValueError('frozen archive inventory differs')
    attempts=complete.get('cases')
    if (complete.get('status')!='attempts_recorded' or complete.get('GT_opened') is not False
        or complete.get('scoring_performed') is not False or not isinstance(attempts,list)
        or [x.get('case') for x in attempts]!=list(worker.CASES)):
        raise ValueError('budget attempt inventory incomplete')
    files=[HERE/'plan001.json',RUN/'launch.json',RUN/'exit.json',OBS/'worker_complete.json',OBS/'progress.jsonl',
           MON/'started.json',MON/'terminal.json',HERE/'preflight001.json',HERE/'independent_prelaunch_review.json']
    budgets=[]
    for attempt in attempts:
        name=attempt['case'];k=int(name[2:]);path=OBS/(name+'.jsonl');stdout=OBS/(name+'.stdout.log');stderr=OBS/(name+'.stderr.log')
        files.extend([stdout,stderr]);rows=[]
        if path.exists():files.append(path);rows=[json.loads(x) for x in path.read_text().splitlines()]
        comparisons=[x for x in rows if x.get('event')=='case_compared'];failures=[x for x in rows if x.get('event')=='failed']
        captures=[x for x in rows if x.get('event')=='capture_completed'];expected=[x for x in entries if x['budget']==k]
        if len(comparisons)>32 or [x.get('image_index') for x in comparisons]!=list(range(len(comparisons))):
            raise ValueError('duplicate or reordered image comparisons')
        mismatches=[]
        for row,item in zip(comparisons,expected):
            if any(row.get(key)!=item[key] for key in ('image_index','task','qid','group_id')) or row.get('case_sha256')!=item['sha256']:
                raise ValueError('case identity mismatch')
            archive=Path(item['path'])
            if sha(archive)!=item['sha256']:raise ValueError('actual archive changed')
            stored=torch.load(archive,map_location='cpu',weights_only=True)['ordered'].tolist()
            author=row.get('author_order');actual=row.get('graph_order')
            if (not isinstance(author,list) or len(author)!=1 or not isinstance(author[0],list) or len(author[0])!=k
                or len(set(author[0]))!=k or any(type(i) is not int or not 0<=i<1024 for i in author[0])
                or not isinstance(actual,list) or len(actual)!=1 or not isinstance(actual[0],list) or len(actual[0])!=k
                or row.get('author_quota')!=[k] or row.get('graph_quota')!=[k]
                or any(not number(row.get(x)) for x in ('input_copy_plus_replay_device_ms','host_wall_ms','graph_only_device_ms'))):
                raise ValueError('selection, quota or timing malformed')
            graph_equal=(actual==author);archive_equal=(author==stored)
            if (row.get('graph_equals_author') is not graph_equal or row.get('author_equals_archive') is not archive_equal
                or type(row.get('repeat_same_input_equal')) is not bool):raise ValueError('comparison flag inconsistent')
            if not graph_equal or not archive_equal or not row['repeat_same_input_equal']:mismatches.append(item['image_index'])
        if type(attempt.get('returncode')) is not int or type(attempt.get('timed_out')) is not bool:
            raise ValueError('child outcome malformed')
        if attempt['returncode']==0:
            done=[x for x in rows if x.get('event')=='budget_completed']
            if (attempt['timed_out'] or failures or len(captures)!=1 or len(comparisons)!=32 or mismatches
                or len(done)!=1 or done[0].get('all_identity_equal') is not True or done[0].get('cases')!=32):
                raise ValueError('successful budget lacks full identity evidence')
        elif not attempt['timed_out'] and not failures and not stderr.stat().st_size:
            raise ValueError('failed budget lacks failure evidence')
        if captures:
            c=captures[0]
            if len(captures)!=1 or not number(c.get('setup_wall_ms')):raise ValueError('capture setup malformed')
            for key in ('memory_before','memory_after'):
                if not isinstance(c.get(key),dict) or any(type(v) is not int or v<0 for v in c[key].values()):raise ValueError('allocator data malformed')
        outcome='completed' if attempt['returncode']==0 else 'timeout' if attempt['timed_out'] else 'failed'
        budgets.append({'k':k,'outcome':outcome,'compared_images':len(comparisons),'identity_mismatch_images':mismatches,
                        'failure_types':[x.get('error_type') for x in failures],
                        'setup_wall_ms':captures[0].get('setup_wall_ms') if captures else None,
                        'memory_before':captures[0].get('memory_before') if captures else None,
                        'memory_after':captures[0].get('memory_after') if captures else None})
    if torch.cuda.is_initialized():raise ValueError('CUDA initialized in verifier')
    return {'status':'passed','scope':'evidence audit only; not automatic identity success','budgets':budgets,
            'complete_96_identity':all(x['outcome']=='completed' and x['compared_images']==32 for x in budgets),
            'GT_opened':False,'scoring_performed':False,'cuda_initialized':False,
            'files_sha256':{str(x.resolve()):sha(x) for x in files},'verifier_sha256':sha(__file__),'errors':[],
            'limitations':['missing capture, OOM or timeout classified separately from identity mismatch',
                           '32 fixed real images only, not model integration, accuracy or TTFT',
                           'per-process PyTorch allocator figures are not device-wide peak memory']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.resolve()!=HERE/'smoke001_verification.json':raise ValueError('unexpected output')
    try:r=check()
    except Exception as exc:r={'status':'failed','error_type':type(exc).__name__,'errors':[str(exc)],'verifier_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False}
    with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if r['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
