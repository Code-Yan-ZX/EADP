"""Read-only completed E2/E3 smoke acceptance; no references/GT or scoring."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from local_imports import load_local
gate=load_local('gate');HERE=gate.HERE;BASE=gate.BASE;sha=gate.sha;check_approval=gate.check_approval
verify_smoke=load_local('verify_smoke');validate_content=verify_smoke.validate_content;lines=verify_smoke.lines
sys.path.insert(0,str(BASE/'e4'))
from identity_worker import PLAN,WEIGHTS,check_e1
worker=load_local('smoke_worker');selected=worker.selected;ARMS=worker.ARMS

def alive(ident):
    p=Path('/proc')/str(ident['pid'])/'stat'
    if not p.exists() or Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=ident['boot_id']:return False
    return int(p.read_text().rsplit(')',1)[1].split()[19])==ident['start_ticks']

def validate_weights(frozen,manifest):
    manifest=Path(manifest)
    if frozen.get('weights_identity_sha256')!=sha(manifest):raise ValueError('weight identity manifest changed')
    identity=json.loads(manifest.read_text());files=identity['files'];before=frozen.get('weight_stats_before',{})
    paths=[x['path'] for x in files]
    expected=[str(Path(identity['local_snapshot'])/('model-%05d-of-00004.safetensors'%i)) for i in range(1,5)]
    if identity.get('all_weight_payloads_match') is not True or paths!=expected or set(before)!=set(expected):raise ValueError('weight shard inventory')
    for item in files:
        digest=item.get('expected_lfs_sha256');path=Path(item['path']);s=path.stat()
        if (not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest) or
            item.get('matches') is not True or item.get('sha256')!=digest or s.st_size!=item['bytes'] or
            before[str(path)]!=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]):
            raise ValueError('weight shard identity or stat changed')

def main():
    ap=argparse.ArgumentParser()
    for name in ('e1-acceptance','e1-started','e1-terminal','run','run-started','run-terminal','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();run=a.run.resolve();launch=json.loads((run/'launch.json').read_text());ex=json.loads((run/'exit.json').read_text())
    started=json.loads(a.run_started.read_text());terminal=json.loads(a.run_terminal.read_text());approval=check_approval()
    if (terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or
        terminal.get('worker_pid')!=started.get('worker_pid') or launch.get('controller',{}).get('pid')!=terminal.get('worker_pid') or
        launch.get('controller')!=ex.get('controller') or launch.get('approval')!=approval or
        launch.get('worker_sha256')!=sha(HERE/'smoke_worker.py') or launch.get('launcher_sha256')!=sha(HERE/'launch.py')):raise ValueError('launch/supervisor identity')
    if (started.get('timeout_seconds')!=3720 or started.get('stale_seconds')!=1800 or started.get('snapshot_seconds')!=3600 or started.get('progress_paths')!=[str(run/'generated.jsonl')]):raise ValueError('supervision budget')
    command=started.get('command',[]);cwd=Path(started.get('cwd','/'))
    if ([(cwd/x).resolve() for x in command if Path(x).name=='launch.py']!=[(HERE/'launch.py').resolve()] or
        command.count('--run-id')!=1 or run.name!='run_'+command[command.index('--run-id')+1]):raise ValueError('supervisor command')
    if (ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or
        ex.get('container_stopped_verified') is not True or ex.get('client_exited_while_container_running') is not False or ex.get('gpu_pids_after')!=[]):raise ValueError('container exit')
    container=ex.get('container_identity',{});ident=container.get('host_init')
    if not ident or alive(ident) or alive(ex['controller']):raise ValueError('recorded process alive')
    state=json.loads(subprocess.run(['docker','inspect',container['container_id']],capture_output=True,text=True,timeout=10,check=True).stdout)[0]
    if state['Id']!=container['container_id'] or state['State']['Running'] or state['State']['ExitCode']!=0:raise ValueError('real container exit')
    proof=check_e1(a.e1_acceptance,a.e1_started,a.e1_terminal);panel=selected(proof)
    frozen=json.loads((run/'frozen.json').read_text());done=json.loads((run/'worker_complete.json').read_text())
    if (frozen.get('e1_acceptance_sha256')!=sha(a.e1_acceptance) or frozen.get('plan_sha256')!=sha(PLAN) or
        frozen.get('selected')!=[{k:r[k] for k in ('task','qid','group_id','message_sha256','image_sha256')} for r in panel] or
        frozen.get('arms')!=[list(x) for x in ARMS] or frozen.get('forced_reference_arms')!=['D0','D1'] or frozen.get('forced_decode_steps')!=32 or frozen.get('GT_opened') is not False or
        frozen.get('scoring_performed') is not False or done!={'questions':3,'natural_arms':21,'forced_arms':27}):raise ValueError('frozen protocol')
    validate_weights(frozen,WEIGHTS)
    for path,digest in frozen['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('source changed')
    refs=lines(proof['files']['journal']['path'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(json.loads(PLAN.read_text())['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
    result=validate_content(run,panel,refs,tokenizer)
    arrays=sorted((run/'arrays').glob('*.npz'))
    refs_array=[Path(r['arrays']['path']).resolve() for r in lines(run/'records.jsonl')]
    if len(arrays)!=48 or len(set(refs_array))!=48 or set(refs_array)!=set(arrays):raise ValueError('array inventory')
    files=[a.e1_acceptance,a.e1_started,a.e1_terminal,a.run_started,a.run_terminal]+[run/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','records.jsonl','generated.jsonl')]
    result.update(status='passed',questions=3,source_sha256=sha(__file__),files_sha256={str(p.resolve()):sha(p) for p in files},array_sha256={str(p):sha(p) for p in arrays},finished_unix=time.time(),
                  weights_check='manifest SHA and four shard current stat against frozen pre-run stat; acceptance did not recompute full shard SHA')
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
