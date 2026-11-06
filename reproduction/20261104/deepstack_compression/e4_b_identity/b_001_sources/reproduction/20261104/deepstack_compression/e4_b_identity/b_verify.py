"""Host-only, no-GT acceptance of completed B diagnostics and actual exit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;E4=ROOT/'e4';NATIVE=ROOT/'e4_native'
sys.path[:0]=[str(E4),str(NATIVE)]
from identity_worker import PLAN,WEIGHTS,sha,read_jsonl
from native_contract import select_anchors
from native_gate import check_gates
from completion import alive
from b_imports import local
verify=local('verify_records').verify

def validate_terminal(run,started,ended,launch,exited):
    if (ended.get('reason')!='completed' or ended.get('worker_returncode')!=0 or
        ended.get('remaining_process_group')!=[] or ended.get('worker_pid')!=started.get('worker_pid')):
        raise ValueError('B supervisor did not finish cleanly')
    ctrl=launch.get('controller');container=exited.get('container_identity')
    if (not isinstance(ctrl,dict) or ctrl!=exited.get('controller') or ctrl.get('pid')!=ended.get('worker_pid') or
        not isinstance(container,dict) or not container.get('container_id') or not isinstance(container.get('host_init'),dict)):
        raise ValueError('B process identity mismatch')
    for ident in (ctrl,container['host_init']):
        if (not isinstance(ident.get('pid'),int) or ident['pid']<=0 or not isinstance(ident.get('start_ticks'),int) or
            ident['start_ticks']<=0 or not ident.get('boot_id') or alive(ident)):raise ValueError('B process still alive')
    if (exited.get('returncode')!=0 or exited.get('container_exit_code')!=0 or exited.get('timed_out') is not False or
        exited.get('interrupted') is not None or exited.get('container_stopped_verified') is not True or
        exited.get('client_exited_while_container_running') is not False or exited.get('gpu_pids_after')!=[]):
        raise ValueError('B container did not exit cleanly')
    cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
    if [(cwd/x).resolve() for x in cmd if Path(x).name=='b_launch.py']!=[(HERE/'b_launch.py').resolve()] or cmd.count('--run-id')!=1:
        raise ValueError('wrong B launcher')
    i=cmd.index('--run-id')
    if i+1>=len(cmd) or run.name!='b_'+cmd[i+1]:raise ValueError('wrong B output directory')
    budget=json.loads((HERE/'budget_plan.json').read_text());initial=ROOT/'initial_plan.json'
    if (sha(initial)!=budget['budget_source_sha256'] or json.loads(initial.read_text()).get('budget_end_unix')!=budget['budget_end_unix'] or
        started.get('timeout_seconds')!=budget['outer_timeout_seconds'] or started.get('stale_seconds')!=budget['journal_stale_seconds'] or
        started.get('snapshot_seconds')!=budget['snapshot_seconds'] or started.get('poll_seconds')!=budget['poll_seconds'] or
        started.get('grace_seconds')!=budget['grace_seconds'] or started.get('progress_paths')!=[str(run/'generated.jsonl')] or
        launch.get('timeout_seconds')!=budget['inner_timeout_seconds'] or
        not isinstance(launch.get('started_unix'),(int,float)) or launch['started_unix']+budget['outer_timeout_seconds']>budget['budget_end_unix']):
        raise ValueError('B supervision or budget mismatch')
    dock=launch.get('command',[])
    if (dock.count(str(HERE/'b_worker.py'))!=1 or dock.count('--out')!=1 or Path(dock[dock.index('--out')+1]).resolve()!=run or
        launch.get('worker_sha256')!=sha(HERE/'b_worker.py') or launch.get('launcher_sha256')!=sha(HERE/'b_launch.py')):
        raise ValueError('B worker source or command changed')
    if launch.get('approval_bundle')!=local('b_gate').check_approval():raise ValueError('B approval changed')
    receipt=run/'host_panel_gate.json'
    if (launch.get('host_panel_gate')!={'path':str(receipt),'sha256':sha(receipt)} or
        json.loads(receipt.read_text())!=local('b_gate').host_validate_panel()):raise ValueError('B host panel receipt invalid')
    p=subprocess.run(['docker','inspect',container['container_id']],capture_output=True,text=True,timeout=10,check=True)
    state=json.loads(p.stdout)[0]
    if state['Id']!=container['container_id'] or state['State']['Running'] or state['State']['ExitCode']!=0:
        raise ValueError('B independent container inspection failed')

def main():
    p=argparse.ArgumentParser()
    for name in ('e1-acceptance','e1-started','e1-terminal','smoke-dir','smoke-proof','smoke-started','smoke-terminal',
                 'run-dir','run-started','run-terminal','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();e1=check_gates(a);anchors=select_anchors(e1);run=a.run_dir.resolve()
    launch=json.loads((run/'launch.json').read_text());exited=json.loads((run/'exit.json').read_text())
    validate_terminal(run,json.loads(a.run_started.read_text()),json.loads(a.run_terminal.read_text()),launch,exited)
    frozen=json.loads((run/'frozen.json').read_text());done=json.loads((run/'worker_complete.json').read_text())
    if (frozen.get('stage')!='skip_drop_identity' or frozen.get('anchors')!=[{k:r[k] for k in ('task','qid','group_id','message_sha256','image_sha256')} for r in anchors] or
        frozen.get('forced_anchor_index')!=0 or frozen.get('natural_pairs')!=3 or frozen.get('forced_pairs')!=1 or
        frozen.get('gt_read') is not False or frozen.get('scoring') is not False or frozen.get('timing_claims') is not False or
        frozen.get('e1_acceptance_sha256')!=sha(a.e1_acceptance) or frozen.get('smoke_proof_sha256')!=sha(a.smoke_proof) or
        frozen.get('plan_sha256')!=sha(PLAN) or done.get('natural_pairs')!=3 or done.get('forced_pairs')!=1 or done.get('generations')!=8):
        raise ValueError('B frozen protocol incomplete')
    for path,digest in frozen.get('code_sha256',{}).items():
        if sha(path)!=digest:raise ValueError('B source changed: '+path)
    sources=[HERE/x for x in ('b_worker.py','b_launch.py','b_verify.py','b_gate.py','b_imports.py','budget_plan.json','arm.py','off_capture.py','pair_contract.py','verify_records.py')]
    sources += [E4/x for x in ('off_adapter.py','skip_deepstack.py','identity_records.py','decode_observer.py')]
    sources += [NATIVE/x for x in ('native_capture.py','native_contract.py','native_gate.py')]
    if any(frozen.get('code_sha256',{}).get(str(p))!=sha(p) for p in sources):raise ValueError('B source inventory incomplete')
    identity=json.loads(WEIGHTS.read_text());weights=identity.get('files',[]);stats=frozen.get('weight_stats_before',{})
    expected_paths=[str(Path(identity['local_snapshot'])/('model-%05d-of-00004.safetensors'%i)) for i in range(1,5)]
    if (frozen.get('weights_identity_sha256')!=sha(WEIGHTS) or identity.get('all_weight_payloads_match') is not True or
        len(weights)!=4 or [x.get('path') for x in weights]!=expected_paths or set(stats)!=set(expected_paths)):
        raise ValueError('B weight identity or inventory mismatch')
    for item in weights:
        path=item['path'];before=stats[path];s=Path(path).stat()
        if (item.get('matches') is not True or item.get('sha256')!=item.get('expected_lfs_sha256') or
            s.st_size!=item.get('bytes') or [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before):
            raise ValueError('B weights changed')
    refs={}
    for r in read_jsonl(a.smoke_dir/'records.jsonl'):
        if r.get('budget')==256 and r.get('forced') is False:
            key=(r['task'],str(r['qid']),r['ds_on'])
            if key in refs:raise ValueError('duplicate smoke reference')
            refs[key]=r
    records=read_jsonl(run/'records.jsonl');journal=read_jsonl(run/'generated.jsonl')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(json.loads(PLAN.read_text())['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
    result=verify(records,journal,anchors,refs,tok,run)
    comparisons=read_jsonl(run/'comparisons.jsonl')
    if len(comparisons)!=4 or any(x.get('status')!='identity_passed_bitwise' for x in comparisons):raise ValueError('B comparisons incomplete')
    files=[a.e1_acceptance,a.e1_started,a.e1_terminal,a.smoke_proof,a.smoke_started,a.smoke_terminal,a.run_started,a.run_terminal]
    files += [run/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','records.jsonl','generated.jsonl','comparisons.jsonl')]
    result.update(stage='skip_drop_identity',files_sha256={str(p.resolve()):sha(p) for p in files},
                  verifier_sha256=sha(__file__),array_sha256={x['arrays']['path']:x['arrays']['sha256'] for x in records},
                  kv_evidence='content fingerprints only; raw KV arrays not archived',
                  weight_evidence='manifest and four payload stats checked; verifier does not rehash payloads')
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
