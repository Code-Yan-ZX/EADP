"""Prepare full MMB root-exit proof after root actually observes PTY exit.

This script cannot inspect the root-owned exec session. The invoking root
must first see functions.write_stdin return exit 0 for that exact session.
No prediction, observation, GT, score, model or device inference is read.
"""
import argparse, hashlib, json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
EXPECTED_PLAN='09404c6685b0e148f3ea6373b3a79fd3842997ef14f5be2891d16d5a8784ea99'
EXPECTED_SESSION=86075

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(ok,msg):
    if not ok:raise ValueError(msg)
def identity(pid):
    try:
        stat=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    return dict(pid=pid,start_ticks=int(stat[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def container_check(container_id):
    p=subprocess.run(['docker','inspect',container_id],capture_output=True,text=True)
    if p.returncode:
        require(p.returncode==1 and ('No such object:' in p.stderr or 'No such container:' in p.stderr),'docker inspect failed without proving absence: '+p.stderr.strip())
        return dict(container_id=container_id,inspect_exit=p.returncode,absent=True,running=False)
    info=json.loads(p.stdout);require(len(info)==1 and info[0]['Id']==container_id,'docker identity mismatch')
    require(info[0]['State']['Running'] is False,'verifier container still running')
    return dict(container_id=container_id,inspect_exit=0,absent=False,running=False)
def gpu_pids():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return sorted({int(s.strip()) for s in p.stdout.splitlines() if s.strip()})

def prepare(root,session,exit_code,via,process_fn=identity,container_fn=container_check,gpu_fn=gpu_pids):
    root=Path(root);out=root/'execution'
    require(session==EXPECTED_SESSION and exit_code==0 and via=='functions.write_stdin','requires caller actual exit-0 observation for expected root PTY')
    paths=dict(plan=root/'plan.json',state=out/'state.json',release=out/'native_release.json',
               verification=out/'generation_verification.json',claim=root/'launch.claim.json',
               allocation=root/'root_gpu_allocation.json',worker_identity=out/'worker_identity.json',
               complete=out/'native_complete.json',anchor_gate=out/'anchor_gate.json',runtime=out/'runtime.json',
               worker_log=out/'worker.log',verifier_log=out/'verifier.log',verifier_cid=out/'verifier.cid')
    require(not (out/'worker_failure.json').exists(),'worker failure record exists')
    hashes={key:sha(path) for key,path in paths.items()}
    d={key:read(path) for key,path in paths.items() if key not in ('worker_log','verifier_log','verifier_cid')}
    p,s,release,report,claim,allocation,worker,done=(d[k] for k in ('plan','state','release','verification','claim','allocation','worker_identity','complete'))
    require(hashes['plan']==EXPECTED_PLAN and p['run_id']=='MMB_full_source_vs_question_only_run02','plan mismatch')
    require(all(x['plan_sha256']==hashes['plan'] for x in (s,release,report,claim,allocation,worker,done)),'plan identity mismatch')
    require(all(x['allocation_sha256']==hashes['allocation'] for x in (s,release,claim,worker)),'allocation hash mismatch')
    require(allocation['authorization']=='explicit_root_gpu_allocation' and allocation['run_id']==p['run_id'] and allocation['single_attempt'] is True and allocation['automatic_retry'] is False,'allocation mismatch')
    require(s['status']=='complete' and s['native_exit_code']==0 and s['verifier_exit_code']==0 and s['verifier_container_absent'] is True,'terminal state incomplete')
    require(release['status']=='native_exited_validation_pending' and release['native_exit_code']==0 and release['original_worker_exited'] is True and release['GPU_pid_absent'] is True,'native release incomplete')
    require(release['worker_log_sha256']==hashes['worker_log'] and s['verifier_log_sha256']==hashes['verifier_log'],'log hash mismatch')
    require(worker['identity']==release['worker']==s['worker'] and worker['parent']==release['controller']==s['controller']==claim['controller'],'process binding mismatch')
    require(report['success'] is True and report['n_rows']==4377 and report['n_generations']==8754 and report['n_anchor_rows']==60 and report['n_groups']==1176,'verification incomplete')
    require(report['CPU_only'] is True and report['model_loaded'] is False and report['CUDA_initialized'] is False and report['GT_read'] is False and report['scoring_started'] is False,'verification boundary mismatch')
    require(done['status']=='complete' and done['n_rows']==4377 and done['n_generations']==8754 and done['scoring'] is False,'generation incomplete')
    require(report['native_complete_sha256']==hashes['complete'] and report['native_release_sha256']==hashes['release'] and report['anchor_gate_sha256']==hashes['anchor_gate'] and report['runtime_sha256']==hashes['runtime'],'verification SHA chain mismatch')
    require(done['anchor_gate']['sha256']==hashes['anchor_gate'] and done['runtime']['sha256']==hashes['runtime'],'native SHA chain mismatch')
    require(d['anchor_gate']['passed'] is True and d['anchor_gate']['n_rows']==60 and d['anchor_gate']['n_generations']==120 and d['anchor_gate']['all_anchors_before_remaining'] is True,'anchor gate incomplete')
    for arm in ('source','question_only'):
        for name,key in (('predictions','prediction_sha256'),('observations','observation_sha256')):
            require(done['ordered'][arm][name]['sha256']==report[key][arm],'recorded ordered metadata mismatch')
    roles={k:s[k] for k in ('controller','worker','verifier_client')}
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(all(v['boot_id']==boot for v in roles.values()) and allocation['boot_id']==boot,'boot changed; process proof unavailable')
    checks=[]
    for role,old in roles.items():
        now=process_fn(old['pid']);require(now!=old,'original process still alive: '+role)
        checks.append(dict(role=role,original=old,original_process_absent=True,current_identity=now))
    cid=paths['verifier_cid'].read_text().strip();require(len(cid)==64 and all(x in '0123456789abcdef' for x in cid),'invalid verifier container ID')
    inspection=container_fn(cid);require(inspection['running'] is False,'verifier container still running')
    pids=gpu_fn();require(roles['worker']['pid'] not in pids,'original worker PID still in GPU list')
    require(all(sha(path)==hashes[key] for key,path in paths.items()),'terminal metadata changed while checking')
    return dict(root_pty_session=session,root_pty_actual_exit_code=exit_code,root_pty_exit_observed_by=via,
        pty_observation_note='Caller attests the root-owned PTY result; this script did not inspect the session.',
        verifier_container_not_running=True,verifier_container_inspection=inspection,
        state_sha256=hashes['state'],verifier_sha256=hashes['verification'],allocation_sha256=hashes['allocation'],
        plan_sha256=hashes['plan'],native_release_sha256=hashes['release'],native_complete_sha256=hashes['complete'],
        claim_sha256=hashes['claim'],worker_identity_sha256=hashes['worker_identity'],verifier_cid_sha256=hashes['verifier_cid'],
        process_checks=checks,gpu_pids_after=pids,created_utc=datetime.now(timezone.utc).isoformat())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root-pty-session',required=True,type=int);ap.add_argument('--root-observed-exit-code',required=True,type=int)
    ap.add_argument('--root-observed-via',required=True,choices=['functions.write_stdin']);args=ap.parse_args()
    target=HERE/'root_exit_proof.json';require(not target.exists(),'proof already exists')
    result=prepare(HERE,args.root_pty_session,args.root_observed_exit_code,args.root_observed_via)
    data=(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode()
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    except BaseException:
        target.unlink(missing_ok=True);raise
    print(json.dumps(dict(path=str(target),sha256=sha(target),root_pty_session=args.root_pty_session)))
if __name__=='__main__':main()
