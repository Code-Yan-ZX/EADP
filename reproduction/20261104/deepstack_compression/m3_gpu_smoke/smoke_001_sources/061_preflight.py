"""Real same-image, CPU-only test of all pre-model worker gates and imports."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from local_imports import local
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    gate=local('gate');approval=gate.check_approval(preflight_only=True);out=HERE/'preflight_001';out.mkdir(exist_ok=False)
    receipt=out/'host_upstream.json';host=gate.upstream(host=True)
    with receipt.open('x') as f:json.dump(host,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    prior=json.loads((ROOT/'e4_panel_v2/panel_003/launch.json').read_text());command=prior['command'][:]
    name='eadp-m3-preflight-001';command[command.index('--name')+1]=name
    for device in ('/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'):
        i=command.index(device);del command[i-1:i+1]
    command[command.index('CUDA_VISIBLE_DEVICES=0')]='CUDA_VISIBLE_DEVICES='
    index=command.index(str(ROOT/'e4_panel_v2/panel_worker.py'))
    command=command[:index]+[str(HERE/'worker.py'),'--execute','--preflight-only','--out',str(out)]
    launch={'command':command,'host_upstream':{'path':str(receipt.resolve()),'sha256':sha(receipt)},
            'upstream_launch_sha256':sha(ROOT/'e4_panel_v2/panel_003/launch.json'),'approval_bundle':approval}
    with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    start=time.time();timed=False
    try:
        p=subprocess.run(command,text=True,capture_output=True,timeout=600);code=p.returncode;stdout=p.stdout;stderr=p.stderr
    except subprocess.TimeoutExpired as exc:
        timed=True;code=None;stdout=str(exc.stdout or '');stderr=str(exc.stderr or '')
        subprocess.run(['docker','stop','--time','5',name],capture_output=True,timeout=20)
    (out/'stdout.log').write_text(stdout);(out/'stderr.log').write_text(stderr)
    q=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=20)
    obj=json.loads(q.stdout)[0] if q.returncode==0 else {};state=obj.get('State',{});result=None
    try:result=json.loads(stdout.strip().splitlines()[-1])
    except Exception:pass
    passed=(not timed and code==0 and state.get('Running') is False and state.get('ExitCode')==0 and isinstance(result,dict) and
        result.get('status')=='preflight_passed' and result.get('cuda_initialized') is False and result.get('model_loaded') is False and
        result.get('predictions') is False and result.get('GT_opened') is False and result.get('target_ids')==[100,200,300] and
        not (out/'events.jsonl').exists() and not (out/'result.json').exists())
    proof={'status':'passed' if passed else 'failed','returncode':code,'timed_out':timed,'container_running':state.get('Running'),
        'container_exit_code':state.get('ExitCode'),'container_id':obj.get('Id'),'worker_sha256':sha(HERE/'worker.py'),
        'preflight_sha256':sha(__file__),'launch_sha256':sha(out/'launch.json'),'receipt_sha256':sha(receipt),
        'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),'result':result,
        'started_unix':start,'finished_unix':time.time()}
    with (HERE/'container_preflight_001.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':proof['status'],'returncode':code,'container_id':obj.get('Id')}),flush=True)
    return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
