"""Host-supervised, CPU-only execution of every worker gate before model loading."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from panel_v2_imports import load_local

HERE=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    launcher=load_local('panel_launch')
    launcher.check_panel_approval(preflight_only=True)
    out=HERE/'preflight_001';out.mkdir(exist_ok=False)
    receipt=launcher.prepare_native_receipt(out)
    prior=json.loads((HERE/'panel_002/launch.json').read_text());command=prior['command'][:]
    name='eadp-e4panel-full-preflight-001';command[command.index('--name')+1]=name
    for device in ('/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'):
        i=command.index(device);del command[i-1:i+1]
    command[command.index('CUDA_VISIBLE_DEVICES=0')]='CUDA_VISIBLE_DEVICES='
    command[command.index('--out')+1]=str(out);command.append('--preflight-only')
    launch={'command':command,'host_native_gate':receipt,'prior_launch_sha256':sha(HERE/'panel_002/launch.json')}
    with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    started=time.time();timed=False
    try:
        p=subprocess.run(command,text=True,capture_output=True,timeout=600)
        code=p.returncode;stdout=p.stdout;stderr=p.stderr
    except subprocess.TimeoutExpired as e:
        timed=True;code=None;stdout=str(e.stdout or '');stderr=str(e.stderr or '')
        subprocess.run(['docker','stop','--time','5',name],capture_output=True,timeout=20)
    (out/'stdout.log').write_text(stdout);(out/'stderr.log').write_text(stderr)
    probe=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=20)
    c=json.loads(probe.stdout)[0] if probe.returncode==0 else {};state=c.get('State',{});result=None
    try:result=json.loads(stdout.strip().splitlines()[-1])
    except Exception:pass
    passed=(not timed and code==0 and state.get('Running') is False and state.get('ExitCode')==0 and
            isinstance(result,dict) and result.get('status')=='preflight_passed' and
            result.get('model_loaded') is False and result.get('predictions') is False and
            not (out/'frozen.json').exists() and not (out/'generated.jsonl').exists())
    proof={'status':'passed' if passed else 'failed','returncode':code,'timed_out':timed,
           'container_running':state.get('Running'),'container_exit_code':state.get('ExitCode'),
           'container_id':c.get('Id'),'worker_sha256':sha(HERE/'panel_worker.py'),
           'host_script_sha256':sha(__file__),'launcher_sha256':sha(HERE/'panel_launch.py'),
           'launch_sha256':sha(out/'launch.json'),'receipt_sha256':sha(out/'host_native_gate.json'),
           'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),
           'result':result,'started_unix':started,'finished_unix':time.time()}
    with (HERE/'container_full_preflight_001.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':proof['status'],'returncode':code,'container_id':c.get('Id')}),flush=True)
    return 0 if passed else 1

if __name__=='__main__':sys.exit(main())
