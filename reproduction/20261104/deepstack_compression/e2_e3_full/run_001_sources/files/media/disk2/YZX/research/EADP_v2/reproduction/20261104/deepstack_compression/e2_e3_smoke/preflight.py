"""Host-run real-container preflight; must be independently reviewed before GPU."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from local_imports import load_local
gate=load_local('gate');HERE=gate.HERE;BASE=gate.BASE;check_approval=gate.check_approval;sha=gate.sha
sys.path.insert(0,str(BASE/'e4'))
from identity_worker import PLAN

def valid_result(result,plan):
    root=Path(plan['author_root'])
    expected={'model_adapter':BASE/'e2_e3_contract/model_adapter.py','intervention_runner':HERE/'intervention_runner.py',
              'cache_fingerprints':HERE/'cache_fingerprints.py','native_pruner':root/'model/pruner.py',
              'author_wrapper':root/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py'}
    return (result.get('status')=='preflight_passed' and result.get('model_loaded') is False and
            result.get('predictions') is False and result.get('cuda_initialized') is False and
            result.get('imports')=={k:str(v.resolve()) for k,v in expected.items()} and result.get('versions')==plan['expected_versions'])

def main():
    p=argparse.ArgumentParser()
    for name in ('e1-acceptance','e1-started','e1-terminal'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();check_approval(preflight=True)
    plan=json.loads(PLAN.read_text());sys.path.insert(0,str(PLAN.parent))
    from run_panel import container_command
    out=HERE/'preflight_002';out.mkdir(exist_ok=False);name='eadp-e2e3-preflight-002';worker=HERE/'smoke_worker.py'
    args=[str(worker),'--execute','--preflight-only','--e1-acceptance',str(a.e1_acceptance.resolve()),'--e1-started',str(a.e1_started.resolve()),'--e1-terminal',str(a.e1_terminal.resolve()),'--out',str(out)]
    command=container_command(plan,args,True,name)
    if command.count('--rm')!=1:raise ValueError('unexpected container policy')
    command.remove('--rm')
    for device in ('/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'):
        i=command.index(device);del command[i-1:i+1]
    command[command.index('CUDA_VISIBLE_DEVICES=0')]='CUDA_VISIBLE_DEVICES='
    with (out/'launch.json').open('x') as f:json.dump({'command':command,'worker_sha256':sha(worker)},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    start=time.time();timeout=False
    try:
        process=subprocess.run(command,capture_output=True,text=True,timeout=600);code=process.returncode;stdout=process.stdout;stderr=process.stderr
    except subprocess.TimeoutExpired as e:
        timeout=True;code=None;stdout=str(e.stdout or '');stderr=str(e.stderr or '')
        subprocess.run(['docker','stop','--time','5',name],capture_output=True,timeout=20)
    (out/'stdout.log').write_text(stdout);(out/'stderr.log').write_text(stderr)
    inspected=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=20)
    c=json.loads(inspected.stdout)[0] if inspected.returncode==0 else {};state=c.get('State',{})
    try:result=json.loads(stdout.strip().splitlines()[-1])
    except Exception:result={}
    passed=(not timeout and code==0 and state.get('Running') is False and state.get('ExitCode')==0 and
            valid_result(result,plan) and not (out/'frozen.json').exists())
    proof={'status':'passed' if passed else 'failed','returncode':code,'container_running':state.get('Running'),'container_exit_code':state.get('ExitCode'),'container_id':c.get('Id'),
           'worker_sha256':sha(worker),'host_source_sha256':sha(__file__),'launch_sha256':sha(out/'launch.json'),'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),'timed_out':timeout,'result':result,'start_unix':start,'end_unix':time.time()}
    with (HERE/'container_full_preflight_002.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
