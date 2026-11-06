"""Review actual no-GPU latency worker preflight for the smoke launch."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;L=ROOT/'e4_c_latency';sys.path.insert(0,str(L))
import latency_admission,latency_gate,latency_worker
from identity_gate import sha
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==ROOT/'e1_independent_review/e4c_latency_smoke_prelaunch_review.json'
 paths={'cpu_proof':L/'cpu_check009/verification.json','cpu_started':L/'monitor_cpu_check009/started.json','cpu_terminal':L/'monitor_cpu_check009/terminal.json',
        'import_launch':L/'import_preflight006/launch.json','import_exit':L/'import_preflight006/exit.json','import_started':L/'monitor_import_preflight006/started.json','import_terminal':L/'monitor_import_preflight006/terminal.json',
        'worker_launch':L/'worker_preflight001/launch.json','worker_exit':L/'worker_preflight001/exit.json','worker_started':L/'monitor_worker_preflight001/started.json','worker_terminal':L/'monitor_worker_preflight001/terminal.json','worker_result':L/'worker_preflight001/worker_preflight.json'}
 cs=latency_admission.terminal(paths['cpu_started'],paths['cpu_terminal']);ins=latency_admission.terminal(paths['import_started'],paths['import_terminal'])
 latency_admission.supervised_run_id(cs,L/'cpu_check.py','cpu_check',paths['cpu_proof']);latency_admission.supervised_run_id(ins,L/'preflight_import.py','import_preflight',paths['import_launch'])
 cpu=read(paths['cpu_proof']);source={str(p):sha(p) for p in latency_worker.sources()};assert cpu['status']=='passed' and cpu['returncode']==0 and cpu['source_sha256_before']==cpu['source_sha256_after'] and all(cpu['source_sha256_after'].get(k)==v for k,v in source.items())
 for stream in ('stdout','stderr'):assert cpu[stream+'_sha256']==sha(paths['cpu_proof'].parent/(stream+'.log'))
 log=(paths['cpu_proof'].parent/'stderr.log').read_text();assert 'Ran 14 tests' in log and log.rstrip().endswith('OK')
 launch=read(paths['import_launch']);ex=read(paths['import_exit']);latency_admission.cpu_container(launch,ex,ins,L/'import_probe.py')
 assert all(launch['source_sha256'].get(k)==v for k,v in source.items())
 import_log=(paths['import_launch'].parent/'docker.log').read_text().splitlines();observed=next(json.loads(x) for x in reversed(import_log) if x.startswith('{'))
 assert observed['status']=='passed' and observed['cuda_initialized'] is False and observed['model_loaded'] is False and observed['predictions'] is False and Path(observed['runner']).resolve()==(L/'runner.py').resolve() and Path(observed['light_adapter']).resolve()==(L/'light_adapter.py').resolve()
 obj=json.loads(subprocess.check_output(['docker','inspect',ex['container_identity']['container_id']],text=True))[0];state=obj['State'];cmd=launch['command'];cfg=obj['Config'];host=obj['HostConfig'];image=ex['container_identity']['image']
 assert obj['Id']==ex['container_identity']['container_id'] and obj['Image']==image and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and state['OOMKilled'] is False and not host['Devices'] and not host['DeviceRequests'] and host['NetworkMode']=='none'
 assert cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['Cmd']==cmd[cmd.index(image)+1:] and cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1]
 identity=latency_gate.require_light_identity()
 prior=latency_admission.admit('smoke',preflight_only=True);ws=latency_admission.terminal(paths['worker_started'],paths['worker_terminal']);latency_admission.supervised_run_id(ws,L/'preflight_worker.py','worker_preflight',paths['worker_launch'])
 wl,we,wr=read(paths['worker_launch']),read(paths['worker_exit']),read(paths['worker_result']);latency_admission.cpu_container(wl,we,ws,L/'latency_worker.py')
 assert wl['approval_bundle']==prior and wr['status']=='passed' and wr['stage']=='smoke' and wr['cuda_initialized'] is False and wr['model_loaded'] is False and wr['predictions'] is False and wr['source_sha256']==source and wr['light_identity']==identity and len(wr['weight_stats_before'])==4
 assert wl['source_sha256']==source and wl['command'].count('--preflight-only')==1 and wl['command'].count(str(L/'latency_worker.py'))==1
 actual=json.loads(subprocess.check_output(['docker','inspect',we['container_identity']['container_id']],text=True))[0];state=actual['State'];cfg=actual['Config'];host=actual['HostConfig'];wcmd=wl['command'];image=we['container_identity']['image']
 assert actual['Id']==we['container_identity']['container_id'] and actual['Image']==image and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and state['OOMKilled'] is False and not host['Devices'] and not host['DeviceRequests'] and host['NetworkMode']=='none'
 assert cfg['Entrypoint']==[wcmd[wcmd.index('--entrypoint')+1]] and cfg['Cmd']==wcmd[wcmd.index(image)+1:] and cfg['User']==wcmd[wcmd.index('--user')+1] and cfg['WorkingDir']==wcmd[wcmd.index('--workdir')+1]
 result={'status':'passed','scope':'latency_smoke_launch','independent_review':True,'GT_opened':False,'scoring_performed':False,'errors':[],
         'light_identity':identity,'source_sha256':source,'reviewer_sha256':sha(__file__),
         'limits':['Only a coordinator-scheduled bounded GPU smoke is authorized.','No empirical speed claim follows from preflight.']}
 for name,path in paths.items():result[name+'_path']=str(path.resolve());result[name+'_sha256']=sha(path)
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','source_count':len(source),'scope':result['scope']}))
if __name__=='__main__':main()
