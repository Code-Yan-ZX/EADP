"""Review full-latency source and no-GPU import evidence; only CPU preflight is admitted."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;F=ROOT/'e4_c_latency_full';sys.path.insert(0,str(F))
import full_admission
from identity_gate import sha

def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==ROOT/'e1_independent_review/e4c_latency_full_prelaunch_review.json'
 paths={'cpu_proof':F/'cpu_check006/verification.json','cpu_started':F/'monitor_cpu_check006/started.json','cpu_terminal':F/'monitor_cpu_check006/terminal.json', 'import_launch':F/'import_preflight003/launch.json','import_exit':F/'import_preflight003/exit.json','import_started':F/'monitor_import_preflight003/started.json','import_terminal':F/'monitor_import_preflight003/terminal.json'}
 cs=full_admission.terminal(paths['cpu_started'],paths['cpu_terminal']);ins=full_admission.terminal(paths['import_started'],paths['import_terminal']);full_admission.run_id(cs,F/'cpu_check.py','cpu_check',paths['cpu_proof']);full_admission.run_id(ins,F/'preflight_import.py','import_preflight',paths['import_launch'])
 cpu=read(paths['cpu_proof']);source={str(p):sha(p) for p in full_admission.sources()};assert cpu['status']=='passed' and cpu['returncode']==0 and cpu['source_sha256_before']==cpu['source_sha256_after']==source and cpu['test_stderr_sha256']==sha(paths['cpu_proof'].parent/'stderr.log')
 log=(paths['cpu_proof'].parent/'stderr.log').read_text();assert 'Ran 9 tests' in log and log.rstrip().endswith('OK')
 launch=read(paths['import_launch']);ex=read(paths['import_exit']);full_admission.cpu_container(launch,ex,ins,F/'import_probe.py');assert launch['source_sha256']==source
 log=(paths['import_launch'].parent/'docker.log').read_text().splitlines();observed=next(json.loads(x) for x in reversed(log) if x.startswith('{'));assert observed['status']=='passed' and observed['cuda_initialized'] is False and observed['model_loaded'] is False and observed['predictions'] is False
 obj=json.loads(subprocess.check_output(['docker','inspect',ex['container_identity']['container_id']],text=True))[0];state=obj['State'];cmd=launch['command'];cfg=obj['Config'];host=obj['HostConfig'];image=ex['container_identity']['image']
 assert obj['Id']==ex['container_identity']['container_id'] and obj['Image']==image and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and state['OOMKilled'] is False and not host['Devices'] and not host['DeviceRequests'] and host['NetworkMode']=='none'
 assert cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['Cmd']==cmd[cmd.index(image)+1:] and cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1]
 expected=[]
 for i,x in enumerate(cmd):
  if x=='--env':
   e=cmd[i+1];assert [v for v in cfg['Env'] if v.startswith(e.split('=',1)[0]+'=')]==[e]
  if x=='--mount':
   bits=cmd[i+1].split(',');m=dict(x.split('=',1) for x in bits if '=' in x);assert m['type']=='bind';expected.append((str(Path(m['src']).resolve()),m['dst'],'readonly' not in bits))
 actual=[(str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in obj['Mounts'] if m['Type']=='bind'];assert len(expected)==len(actual)==len(obj['Mounts']) and sorted(expected)==sorted(actual)
 previous=full_admission.admit(preflight_only=True,host=True)
 paths.update({'worker_launch':F/'worker_preflight001/launch.json','worker_exit':F/'worker_preflight001/exit.json','worker_started':F/'monitor_worker_preflight001/started.json','worker_terminal':F/'monitor_worker_preflight001/terminal.json','worker_result':F/'worker_preflight001/worker_preflight.json'})
 ws=full_admission.terminal(paths['worker_started'],paths['worker_terminal']);full_admission.run_id(ws,F/'preflight_worker.py','worker_preflight',paths['worker_launch']);wl=read(paths['worker_launch']);we=read(paths['worker_exit']);wr=read(paths['worker_result']);full_admission.cpu_container(wl,we,ws,F/'full_worker.py')
 assert wl['approval_bundle']==previous and wl['source_sha256']==source and wr['status']=='passed' and wr['jobs']==5940 and wr['cuda_initialized'] is False and wr['model_loaded'] is False and wr['predictions'] is False and wr['GT_opened'] is False and wr['scoring_performed'] is False and wr['full_source_sha256']==source and len(wr['weight_stats_before'])==4
 obj2=json.loads(subprocess.check_output(['docker','inspect',we['container_identity']['container_id']],text=True))[0];st=obj2['State'];host2=obj2['HostConfig'];cfg2=obj2['Config'];cmd2=wl['command'];image2=we['container_identity']['image'];assert obj2['Id']==we['container_identity']['container_id'] and obj2['Image']==image2 and st['Running'] is False and st['Pid']==0 and st['ExitCode']==0 and st['OOMKilled'] is False and not host2['Devices'] and not host2['DeviceRequests'] and host2['NetworkMode']=='none'
 assert cfg2['Entrypoint']==[cmd2[cmd2.index('--entrypoint')+1]] and cfg2['Cmd']==cmd2[cmd2.index(image2)+1:] and cfg2['User']==cmd2[cmd2.index('--user')+1] and cfg2['WorkingDir']==cmd2[cmd2.index('--workdir')+1]
 expected2=[]
 for i,x in enumerate(cmd2):
  if x=='--env':
   e=cmd2[i+1];assert [v for v in cfg2['Env'] if v.startswith(e.split('=',1)[0]+'=')]==[e]
  if x=='--mount':
   bits=cmd2[i+1].split(',');m=dict(x.split('=',1) for x in bits if '=' in x);assert m['type']=='bind';expected2.append((str(Path(m['src']).resolve()),m['dst'],'readonly' not in bits))
 actual2=[(str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in obj2['Mounts'] if m['Type']=='bind'];assert len(expected2)==len(actual2)==len(obj2['Mounts']) and sorted(expected2)==sorted(actual2)
 pilot=full_admission.pilot_gate(host=True);assert wr['pilot_identity']==pilot

 out={'status':'passed','scope':'full_latency_launch','independent_review':True,'GT_opened':False,'scoring_performed':False,'errors':[],'pilot_identity':pilot,'source_sha256':source,'reviewer_sha256':sha(__file__),'limits':['Prelaunch correctness only; GPU scheduling remains with the coordinator.','Single-process warm-cache interleaved timing, not a per-configuration process-memory estimate.']}
 for name,path in paths.items():out[name+'_path']=str(path.resolve());out[name+'_sha256']=sha(path)
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','source_count':len(source),'scope':out['scope']}))
if __name__=='__main__':main()
