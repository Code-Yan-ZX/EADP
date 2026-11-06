"""Review memory CPU preflight admission; no GPU run is authorized."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;M=ROOT/'e4_c_memory';sys.path.insert(0,str(M))
import admission
from identity_gate import sha

def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==admission.REVIEW_PROOF
 cpu=read(admission.CPU);source={str(p):sha(p) for p in admission.sources()};assert cpu['status']=='passed' and cpu['tests']==23 and cpu['source_sha256_before']==cpu['source_sha256_after']==source
 cs=admission.terminal(admission.CS,admission.CT);ins=admission.terminal(admission.IS,admission.IT)
 for started,script in ((cs,M/'cpu_check.py'),(ins,M/'preflight_import.py')):
  cmd=started['command'];cwd=Path(started['cwd']);assert [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script] and cmd.count('--run-id')==1 and cmd[cmd.index('--run-id')+1]=='002'
 for n,count in [('sampler',13),('worker',10)]:
  p=admission.CPU.parent/(n+'_stderr.log');log=p.read_text();assert cpu[n+'_stderr_sha256']==sha(p) and f'Ran {count} tests' in log and log.rstrip().endswith('OK')
 host=M/'host_preflight001_proof.json';old=M/'cpu_proof012.json'
 for proof in (read(host),read(old)):
  assert proof['status']=='passed' and proof['files_sha256'] and all(sha(p)==d for p,d in proof['files_sha256'].items())
 hp=read(host);assert hp['jobs_per_mode']=={'C-DROP':24,'C-ON':24,'C-SKIP':24} and hp['cuda_initialized'] is False and hp['model_loaded'] is False and all(sha(p)==d for p,d in hp['source_sha256'].items())
 hs=read(M/'monitor_host_preflight001/started.json');ht=read(M/'monitor_host_preflight001/terminal.json');assert hs['worker_pid']==hs['process_group']==ht['worker_pid'] and ht['reason']=='completed' and ht['worker_returncode']==0 and ht['remaining_process_group']==[] and Path(hs['cwd']).resolve()==M and 'prepare_memory("smoke")' in hs['command'][-1]
 launch=read(admission.IL);ex=read(admission.IE);admission.cpu_container(launch,ex,ins,M/'import_probe.py');assert launch['source_sha256']==source
 log=(admission.IL.parent/'docker.log').read_text().splitlines();v=next(json.loads(x) for x in reversed(log) if x.startswith('{'));assert v['status']=='passed' and v['cuda_initialized'] is False and v['model_loaded'] is False and v['predictions'] is False and v['worker_contract']==str((ROOT/'e4_c_latency/contract.py').resolve()) and v['sampler_contract']==str((M/'contract.py').resolve())
 obj=json.loads(subprocess.check_output(['docker','inspect',ex['container_identity']['container_id']],text=True))[0];state=obj['State'];cmd=launch['command'];cfg=obj['Config'];hostcfg=obj['HostConfig'];image=ex['container_identity']['image'];assert obj['Id']==ex['container_identity']['container_id'] and obj['Image']==image and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and state['OOMKilled'] is False and not hostcfg['Devices'] and not hostcfg['DeviceRequests'] and hostcfg['NetworkMode']=='none'
 assert cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['Cmd']==cmd[cmd.index(image)+1:] and cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1]
 mounts=[]
 for i,x in enumerate(cmd):
  if x=='--env':
   e=cmd[i+1];assert [v for v in cfg['Env'] if v.startswith(e.split('=',1)[0]+'=')]==[e]
  if x=='--mount':
   v=cmd[i+1].split(',');m=dict(x.split('=',1) for x in v if '=' in x);assert m['type']=='bind';mounts.append((str(Path(m['src']).resolve()),m['dst'],'readonly' not in v))
 actual=[(str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in obj['Mounts'] if m['Type']=='bind'];assert len(actual)==len(mounts)==len(obj['Mounts']) and sorted(actual)==sorted(mounts)
 out={'status':'passed','scope':'memory_cpu_container_preflight_only','independent_review':True,'GT_opened':False,'scoring_performed':False,'errors':[],'reviewer_sha256':sha(__file__),'source_sha256':source,'host_proof_sha256':sha(host),'cpu_proof_sha256':sha(old),'limits':['Only retained no-GPU worker preflight is authorized.','Real sampler/controller and GPU process identity remain a separate future admission.']}
 for name,path in [('suite_proof',admission.CPU),('suite_started',admission.CS),('suite_terminal',admission.CT),('import_launch',admission.IL),('import_exit',admission.IE),('import_started',admission.IS),('import_terminal',admission.IT)]:out[name+'_sha256']=sha(path)
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','source_count':len(source)}))
if __name__=='__main__':main()
