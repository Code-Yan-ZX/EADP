"""Independent review of batch-2 CPU-only real container preflight."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from review_m3batch2_preflight import ROOT,B,HERE,sha,read,req,supervisor

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args();out=args.out.resolve()
 req(out==(B/'independent_final_review.json').resolve(),'wrong output')
 cpu=B/'cpu_check_003/verification.json';cs=B/'monitor_cpu003/started.json';ct=B/'monitor_cpu003/terminal.json'
 supervisor(cs,ct,B/'cpu_validation.py',cpu.parent);c=read(cpu);src=c['source_sha256_after']
 req(c['status']=='passed' and c['source_sha256_before']==src and len(src)==147 and all(sha(p)==s for p,s in src.items()),'source drift')
 provisional=B/'independent_preflight_review.json';ps=HERE/'monitor_m3batch2_preflight/started.json';pt=HERE/'monitor_m3batch2_preflight/terminal.json'
 supervisor(ps,pt,HERE/'review_m3batch2_preflight.py',provisional);pr=read(provisional)
 req(pr['status']=='passed' and pr['independent_review'] is True and pr['preflight_only'] is True and pr['GT_opened'] is False and pr['gpu_used'] is False and pr['reviewer_sha256']==sha(HERE/'review_m3batch2_preflight.py') and pr['cpu_proof_sha256']==sha(cpu) and pr['source_sha256']==src,'provisional invalid')
 pp=B/'preflight_001.json';ss=B/'monitor_preflight001/started.json';tt=B/'monitor_preflight001/terminal.json';s,t=read(ss),read(tt);cmd=s['command'];cwd=Path(s['cwd'])
 req(t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name=='preflight.py']==[(B/'preflight.py').resolve()] and cmd[cmd.index('--run-id')+1]=='001' and cmd[cmd.index('--cpu-id')+1]=='003','preflight supervisor')
 p=read(pp);d=B/'preflight_001';launch=read(d/'launch.json');exit=read(d/'exit.json');receipt=read(d/'host_receipt.json');old=read(d/'approval_snapshot.json');docker=launch['command'];result=p['result'];plan=B/'execution_plan_002.json'
 req(p['status']=='passed' and p['container_config_verified'] is True and p['exit']==exit and exit['returncode']==0 and exit['container_exit_code']==0 and exit['container_stopped_verified'] is True and exit['interrupted'] is None and exit['timed_out'] is False and exit['errors']==[] and exit['client_exited_while_container_running'] is False,'preflight exit')
 req(all(p[k]==sha(d/f) for k,f in [('launch_sha256','launch.json'),('exit_sha256','exit.json'),('log_sha256','launcher.log'),('receipt_sha256','host_receipt.json'),('approval_snapshot_sha256','approval_snapshot.json')]) and p['cpu_proof_sha256']==sha(cpu) and p['preflight_sha256']==sha(B/'preflight.py'),'preflight hashes')
 req(old['review_proof']==str(provisional) and old['review_started']==str(ps) and old['review_terminal']==str(pt) and receipt['approval_sha256']==sha(d/'approval_snapshot.json') and receipt['review_sha256']==sha(provisional) and receipt['cpu_sha256']==sha(cpu) and receipt['plan_sha256']==sha(plan),'admission receipt')
 manifest=hashlib.sha256(json.dumps(src,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 req(receipt['source_count_verified']==147 and receipt['source_manifest_sha256']==manifest and result['source_count_verified']==147 and result['source_manifest_sha256']==manifest and (result['inputs_verified'],result['sample_verified'],result['weights_sha256_verified'])==(384,128,4) and all(result[x] is False for x in ['cuda_initialized','model_loaded','GT_opened','predictions_opened']) and result['prediction_journal_byte_hash_deferred'] is True,'preflight result')
 logs=(d/'launcher.log').read_text().splitlines();matches=[]
 for line in logs:
  try: obj=json.loads(line)
  except ValueError: continue
  if isinstance(obj,dict) and obj.get('status')=='passed' and 'sample_verified' in obj:matches.append(obj)
 req(matches==[result],'worker result absent or altered')
 expected=read(ROOT/'m3_gpu_smoke/preflight_001/launch.json')['command'][:];expected[expected.index('--name')+1]='eadp-m3-batch2-preflight-001';i=expected.index(str(ROOT/'m3_gpu_smoke/worker.py'));expected=expected[:i]+[str(B/'runtime_worker.py'),'--execute','--preflight-only','--out',str(d)]
 req(docker==expected and launch['worker_sha256']==sha(B/'runtime_worker.py') and launch['cpu_proof_sha256']==sha(cpu) and launch['host_receipt']=={'path':str((d/'host_receipt.json').resolve()),'sha256':sha(d/'host_receipt.json')},'command altered')
 info=json.loads(subprocess.run(['docker','inspect',p['container_id']],check=True,capture_output=True,text=True,timeout=30).stdout)[0];state=info['State'];config=info['Config'];host=info['HostConfig']
 image=read(ROOT.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')['container_image']
 req(info['Id']==p['container_id']==exit['container_identity']['container_id'] and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and not state['OOMKilled'] and info['Image']==image and not host.get('Devices') and not host.get('DeviceRequests') and host['NetworkMode']=='none' and config['Entrypoint']==[docker[docker.index('--entrypoint')+1]] and config['Cmd']==docker[docker.index(image)+1:] and config['User']==docker[docker.index('--user')+1] and config['WorkingDir']==docker[docker.index('--workdir')+1] and 'CUDA_VISIBLE_DEVICES=' in docker,'container identity mismatch')
 expected_mounts=[];env={}
 for i,x in enumerate(docker):
  if x=='--mount':
   fields=docker[i+1].split(','); values=dict(y.split('=',1) for y in fields if '=' in y)
   req(values['type']=='bind','unexpected mount');expected_mounts.append((str(Path(values['src']).resolve()),values['dst'],'readonly' not in fields and 'ro' not in fields))
  if x in ('--env','-e'):
   k,v=docker[i+1].split('=',1);req(k not in env,'duplicate env');env[k]=v
 actual=[(str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in info['Mounts'] if m['Type']=='bind'];actual_env=dict(x.split('=',1) for x in config['Env'] if x.split('=',1)[0] in env)
 req(len(actual)==len(info['Mounts'])==len(expected_mounts) and sorted(actual)==sorted(expected_mounts) and actual_env==env,'mounts or env mismatch')
 proof={'status':'passed','independent_review':True,'preflight_only':False,'reviewer_sha256':sha(__file__),'cpu_proof_sha256':sha(cpu),'cpu_started_sha256':sha(cs),'cpu_terminal_sha256':sha(ct),'plan_sha256':sha(plan),'source_sha256':src,'preflight_sha256':sha(pp),'preflight_started_sha256':sha(ss),'preflight_terminal_sha256':sha(tt),'GT_opened':False,'predictions_opened':False,'gpu_used':False,'scope':'CPU and real no-GPU container preflight reviewed; actual GPU requires parent scheduling and deadline/GPU checks'}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
