"""Independent post-review of the 62-input reference no-GPU preflight."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parents[1];D=B/'e4_c_multibudget_references';I=B/'e1_independent_review';RUN=D/'run_preflight001';OUT=I/'e4c_multibudget_references_preflight001_result_review.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(folder,script,out=None):
 s=read(folder/'started.json');t=read(folder/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/c[1]).resolve()!=script.resolve():raise ValueError('supervisor mismatch')
 if out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('supervisor output')
 return s
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def run():
 proof=I/'e4c_multibudget_references_worker_preflight_review.json';code=I/'review_e4c_multibudget_references_worker_preflight.py';term(I/'monitor_e4c_multibudget_references_worker_preflight',code,proof);r=read(proof);plan=read(D/'execution_plan003.json')
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('scope')!='references_cpu_preflight_only' or r.get('plan_sha256')!=sha(D/'execution_plan003.json') or r.get('source_sha256')!=plan['source_sha256'] or r.get('reviewer_sha256')!=sha(code) or r.get('GT_opened') is not False or r.get('predictions_opened') is not True or r.get('errors')!=[]:raise ValueError('preflight source review')
 for p,h in plan['source_sha256'].items():
  if sha(p)!=h:raise ValueError('source changed '+p)
 mon=D/'monitor_preflight001';s=term(mon,D/'launch.py');launch=read(RUN/'launch.json');ex=read(RUN/'exit.json');w=read(RUN/'worker_preflight.json');pp=read(RUN/'processor_preflight.json')
 if s['command'][-3:]!=['--run-id','001','--preflight-only'] or s['timeout_seconds']!=900 or s['grace_seconds']!=90 or launch['controller']!=ex['controller'] or launch['controller']['pid']!=s['worker_pid'] or launch.get('plan_sha256')!=sha(D/'execution_plan003.json') or launch.get('worker_sha256')!=sha(D/'worker.py') or launch.get('launcher_sha256')!=sha(D/'launch.py') or launch.get('approval_bundle') is not None or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('client_exited_while_container_running') is not False:raise ValueError('launcher/exit identity')
 if w.get('status')!='passed' or w.get('plan_sha256')!=sha(D/'execution_plan003.json') or w.get('processor_preflight_sha256')!=sha(RUN/'processor_preflight.json') or w.get('GT_opened') is not False or w.get('scoring_performed') is not False or w.get('predictions_opened') is not True or w.get('cuda_initialized') is not False or w.get('model_loaded') is not False:raise ValueError('worker preflight')
 if (pp.get('status')!='passed' or pp.get('sources_checked')!=len(plan['source_sha256']) or pp.get('inputs_checked')!=62 or pp.get('references_checked')!=248 or pp.get('jobs')!=1520 or pp.get('weight_payloads_hashed')!=4 or len(pp.get('weight_stats',{}))!=4 or pp.get('cuda_initialized') is not False or pp.get('model_loaded') is not False or pp.get('GT_opened') is not False or pp.get('predictions_opened') is not True or pp.get('GPU_authorized') is not False):raise ValueError('processor/weight preflight counts')
 refs=load('_independent_multibudget_preflight_refs',B/'e4_c_multibudget_worker/references.py').load_references();actual=pp['processor_inputs'];by={(x['task'],str(x['qid'])):x for x in actual}
 if len(by)!=62 or len(refs['rows'])!=62 or len(refs['records'])!=248:raise ValueError('input/reference denominator')
 for row in refs['rows']:
  x=by[(row['task'],str(row['qid']))]
  if x['group_id']!=row['group_id'] or x['image_start']!=row['image_start'] or x['processor']!=row['processor_expected']:raise ValueError('processor identity')
 cid=ex['container_identity']['container_id'];obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=10,check=True).stdout)[0];cfg=obj['Config'];host=obj['HostConfig'];st=obj['State'];cmd=launch['command'];image=obj['Image']
 if obj['Id']!=cid or obj['Name']!='/'+launch['container_name'] or st['Running'] or st['Pid']!=0 or st['ExitCode']!=0 or st['OOMKilled'] or cfg['Image']!=image or image!=ex['container_identity']['image'] or host.get('Devices') or host.get('DeviceRequests') or cfg['Entrypoint']!=[cmd[cmd.index('--entrypoint')+1]] or cfg['Cmd']!=cmd[cmd.index(image)+1:] or cfg['User']!=cmd[cmd.index('--user')+1] or cfg['WorkingDir']!=cmd[cmd.index('--workdir')+1] or host['NetworkMode']!=cmd[cmd.index('--network')+1]:raise ValueError('actual docker identity')
 for n,v in enumerate(cmd[:-1]):
  if v=='--env':
   key=cmd[n+1].split('=',1)[0]
   if [x for x in cfg['Env'] if x.startswith(key+'=')]!=[cmd[n+1]]:raise ValueError('docker environment')
 mounts=[]
 for n,v in enumerate(cmd[:-1]):
  if v=='--mount':
   chunks=cmd[n+1].split(',');d=dict(x.split('=',1) for x in chunks if '=' in x);mounts.append((str(Path(d['src']).resolve()),d['dst'],'readonly' not in chunks))
 actual=[(str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind']
 if len(actual)!=len(obj['Mounts']) or sorted(actual)!=sorted(mounts):raise ValueError('actual binds')
 return {'status':'passed','independent_review':True,'preflight_only':True,'GT_opened':False,'predictions_opened':True,'scoring_performed':False,'GPU_authorized':False,'images':62,'references':248,'weights_verified':4,'source_count':len(plan['source_sha256']),'plan_sha256':sha(D/'execution_plan003.json'),'preflight_review_sha256':sha(proof),'worker_preflight_sha256':sha(RUN/'worker_preflight.json'),'processor_preflight_sha256':sha(RUN/'processor_preflight.json'),'launch_sha256':sha(RUN/'launch.json'),'exit_sha256':sha(RUN/'exit.json'),'started_sha256':sha(mon/'started.json'),'terminal_sha256':sha(mon/'terminal.json'),'reviewer_sha256':sha(__file__),'errors':[],'limitations':['Only existing DEV prediction identity and real no-GPU processor/weight preflight; no new model generation or GPU identity.','248 fixed references and 12-pair identity must be generated and independently accepted before any timing or GPU release.']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=OUT:raise ValueError('unexpected output')
 result=run()
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
