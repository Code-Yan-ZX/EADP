"""Pre-frozen CPU acceptance of the t=.2 three-anchor model identity run.
This is an identity diagnostic, not a latency or accuracy evaluation.
"""
import argparse,hashlib,importlib.util,json,os,re,subprocess,sys
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[2]; D=B/'e6/pixel_wavefront_model_t020_run'; P=B/'e6/pixel_wavefront_model_t020'; I=B/'e1_independent_review'; A=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(d,script):
 s=read(d/'started.json');t=read(d/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or t.get('worker_pid')!=s.get('worker_pid') or (cwd/cmd[1]).resolve()!=script.resolve():raise ValueError('supervisor '+str(d))
 return s,t
def require(cond,msg):
 if not cond:raise ValueError(msg)
def checked_array(arr,meta):
 sizes={'torch.bfloat16':2,'torch.float32':4,'torch.int64':8,'torch.bool':1};shape=list(meta['shape']);shape[-1]*=sizes[meta['dtype']]
 require(arr.dtype==np.uint8 and list(arr.shape)==shape and hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()==meta['sha256'],'array bytes')
def verify_docker(doc,cmd,job,out):
 def values(flag):return [cmd[i+1] for i,x in enumerate(cmd[:-1]) if x==flag]
 require(cmd[:2]==['docker','run'] and len(values('--entrypoint'))==len(values('--name'))==len(values('--user'))==len(values('--workdir'))==1,'docker arguments')
 ent=values('--entrypoint')[0];i=cmd.index('--entrypoint');require(doc['Image']==cmd[i+2] and doc['Config']['Entrypoint']==[ent] and doc['Config']['Cmd']==cmd[i+3:] and doc['Name'].lstrip('/')==values('--name')[0] and doc['Config']['User']==values('--user')[0] and doc['Config']['WorkingDir']==values('--workdir')[0],'container command')
 require(doc['Config']['Cmd']==[str(D/'worker_entry.py'),'--task',job['task'],'--qid',str(job['qid']),'--arm',job['arm'],'--out',str(out/'observations')],'container worker')
 env=dict(x.split('=',1) for x in doc['Config']['Env'] if '=' in x)
 for x in values('--env'):
  key,value=x.split('=',1);require(env.get(key)==value,'container env')
 expected_mounts={}
 for x in values('--mount'):
  t=x.split(',');v=dict(y.split('=',1) for y in t if '=' in y);require(v.get('type')=='bind','mount type');expected_mounts[v['dst']]=(v['src'],'readonly' not in t)
 mounts={x['Destination']:(x['Source'],x['RW']) for x in doc['Mounts'] if x['Type']=='bind'}
 require(mounts==expected_mounts,'bind mounts')
 expected_devices={x.split(':')[0] for x in values('--device')};actual_devices={x['PathOnHost'] for x in doc['HostConfig']['Devices'] or []};require(expected_devices==actual_devices and any('/dev/nvidia' in x for x in actual_devices),'GPU devices')
def validate_pair(a,b,expected,archives):
 fields=('task','qid','group_id','message_sha256','image_sha256','selected_indices','output_ids','eos_ids','generation_scalars','processor','original_sequence_length','original_max_position','prefill_mask','prefill_position_ids','visual_mask','main_visual','injected_deepstack','prefill_embeddings','prefill_logits','prefill_tensors','rope_delta','decode_trace','cache_fingerprints')
 require(a['path']=='author' and b['path']=='wavefront' and all(a.get(k)==b.get(k) for k in fields),'paired identity')
 for x in (a,b):
  for k in ('task','qid','group_id','message_sha256','image_sha256'):require(x[k]==expected['anchor'][k],'anchor '+k)
  ref=expected['expected'];require(x['status']=='completed' and x['GT_opened'] is False and x['scoring_performed'] is False and all(x[k]==ref[j] for k,j in [('selected_indices','selected_indices'),('output_ids','tokens'),('eos_ids','eos_ids'),('processor','processor'),('generation_scalars','generation_kwargs')]),'accepted input/output')
  n=len(x['selected_indices']);out=x['output_ids'];mask=x['visual_mask'];require(n>0 and x['selected_indices']==sorted(set(x['selected_indices'])) and len(mask)==1 and sum(mask[0])==n and len(x['injected_deepstack'])==3 and all(z['shape'][0]==n for z in x['injected_deepstack']),'selection/DS')
  require(len(x['decode_trace'])==len(x['cache_fingerprints'])==len(out)>0,'decode count')
  init=len(mask[0]);require(x['decode_trace'][0]['position_ids']==x['prefill_position_ids'],'prefill trace')
  for i,(trace,layers) in enumerate(zip(x['decode_trace'],x['cache_fingerprints'])):
   require(trace['rope_delta']==x['rope_delta'] and len(layers)==36,'delta/layers');after=init+i
   for layer in layers:
    require(set(layer)=={'key','value'},'KV inventory')
    for meta in layer.values():require(meta['shape']==[1,8,after,128] and meta['dtype']=='torch.bfloat16' and re.fullmatch('[0-9a-f]{64}',meta['sha256']) is not None,'KV fingerprint')
  archive=Path(x['archive_path']).resolve();require(archive.is_relative_to((D/'run_001').resolve()) and sha(archive)==x['archive_sha256'],'prefill archive')
  with np.load(archive,allow_pickle=False) as z:
   require(set(z.files)==set(x['prefill_tensors'])=={'prefill_embeddings','prefill_mask','prefill_positions','visual_mask','prefill_logits','injected_ds0','injected_ds1','injected_ds2'},'array inventory')
   for name in z.files:checked_array(z[name],x['prefill_tensors'][name])
  archives[str(archive)]=x['archive_sha256']

def run(out):
 verification=read(A/'verification_plan001.json');require(verification['verifier_sha256']==sha(__file__) and verification['test_sha256']==sha(A/'test_verify.py') and verification['execution_plan_sha256']==sha(D/'execution_plan001.json'),'frozen verification source')
 started,_=term(D/'monitor_run001',D/'controller.py');require(started['command'][-2:]==['--run-id','001'],'run argv')
 plan=read(D/'execution_plan001.json');approval=read(D/'independent_gpu_prelaunch_review.json');require(approval['status']=='passed' and approval['plan_sha256']==sha(D/'execution_plan001.json'),'GPU approval')
 manifest=read(D/'run_001/sources/manifest.json');require(manifest['status']=='passed' and set(manifest['sources'])==set(plan['source_sha256']),'snapshot inventory')
 for path,digest in plan['source_sha256'].items():
  item=manifest['sources'][path];require(item['sha256']==digest and sha(item['path'])==digest,'source snapshot')
 require(read(D/'run_001/weights_before.json')==read(D/'run_001/weights_after.json'),'weight metadata changed')
 completed=read(D/'run_001/complete.json');require(completed=={'status':'passed','jobs':6,'weights_stat_unchanged':True,'GT_opened':False,'scoring_performed':False},'completion')
 spec=importlib.util.spec_from_file_location('_t020_accept_refs',P/'reference_contract.py');refs=importlib.util.module_from_spec(spec);spec.loader.exec_module(refs)
 ref=refs.reference(I/'pixel_t020_result_review.json',I/'review_pixel_t020_result.py',I/'monitor_pixel_t020_result/started.json',I/'monitor_pixel_t020_result/terminal.json',B/'e6/pixelprune_t020_smoke/run_smoke001/observations')['images'];ref={(x['anchor']['task'],str(x['anchor']['qid'])):x for x in ref}
 events=[json.loads(x) for x in (D/'run_001/journal.jsonl').read_text().splitlines()];require(len(events)==13 and events[-1]=={'event':'all_arms_completed','images':3,'jobs':6},'journal inventory')
 records={};archives={};files={};cids=set()
 for i,job in enumerate(plan['jobs']):
  start,end=events[2*i:2*i+2];require(start=={'event':'arm_invoked','job':job} and end['event']=='arm_result' and end['job']==job,'job order')
  d=D/'run_001'/f'arm_{i:03d}';launch=read(d/'launch.json');ex=read(d/'exit.json');r=end['result'];cid=r['container_id'];require(launch['job']==job and cid not in cids and len(cid)==64 and re.fullmatch('[0-9a-f]{64}',cid),'container identity');cids.add(cid)
  require(r['returncode']==r['container_exit_code']==ex['returncode']==ex['container_exit_code']==0 and r['container_stopped_verified'] is ex['container_stopped_verified'] is True and r['timed_out'] is ex['timed_out'] is False and r['interrupted'] is ex['interrupted'] is None and r['result_status']=='passed' and ex['container_identity']['container_id']==cid,'container exit')
  doc=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=15).stdout)[0];require(doc['Id']==cid and doc['State']['Running'] is False and doc['State']['Pid']==0 and doc['State']['ExitCode']==0 and doc['State']['OOMKilled'] is False and doc['HostConfig']['NetworkMode']=='none','actual Docker');verify_docker(doc,launch['command'],job,d)
  o=d/'observations';rec=read(o/'record.json');returned=read(o/'returned.json');require((rec['task'],str(rec['qid']),rec['path'])==(job['task'],str(job['qid']),job['arm']) and returned['arm']==job['arm'] and returned['task']==job['task'] and str(returned['qid'])==str(job['qid']) and returned['input_length']==rec['original_sequence_length'] and returned['full_ids'][returned['input_length']:]==rec['output_ids'],'raw generation')
  ev=[json.loads(x) for x in (o/'events.jsonl').read_text().splitlines()];require([x['event'] for x in ev]==['generation_started','selector_returned','generation_returned'] and ev[1]['indices']==rec['selected_indices'],'event exposure')
  key=(job['task'],str(job['qid']));records.setdefault(key,{})[job['arm']]=rec
  for p in (d/'launch.json',d/'exit.json',d/'container_identity.json',o/'record.json',o/'returned.json',o/'events.jsonl'):files[str(p)]=sha(p)
 require(len(records)==3 and len(cids)==6,'denominator')
 for key,pair in records.items():require(set(pair)=={'author','wavefront'},'missing pair');validate_pair(pair['author'],pair['wavefront'],ref[key],archives)
 require(len(archives)==6,'archive denominator')
 result={'status':'passed','images':3,'jobs':6,'pairs':3,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'errors':[],'plan_sha256':sha(D/'execution_plan001.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'snapshot_manifest_sha256':sha(D/'run_001/sources/manifest.json'),'journal_sha256':sha(D/'run_001/journal.jsonl'),'verifier_sha256':sha(__file__),'verification_plan_sha256':sha(A/'verification_plan001.json'),'files_sha256':files,'archive_sha256':archives,'limitations':['Post-hoc three-anchor DEV identity only; not a model timing or quality result.','Actual 36-layer KV tensors were not archived; runtime fingerprints are compared, not independently reconstructed.','Prior no-GPU preflight reviewer authored the candidate; no independent implementation claim.','Weight file metadata before/after cannot rule out transient changes.']}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();require(a.out.resolve()==A/'verification001.json' and not a.out.exists(),'unexpected output');run(a.out)
