"""Independent no-GT reconstruction of six retained t=.2 model arms.

Run only after the real controller and the pre-frozen formal verifier exit 0.
Actual KV contents and pixel_values are fingerprints, not archived tensors.
"""
import argparse,hashlib,json,os,re,struct,subprocess
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent.parent;R=B/'e1_independent_review';D=B/'e6/pixel_wavefront_model_t020_run_v2';A=B/'e6/pixel_wavefront_t020_accept_v2';O=D/'run_001';PRIOR=B/'e6/pixelprune_t020_smoke/run_smoke001/observations'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def term(directory,script):
 s=read(directory/'started.json');t=read(directory/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervised exit: '+str(directory))
 return s
def ints(values):return hashlib.sha256(struct.pack('<'+'q'*len(values),*values)).hexdigest()
def p3(prefix,selected,cfg):
 image=cfg['image_token_id'];vision=cfg['vision_start_token_id'];video=cfg['video_token_id'];points=[i for i,x in enumerate(prefix) if x==image]
 if video in prefix or len(points)!=1024 or points!=list(range(points[0],points[0]+1024)) or points[0]<1 or prefix[points[0]-1]!=vision or prefix.count(vision)!=1:raise ValueError('not one static image')
 start=points[0];n=len(prefix);tail=n-start-1024
 axes=[list(range(start))+[start]*1024+list(range(start+32,start+32+tail)),list(range(start))+[start+i//32 for i in range(1024)]+list(range(start+32,start+32+tail)),list(range(start))+[start+i%32 for i in range(1024)]+list(range(start+32,start+32+tail))]
 keep=list(range(start))+[start+i for i in selected]+list(range(start+1024,n));maxpos=max(map(max,axes));return start,keep,[[[axis[i] for i in keep]] for axis in axes],maxpos,[[maxpos+1-n]]
def checked_archive(rec):
 path=Path(rec['archive_path']).resolve()
 if not path.is_relative_to(O.resolve()) or sha(path)!=rec['archive_sha256']:raise ValueError('archive binding')
 names={'prefill_embeddings','prefill_mask','prefill_positions','visual_mask','prefill_logits','injected_ds0','injected_ds1','injected_ds2'};sizes={'torch.bfloat16':2,'torch.float32':4,'torch.int64':8,'torch.bool':1};data={}
 with np.load(path,allow_pickle=False) as z:
  if set(z.files)!=names or set(rec['prefill_tensors'])!=names:raise ValueError('archive inventory')
  for name in names:
   arr=z[name];meta=rec['prefill_tensors'][name];shape=list(meta['shape']);shape[-1]*=sizes[meta['dtype']]
   if arr.dtype!=np.uint8 or list(arr.shape)!=shape or hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()!=meta['sha256']:raise ValueError('archive tensor metadata '+name)
   data[name]=arr.copy()
 mask=data['visual_mask'].view(np.bool_).reshape(rec['visual_mask'].__len__(),-1)
 positions=data['prefill_positions'].view(np.int64).reshape(rec['prefill_tensors']['prefill_positions']['shape'])
 attention=data['prefill_mask'].view(np.int64).reshape(rec['prefill_tensors']['prefill_mask']['shape'])
 if mask.tolist()!=rec['visual_mask'] or positions.tolist()!=rec['prefill_position_ids'] or attention.tolist()!=rec['prefill_mask']:raise ValueError('archived mask/position bytes')
 chosen=data['prefill_embeddings'][mask];
 if hashlib.sha256(np.ascontiguousarray(chosen).tobytes()).hexdigest()!=rec['main_visual']['sha256']:raise ValueError('actual prefill visual bytes')
 if rec['prefill_tensors']['prefill_embeddings']!=rec['prefill_embeddings'] or rec['prefill_tensors']['prefill_logits']!=rec['prefill_logits'] or [rec['prefill_tensors']['injected_ds'+str(i)] for i in range(3)]!=rec['injected_deepstack']:raise ValueError('prefill metadata binding')
 logits=data['prefill_logits'].view(np.uint16)
 if rec['prefill_logits']['dtype']!='torch.bfloat16' or np.any((logits&0x7f80)==0x7f80):raise ValueError('nonfinite BF16 logits')
 return path,sha(path)
def verify_docker(doc,cmd,job,out):
 def vals(name):return [cmd[i+1] for i,x in enumerate(cmd[:-1]) if x==name]
 ent=vals('--entrypoint');user=vals('--user');work=vals('--workdir');name=vals('--name')
 if cmd[:2]!=['docker','run'] or any(len(x)!=1 for x in (ent,user,work,name)):raise ValueError('container command')
 i=cmd.index('--entrypoint')
 if doc['Image']!=cmd[i+2] or doc['Config']['Entrypoint']!=ent or doc['Config']['Cmd']!=[str(D/'worker_entry.py'),'--task',job['task'],'--qid',str(job['qid']),'--arm',job['arm'],'--out',str(out/'observations')] or doc['Config']['Cmd']!=cmd[i+3:] or doc['Config']['User']!=user[0] or doc['Config']['WorkingDir']!=work[0] or doc['Name'].lstrip('/')!=name[0]:raise ValueError('container environment')
 env=dict(x.split('=',1) for x in doc['Config']['Env'] if '=' in x)
 for item in vals('--env'):
  k,v=item.split('=',1)
  if env.get(k)!=v:raise ValueError('container environment variable')
 mounts={}
 for item in vals('--mount'):
  parts=item.split(',');fields=dict(x.split('=',1) for x in parts if '=' in x)
  if fields.get('type')!='bind':raise ValueError('unexpected mount type')
  mounts[fields['dst']]=(fields['src'],'readonly' not in parts)
 actual={x['Destination']:(x['Source'],x['RW']) for x in doc['Mounts'] if x['Type']=='bind'}
 if mounts!=actual:raise ValueError('mounts')
 devices={x.split(':')[0] for x in vals('--device')};actual_devices={x['PathOnHost'] for x in doc['HostConfig']['Devices'] or []}
 if devices!=actual_devices or not any('/dev/nvidia' in x for x in devices):raise ValueError('GPU devices')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=R/'pixel_wavefront_t020_run_v2_result_review.json' or a.out.exists():raise ValueError('output path')
 term(D/'monitor_run001',D/'controller.py');term(A/'monitor_verify001',A/'verify.py');formal=read(A/'verification001.json');acceptplan=read(A/'verification_plan001.json');plan=read(D/'execution_plan001.json')
 if formal.get('status')!='passed' or (formal.get('images'),formal.get('jobs'),formal.get('pairs'))!=(3,6,3) or formal.get('GT_opened') is not False or formal.get('scoring_performed') is not False or formal.get('errors')!=[] or formal.get('verifier_sha256')!=sha(A/'verify.py') or formal.get('verification_plan_sha256')!=sha(A/'verification_plan001.json') or acceptplan.get('execution_plan_sha256')!=sha(D/'execution_plan001.json') or acceptplan.get('verifier_sha256')!=sha(A/'verify.py'):raise ValueError('formal acceptance')
 for key,path in [('run_started_sha256',D/'monitor_run001/started.json'),('run_terminal_sha256',D/'monitor_run001/terminal.json'),('snapshot_manifest_sha256',O/'sources/manifest.json'),('journal_sha256',O/'journal.jsonl')]:
  if formal.get(key)!=sha(path):raise ValueError('formal binding '+key)
 snapshot=read(O/'sources/manifest.json')
 if snapshot.get('status')!='passed' or set(snapshot['sources'])!=set(plan['source_sha256']):raise ValueError('source inventory')
 for path,digest in plan['source_sha256'].items():
  item=snapshot['sources'][path]
  if item['sha256']!=digest or sha(item['path'])!=digest:raise ValueError('source snapshot '+path)
 if read(O/'weights_before.json')!=read(O/'weights_after.json') or read(O/'complete.json')!={'status':'passed','jobs':6,'weights_stat_unchanged':True,'GT_opened':False,'scoring_performed':False}:raise ValueError('run complete / weight stats')
 prior=read(R/'pixel_t020_result_review.json');term(R/'monitor_pixel_t020_result',R/'review_pixel_t020_result.py')
 if prior.get('status')!='passed' or prior.get('independent_review') is not True or prior.get('images')!=4 or prior.get('GT_opened') is not False or prior.get('reviewer_sha256')!=sha(R/'review_pixel_t020_result.py'):raise ValueError('prior reference review')
 for path,digest in prior['files_sha256'].items():
  if sha(path)!=digest:raise ValueError('prior reference file changed')
 cfg=read('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/config.json');events=[json.loads(x) for x in (O/'journal.jsonl').read_text().splitlines()];jobs=plan['jobs']
 if len(jobs)!=6 or len(events)!=13 or events[-1]!={'event':'all_arms_completed','images':3,'jobs':6}:raise ValueError('arm denominator')
 records={};archives={};containers=set();files={}
 for i,job in enumerate(jobs):
  start,end=events[i*2:i*2+2]
  if start!={'event':'arm_invoked','job':job} or end.get('event')!='arm_result' or end.get('job')!=job:raise ValueError('ordered arm journal')
  arm=O/f'arm_{i:03d}';launch=read(arm/'launch.json');ex=read(arm/'exit.json');cid=ex['container_identity']['container_id'];run=end['result']
  if cid in containers or len(cid)!=64 or run.get('container_id')!=cid or run.get('returncode')!=0 or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None:raise ValueError('arm exit')
  containers.add(cid);doc=json.loads(subprocess.run(['docker','inspect',cid],text=True,capture_output=True,check=True,timeout=15).stdout)[0]
  if doc['Id']!=cid or doc['State']['Running'] is not False or doc['State']['Pid']!=0 or doc['State']['ExitCode']!=0 or doc['State']['OOMKilled'] is not False or doc['HostConfig']['NetworkMode']!='none':raise ValueError('actual retained Docker state')
  verify_docker(doc,launch['command'],job,arm)
  obs=arm/'observations';rec=read(obs/'record.json');raw=read(obs/'returned.json');small=[json.loads(x) for x in (obs/'events.jsonl').read_text().splitlines()]
  if [x.get('event') for x in small]!=['generation_started','selector_returned','generation_returned'] or any(x.get('task')!=job['task'] or str(x.get('qid'))!=str(job['qid']) or x.get('arm')!=job['arm'] for x in small) or small[1]['indices']!=rec['selected_indices']:raise ValueError('selector and exposure journal')
  if any(str(rec.get(k))!=str(job[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) or rec['path']!=job['arm'] or rec['status']!='completed' or rec['GT_opened'] is not False or rec['scoring_performed'] is not False:raise ValueError('arm input identity')
  reference=PRIOR/(job['task']+'_'+str(job['qid'])+'.json');baseline=read(reference);generated=read(PRIOR/(job['task']+'_'+str(job['qid'])+'_generated.json'))
  tokens=rec['output_ids'];selected=rec['selected_indices'];full=raw['full_ids'];n=raw['input_length']
  if any(prior['files_sha256'].get(str(x))!=sha(x) for x in (reference,PRIOR/(job['task']+'_'+str(job['qid'])+'_generated.json'))) or selected!=baseline['selected_indices'] or selected!=baseline['cpu_expected_indices'] or tokens!=generated['generated_token_ids'] or full[n:]!=tokens or rec['processor']!=baseline['processor_identity']['processor'] or rec['generation_scalars']!=baseline['generation_kwargs'] or rec['eos_ids']!=baseline['eos_token_ids']:raise ValueError('accepted prior selector/input/output')
  if any(str(baseline.get(k))!=str(job[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) or (raw.get('task'),str(raw.get('qid')),raw.get('arm'))!=(job['task'],str(job['qid']),job['arm']) or n!=rec['original_sequence_length'] or not selected or selected!=sorted(set(selected)) or any(type(x) is not int or not 0<=x<1024 for x in selected):raise ValueError('prior and arm identity')
  if ints(full[:n])!=rec['processor']['input_ids']['sha256'] or ints([1]*n)!=rec['processor']['attention_mask']['sha256'] or ints([1,64,64])!=rec['processor']['image_grid_thw']['sha256']:raise ValueError('processor input bytes')
  start_index,keep,positions,maxpos,delta=p3(full[:n],selected,cfg)
  if rec['prefill_position_ids']!=positions or rec['original_max_position']!=maxpos or rec['rope_delta']!=delta:raise ValueError('independent P3 reconstruction')
  mask=[False]*len(keep);mask[start_index:start_index+len(selected)]=[True]*len(selected)
  if rec['visual_mask']!=[mask] or rec['prefill_mask']!=[[1]*len(keep)] or len(rec['injected_deepstack'])!=3 or not tokens or not (tokens[-1] in rec['eos_ids'] or len(tokens)==2048):raise ValueError('mask, DS or termination')
  if len(rec['decode_trace'])!=len(tokens) or len(rec['cache_fingerprints'])!=len(tokens):raise ValueError('decode inventory')
  for step,(trace,layers) in enumerate(zip(rec['decode_trace'],rec['cache_fingerprints'])):
   if trace['rope_delta']!=delta or trace['position_ids']!=(positions if step==0 else [[[maxpos+step]]]*3) or trace['cache_position']!=(list(range(len(keep))) if step==0 else [n+step-1]) or len(layers)!=36:raise ValueError('decode trajectory')
   for pair in layers:
    if set(pair)!={'key','value'}:raise ValueError('KV pair')
    for meta in pair.values():
     if meta.get('dtype')!='torch.bfloat16' or meta.get('shape')!=[1,8,len(keep)+step,128] or not re.fullmatch('[0-9a-f]{64}',meta.get('sha256','')):raise ValueError('KV fingerprint')
  path,digest=checked_archive(rec);archives[str(path)]=digest
  if formal['archive_sha256'].get(str(path))!=digest:raise ValueError('formal archive binding')
  for f in (arm/'launch.json',arm/'exit.json',arm/'container_identity.json',obs/'record.json',obs/'returned.json',obs/'events.jsonl'):
   digest=sha(f);files[str(f)]=digest
   if formal['files_sha256'].get(str(f))!=digest:raise ValueError('formal arm file binding')
  key=(job['task'],str(job['qid']));records.setdefault(key,{})[job['arm']]=rec
 shared=('task','qid','group_id','message_sha256','image_sha256','selected_indices','output_ids','eos_ids','generation_scalars','processor','original_sequence_length','original_max_position','prefill_mask','prefill_position_ids','visual_mask','main_visual','injected_deepstack','prefill_embeddings','prefill_logits','prefill_tensors','rope_delta','decode_trace','cache_fingerprints')
 if len(records)!=3 or len(archives)!=6 or len(containers)!=6 or any(set(pair)!={'author','wavefront'} or any(pair['author'][k]!=pair['wavefront'][k] for k in shared) for pair in records.values()):raise ValueError('six-arm paired identity')
 proof={'status':'passed','independent_review':True,'images':3,'jobs':6,'pairs':3,'formal_sha256':sha(A/'verification001.json'),'formal_started_sha256':sha(A/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(A/'monitor_verify001/terminal.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'reviewer_sha256':sha(__file__),'archive_sha256':archives,'files_sha256':files,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'timing_claims':False,'limitations':['posthoc DEV 3 anchors only, no latency or quality conclusion','36-layer KV content only fingerprints, not archived raw cache','actual pixel_values represented by processor fingerprint, not archived full tensor','weight metadata comparison cannot exclude transient replacement'],'errors':[]}
 with a.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('passed')
if __name__=='__main__':main()
