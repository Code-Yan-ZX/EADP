"""Independent 32-image component acceptance, after formal exit only; no GT."""
import argparse,collections,hashlib,json,math,subprocess
from pathlib import Path
import torch
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_microbench_full';R=D/'full_001';A=B/'e1_independent_review'
def get(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def term(d):
 s=get(d/'started.json');t=get(d/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];return s
def meta(t):
 x=t.detach().contiguous().cpu();return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}
def tree(x):
 if torch.is_tensor(x):return dict(type='tensor',**meta(x))
 if isinstance(x,(list,tuple)):return {'type':'tuple' if isinstance(x,tuple) else 'list','items':[tree(v) for v in x]}
 if isinstance(x,dict):return {'type':'dict','items':{k:tree(x[k]) for k in sorted(x)}}
 if x is None or type(x) in (str,int,bool):return {'type':type(x).__name__,'value':x}
 if type(x) is float and math.isfinite(x):return {'type':'float','value':x}
 raise ValueError('unsupported output tree')
def fp(x):return hashlib.sha256(json.dumps(tree(x),sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();formal=get(D/'full_001_verification.json');assert formal['status']=='passed' and formal['images']==32 and formal['jobs']==1920 and formal['warmups']==288 and formal['diagnostic_generations']==32 and formal['GT_opened'] is False and formal['scoring_performed'] is False
term(D/'monitor_verify_full001');s=term(D/'monitor_full001');term(A/'monitor_component_full_final');assert formal['verifier_sha256']==sha(D/'full_verify.py')
for f,h in list(formal['source_sha256'].items())+list(formal['files_sha256'].items())+list(formal['array_sha256'].items()):assert sha(f)==h,f
l=get(R/'launch.json');ex=get(R/'exit.json');done=get(R/'worker_complete.json');assert l['controller']['pid']==s['worker_pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['client_exited_while_container_running'] is False and ex['gpu_pids_after']==[] and done['status']=='completed'
ins=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=ins['State'];assert ins['Id']==ex['last_inspect']['id'] and ins['Image']==ex['container_identity']['image'] and not st['Running'] and st['Pid']==0 and st['ExitCode']==0 and not st['OOMKilled'];assert ins['Config']['Cmd']==[str(D/'full_worker.py'),'--out',str(R)] and ins['HostConfig']['NetworkMode']=='none'
devices=ins['HostConfig']['Devices'];assert {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in devices}=={(x,x,'rwm') for x in ['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']}
cmd=l['command'];mount=[]
for i,c in enumerate(cmd[:-1]):
 if c=='--mount':
  v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mount.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
 if c=='--env':assert cmd[i+1] in ins['Config']['Env']
assert sorted(mount)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in ins['Mounts'] if x['Type']=='bind')
e1=get(B/'e1/e1full001_completion.json');j=Path(e1['files']['journal']['path']);assert sha(j)==e1['files']['journal']['sha256'];schedule=get(B/'e4_c_microbench/schedule001.json');identities=schedule['input_identities'];expected={(r['task'],str(r['qid'])):r for r in identities};assert len(expected)==32;refs={}
with j.open() as f:
 for line in f:
  x=json.loads(line);key=(x.get('task'),str(x.get('qid')))
  if key in expected and x.get('arm')=='P3-D1' and x.get('forced') is False:assert key not in refs;refs[key]=x
assert len(refs)==32 and sha(j)==e1['files']['journal']['sha256'];events=[json.loads(z) for z in (R/'journal.jsonl').read_text().splitlines()];assert len(formal['array_sha256'])==32
for i,identity in enumerate(identities):
 key=(identity['task'],str(identity['qid']));ref=refs[key];assert all(str(ref[k])==str(identity[k]) for k in ('task','qid','group_id','message_sha256'))
 chunk=[e for e in events if e.get('image_index')==i];assert collections.Counter(e['status'] for e in chunk)=={'image_started':1,'image_completed':1,'capture_generated':1,'capture_cleanup':1,'archive_written':1,'capture_identity_passed':1,'reference_fingerprints':1,'warmup_validated':9,'measured':60}
 for status in ('image_started','image_completed','capture_generated','capture_identity_passed'):
  v=[e for e in chunk if e['status']==status][0];assert all(str(v[k])==str(identity[k]) for k in ('task','qid','group_id'))
 assert chunk[0]['status']=='image_started' and chunk[-1]['status']=='image_completed';clean=[e for e in chunk if e['status']=='capture_cleanup'][0];assert clean['passed'] and clean['capture_completed']
 path=R/('image_%03d'%i)/'reference.pt';assert str(path) in formal['array_sha256'];data=torch.load(path,map_location='cpu',weights_only=True);x=data['actual'];assert {k:meta(x['processor'][k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}==ref['processor'];v=x['visual'];assert meta(v[0])==ref['visual_sha256'] and len(v[1])==3 and [meta(z) for z in v[1]]==ref['deepstack_full_sha256']
 ordered=x['ordered'].reshape(-1).tolist();assert len(ordered)==len(set(ordered))==256 and ordered==ref['ordered_ids'] and sorted(ordered)==ref['spatial_ids'];pre=x['prefill'];assert meta(pre['inputs_embeds'])==ref['prefill_embed_sha256'] and meta(pre['attention_mask'])==ref['attention_mask_sha256'] and pre['position_ids'].tolist()==ref['prefill_position_ids'] and x['rope_delta'].tolist()==ref['rope_delta'] and pre['visual_pos_masks'].tolist()==ref['injected_mask'] and [meta(z) for z in pre['deepstack_visual_embeds']]==ref['injected_deepstack']
 fps={k:fp(v) for k,v in data['references'].items()};assert [e for e in chunk if e['status']=='reference_fingerprints'][0]['sha256']==fps;assert [e for e in chunk if e['status']=='archive_written'][0]['sha256']==formal['array_sha256'][str(path)];warm=collections.defaultdict(list);n=0
 for e in chunk:
  if e['status']=='warmup_validated':c=e['component'];warm[c].append(e['index']);assert e['full_output_equal'] and e['output_sha256']==fps[c]
  elif e['status']=='measured':
   job=e['job'];c=job['component'];original=schedule['schedule']['jobs'][60*i+n];assert job==dict(original,order=n,global_order=original['order']) and warm[c]==[0,1,2] and e['full_output_equal'] and e['output_sha256']==fps[c] and all(type(e[k]) in (int,float) and math.isfinite(e[k]) and e[k]>=0 for k in ('gpu_event_ms','host_wall_ms'));n+=1
 assert n==60 and all(warm[c]==[0,1,2] for c in fps)
assert len(events)==32*76
r={'status':'passed','independent_review':True,'images':32,'jobs':1920,'warmups':288,'diagnostic_generations':32,'GT_opened':False,'predictions_opened':True,'scoring_performed':False,'formal_sha256':sha(D/'full_001_verification.json'),'formal_started_sha256':sha(D/'monitor_verify_full001/started.json'),'formal_terminal_sha256':sha(D/'monitor_verify_full001/terminal.json'),'run_started_sha256':sha(D/'monitor_full001/started.json'),'run_terminal_sha256':sha(D/'monitor_full001/terminal.json'),'reviewer_sha256':sha(__file__),'array_sha256':formal['array_sha256'],'limitations':['32-image expansion was selected after viewing single-image pilot','each output compared online, no per-measurement tensor archive','DS staging is not decoder DS addition','warm-cache component measurements, not TTFT or end-to-end','during measurement one Python stat syscall window 13974385382685808..13974385382692250 ns; process start and residual overhead unknown; all measurements retained'],'errors':[]};a.out.write_text(json.dumps(r,indent=2)+'\n');print('passed')
