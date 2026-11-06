"""Independent no-GT review of one-image E4 component pilot."""
import argparse,collections,hashlib,json,math,subprocess
from pathlib import Path
import torch
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_microbench_pilot';R=D/'pilot_001'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def meta(t):
 x=t.detach().contiguous().cpu();return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}
def tree(x):
 if torch.is_tensor(x):return dict(type='tensor',**meta(x))
 if isinstance(x,(list,tuple)):return {'type':'tuple' if isinstance(x,tuple) else 'list','items':[tree(v) for v in x]}
 if isinstance(x,dict):return {'type':'dict','items':{k:tree(x[k]) for k in sorted(x)}}
 if x is None or type(x) in (str,int,bool):return {'type':type(x).__name__,'value':x}
 if type(x) is float and math.isfinite(x):return {'type':'float','value':x}
 raise ValueError('bad tree')
def fp(x):return hashlib.sha256(json.dumps(tree(x),sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def terminal(folder):
 s=read(folder/'started.json');t=read(folder/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];return s,t
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
formal=read(D/'pilot_001_verification.json');assert formal['status']=='passed' and formal['jobs']==60 and formal['warmups']==9 and formal['GT_opened'] is False
vs,vt=terminal(D/'monitor_verify_pilot001');rs,rt=terminal(D/'monitor_pilot001')
for path,h in list(formal['files_sha256'].items())+list(formal['source_sha256'].items()):assert sha(path)==h,(path,'sha')
assert formal['verifier_sha256']==sha(D/'pilot_verify.py')
launch=read(R/'launch.json');ex=read(R/'exit.json');done=read(R/'worker_complete.json');assert launch['controller']['pid']==rs['worker_pid'] and ex['controller']==launch['controller'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and not ex['client_exited_while_container_running'] and ex['gpu_pids_after']==[] and done['status']=='completed'
ins=json.loads(subprocess.run(['docker','inspect',launch['name']],check=True,text=True,capture_output=True,timeout=15).stdout)[0];state=ins['State'];assert not state['Running'] and state['Pid']==0 and state['ExitCode']==0 and not state['OOMKilled'];assert ins['Id']==formal['inspection']['container_id'] and ins['Image']==formal['inspection']['image']
assert {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in ins['HostConfig']['Devices']}=={(x,x,'rwm') for x in formal['inspection']['devices']}
assert len([x for x in ins['Mounts'] if x['Type']=='bind'])==formal['inspection']['bind_mounts']
e1=read(B/'e1/e1full001_completion.json');j=Path(e1['files']['journal']['path']);assert sha(j)==e1['files']['journal']['sha256'];matches=[]
with j.open() as f:
 for line in f:
  r=json.loads(line)
  if r.get('task')=='DocVQA_VAL' and str(r.get('qid'))=='16539' and r.get('arm')=='P3-D1' and r.get('forced') is False:matches.append(r)
assert len(matches)==1;ref=matches[0];assert sha(j)==e1['files']['journal']['sha256'];data=torch.load(R/'reference.pt',map_location='cpu',weights_only=True);x=data['actual'];pr=x['processor'];assert {k:meta(pr[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}==ref['processor']
v=x['visual'];assert meta(v[0])==ref['visual_sha256'] and [meta(z) for z in v[1]]==ref['deepstack_full_sha256'] and len(v[1])==3
ordered=x['ordered'].reshape(-1).tolist();assert len(ordered)==len(set(ordered))==256 and ordered==ref['ordered_ids'] and sorted(ordered)==ref['spatial_ids']
pre=x['prefill'];assert meta(pre['inputs_embeds'])==ref['prefill_embed_sha256'] and meta(pre['attention_mask'])==ref['attention_mask_sha256'];assert pre['position_ids'].tolist()==ref['prefill_position_ids'] and x['rope_delta'].tolist()==ref['rope_delta'] and pre['visual_pos_masks'].tolist()==ref['injected_mask'] and [meta(z) for z in pre['deepstack_visual_embeds']]==ref['injected_deepstack']
events=[json.loads(z) for z in (R/'journal.jsonl').read_text().splitlines()];by=collections.Counter(z['status'] for z in events);assert by=={'capture_generated':1,'capture_cleanup':1,'archive_written':1,'capture_identity_passed':1,'reference_fingerprints':1,'warmup_validated':9,'measured':60}
assert [e['status'] for e in events[:5]]==['capture_generated','capture_cleanup','archive_written','capture_identity_passed','reference_fingerprints'];assert events[1]['passed'] and events[1]['capture_completed'] and events[2]['sha256']==sha(R/'reference.pt');assert events[0]['task']==ref['task'] and str(events[0]['qid'])==str(ref['qid']) and events[0]['group_id']==ref['group_id']
fps={k:fp(v) for k,v in data['references'].items()};assert events[4]['sha256']==fps
sched=done['schedule'];full=read(B/'e4_c_microbench/schedule001.json')['schedule'];assert sched=={'stage':'pilot','images':full['images'][:1],'components':full['components'],'jobs':full['jobs'][:60]}
warm=collections.defaultdict(list);measured=[]
for e in events[5:]:
 c=e.get('component') if e['status']=='warmup_validated' else e['job']['component'];assert e['full_output_equal'] is True and e['output_sha256']==fps[c]
 if e['status']=='warmup_validated':warm[c].append(e['index'])
 else:
  assert warm[c]==[0,1,2] and e['job']==sched['jobs'][len(measured)]
  assert all(type(e[k]) in (int,float) and math.isfinite(e[k]) and e[k]>=0 for k in ('gpu_event_ms','host_wall_ms'))
  if c=='ds_injection':assert 'excludes decoder DS addition' in e['scope']
  measured.append(e)
assert len(measured)==60 and all(warm[c]==[0,1,2] for c in fps) and collections.Counter(e['job']['component'] for e in measured)=={c:20 for c in fps}
result={'status':'passed','independent_review':True,'jobs':60,'warmups':9,'diagnostic_generations':1,'GT_opened':False,'predictions_opened':True,'scoring_performed':False,'formal_sha256':sha(D/'pilot_001_verification.json'),'formal_started_sha256':sha(D/'monitor_verify_pilot001/started.json'),'formal_terminal_sha256':sha(D/'monitor_verify_pilot001/terminal.json'),'reference_sha256':sha(R/'reference.pt'),'journal_sha256':sha(R/'journal.jsonl'),'run_started_sha256':sha(D/'monitor_pilot001/started.json'),'run_terminal_sha256':sha(D/'monitor_pilot001/terminal.json'),'reviewer_sha256':sha(__file__),'container_id':ins['Id'],'reference_fingerprints':fps,'limitations':['single image warm-cache component pilot','individual outputs verified online and only fingerprints retained; offline reconstruction impossible','ds_injection measures selection and mask staging, excluding decoder addition','not end-to-end TTFT or a statistical generalization'],'errors':[]}
a.out.write_text(json.dumps(result,indent=2)+'\n');print('passed')
