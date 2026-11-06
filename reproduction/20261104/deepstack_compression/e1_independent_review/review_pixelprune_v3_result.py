"""Independent reconstruction of archived PixelPrune one-image diagnostic metadata."""
import argparse,collections,hashlib,json,math,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixelprune_gpu_smoke_v3';O=D/'run_smoke001/observations'
def get(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def term(d):
 s=get(d/'started.json');t=get(d/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];return s
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
plan=get(D/'plan001.json');assert sha(D/'plan001.json')=='dd6b8100d0ebfe1e292553cd65faf8579a619dc4e95560cde8569e863aa03cf1'
for f,h in plan['sha256'].items():assert sha(f)==h,f
term(B/'e1_independent_review/monitor_pixelprune_v3_prelaunch');s=term(D/'monitor_smoke001');l=get(D/'run_smoke001/launch.json');ex=get(D/'run_smoke001/exit.json');assert l['plan_sha256']==sha(D/'plan001.json') and s['worker_pid']==l['controller']['pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['client_exited_while_container_running'] is False and ex['gpu_pids_after']==[]
ins=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];state=ins['State'];assert ins['Id']==ex['last_inspect']['id'] and ins['Image']==ex['container_identity']['image'] and not state['Running'] and state['Pid']==0 and state['ExitCode']==0 and not state['OOMKilled'];assert ins['Config']['Cmd']==[str(D/'worker.py'),'--out',str(O)] and ins['HostConfig']['NetworkMode']=='none'
devices=ins['HostConfig']['Devices'];assert {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in devices}=={(x,x,'rwm') for x in ['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']}
cmd=l['command'];mount=[]
for i,c in enumerate(cmd[:-1]):
 if c=='--mount':
  v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mount.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
assert sorted(mount)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in ins['Mounts'] if x['Type']=='bind')
names=plan['arms'];arms={n:get(O/(n+'.json')) for n in names};assert get(O/'worker_complete.json')['arm_statuses']=={n:'completed' for n in names};started=get(O/'started.json');assert started['task']==plan['anchor']['task'] and str(started['qid'])==plan['anchor']['qid'] and started['plan_sha256']==sha(D/'plan001.json')
progress=[json.loads(x) for x in (O/'progress.jsonl').read_text().splitlines()];assert progress[0]['event']=='worker_started'
for n in names:
 x=arms[n];ids=x['selected_indices'];count=x['selected_count'];assert len(ids)==count==len(set(ids)) and ids==sorted(ids) and all(type(i)==int and 0<=i<1024 for i in ids);assert count==1024 if n!='patched_enabled' else 1<=count<=1024
 assert x['status']=='completed' and x['author_dispatch_valid'] and x['hook_cleanup_valid'];g=get(O/(n+'_generated.json'));assert len(g['tokens'])==33 and all(z==g['forced_token_id'] for z in g['tokens']);assert get(O/(n+'_started.json'))['arm']==n
 assert all(x[k] is True for k in ['prefill_positions_match','prefill_main_match','prefill_text_match']);assert x['prefill_ds_matches']==[True]*3 and x['prefill_ds_count']==3 and x['prefill_ds_rows']==[count]*3 and x['prefill_visual_count']==count
 trace=x['trace'];post=x['post'];assert len(trace)==len(post)==33;seq=x['original_sequence_length'];initial=seq-1024+count;delta=x['original_rope_delta'];assert trace[0]['input_length']==initial and trace[0]['cache_position']==list(range(initial))
 for i,(t,z) in enumerate(zip(trace,post)):
  assert z['finite_logits'] and z['rope_delta']==delta==z['model_rope_delta'] and len(z['layers'])==36
  assert all(k['key']==[1,8,initial+i,128] and k['value']==[1,8,initial+i,128] for k in z['layers'])
  if i:
   assert t['input_length']==1 and t['cache_position']==[seq+i-1] and t['positions']==[[[x['original_max_position']+i]]]*3 and len(t['cache_before'])==36
   assert all(k['key']==[1,8,initial+i-1,128] and k['value']==[1,8,initial+i-1,128] for k in t['cache_before'])
 assert len(x['prefix'])==2 and len(x['prefix'][0])==36 and x['prefix'][0]==x['prefix'][1] and all(len(v)==64 for layer in x['prefix'][0] for v in layer.values())
 assert [e['call_index'] for e in progress if e.get('event')=='forward_complete' and e.get('arm')==n]==list(range(33));assert len([e for e in progress if e.get('event')=='arm_started' and e.get('arm')==n])==1 and len([e for e in progress if e.get('event')=='arm_recorded' and e.get('arm')==n])==1
base=arms['unpatched']['trace'][0]['positions'];disabled=arms['patched_disabled'];enabled=arms['patched_enabled'];assert disabled['trace'][0]['positions']==base and disabled['selected_indices']==list(range(1024)) and len(disabled['logit_checks'])==33 and all(z['comparable'] and z['allclose'] and z['argmax_equal'] and math.isfinite(z['max_abs_difference']) for z in disabled['logit_checks'])
# Recover the image block by searching positions, not relying on worker's prefill boolean.
seq=enabled['original_sequence_length'];picked=enabled['selected_indices'];pruned=enabled['trace'][0]['positions'];starts=[]
for start in range(seq-1024+1):
 indices=list(range(start))+[start+j for j in picked]+list(range(start+1024,seq))
 if all([[base[axis][0][j] for j in indices]]==pruned[axis] for axis in range(3)):starts.append(start)
assert len(starts)>=1
files=list(O.glob('*.json'))+[O/'progress.jsonl',D/'run_smoke001/launch.json',D/'run_smoke001/exit.json',D/'monitor_smoke001/started.json',D/'monitor_smoke001/terminal.json',D/'independent_prelaunch_review.json']
r={'status':'passed','independent_review':True,'GT_opened':False,'scoring_performed':False,'selected_count':{n:arms[n]['selected_count'] for n in names},'forced_calls_per_arm':33,'decode_calls_per_arm':32,'layers_per_call':36,'position_candidate_image_starts':starts,'disabled_unpatched_logits_runtime_allclose_argmax':True,'files_sha256':{str(f):sha(f) for f in files},'reviewer_sha256':sha(__file__),'limitations':['one image forced artificial prefix only','author patch applied after loading shared model','per-step logits, original main/DS tensors and KV caches not archived; runtime equality and hashes cannot be independently rebuilt','enabled budget is data dependent, no quality/latency or general correctness claims'],'errors':[]}
a.out.write_text(json.dumps(r,indent=2)+'\n');print('passed')
