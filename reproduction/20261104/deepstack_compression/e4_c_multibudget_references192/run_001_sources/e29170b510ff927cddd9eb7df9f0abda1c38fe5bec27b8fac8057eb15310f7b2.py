"""Independent no-GT reconstruction of 16 four-budget model identity pairs."""
import argparse,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_budget_execution';A=B/'e4_c_budget_accept';I=B/'e1_independent_review';RUN=D/'run_001';DIAG=RUN/'diagnostic'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def supervised(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=script:raise ValueError('supervisor incomplete '+str(mon))
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out):raise ValueError('supervisor output mismatch')
 return s
def module(path):
 spec=importlib.util.spec_from_file_location('_budget_independent_reference',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def metadata_pair(fresh,light,job,token):
 fields=('task','qid','group_id','image_sha256','message_sha256','budget','ds_on','selection_mode','greedy_calls','processor','main_visual','pruned_visual','image_start','original_length','full_deepstack','ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','injected_mask','injected_deepstack','prefill_logits','output_ids','answer','eos_ids','eos','truncated','max_new_tokens','generation_scalars','forced','forced_token','decode_trace','cache_fingerprints','block_calls','main_merger_calls','ds_calls')
 if any(k not in fresh or k not in light or fresh[k]!=light[k] for k in fields):raise ValueError('fresh/light metadata mismatch')
 r=fresh;k=job['budget'];n=r['original_length'];start=r['image_start'];ordered=r['ordered_ids'];spatial=r['spatial_ids']
 if (r['task'],str(r['qid']),r['budget'],r['forced'])!=(job['task'],job['qid'],k,job['forced']) or r['actual_injection_verified'] is not True or r['ds_on'] is not True:raise ValueError('job/DS mismatch')
 if len(ordered)!=k or len(set(ordered))!=k or any(type(i) is not int or not 0<=i<1024 for i in ordered) or spatial!=sorted(ordered):raise ValueError('selection')
 if r['selection_mode']!=('keep_all' if k==1024 else 'author_greedy') or r['greedy_calls']!=(0 if k==1024 else 1) or (k==1024 and ordered!=list(range(1024))):raise ValueError('selection implementation')
 retained=list(range(start))+[start+i for i in spatial]+list(range(start+1024,n));full=r['full_position_ids'];pos=r['prefill_position_ids'];delta=max(max(axis[0]) for axis in full)+1-n;adjusted=delta+1024-k;seq=len(retained)
 if r['retained_ids']!=retained or len(full)!=3 or any(len(axis)!=1 or len(axis[0])!=n for axis in full) or pos!=[[[axis[0][i] for i in retained]] for axis in full] or r['full_delta']!=[[delta]] or r['adjusted_delta']!=[[adjusted]]:raise ValueError('P3/delta')
 expected=[False]*seq;expected[start:start+k]=[True]*k
 if r['injected_mask']!=[expected] or len(r['injected_deepstack'])!=3:raise ValueError('DS mask')
 ids=r['output_ids'];trace=r['decode_trace'];eos=r['eos_ids']
 if not ids or len(trace)!=len(ids):raise ValueError('decode length')
 if job['forced']:
  if ids!=[token]*33 or eos!=[] or r['forced_token']!=token or r['max_new_tokens']!=33:raise ValueError('forced sequence')
 elif r['forced_token'] is not None or not eos or r['max_new_tokens']!=2048 or not (ids[-1] in eos or len(ids)==2048):raise ValueError('natural termination')
 for i,x in enumerate(trace):
  pre=[] if i==0 else [seq+i-1]*36
  if i==0 and x['layer_cache_lengths'] not in ([],[0]*36):raise ValueError('initial cache')
  if i and x['layer_cache_lengths']!=pre:raise ValueError('prior cache')
  if x['layer_cache_lengths_after']!=[seq+i]*36 or x['rope_delta']!=[[adjusted]] or x['position_ids']!=(pos if i==0 else [[[seq+i-1+adjusted]]]*3) or x['cache_position']!=(list(range(seq)) if i==0 else [seq+i-1]):raise ValueError('decode position/cache trajectory')
 fps=r['cache_fingerprints']
 if len(fps)!=(33 if job['forced'] else 1):raise ValueError('KV fingerprint length')
 for i,layers in enumerate(fps):
  if len(layers)!=36 or any(set(layer)!={'key','value'} or any(len(v.get('shape',[]))<2 or v['shape'][-2]!=seq+i or len(v.get('sha256',''))!=64 for v in layer.values()) for layer in layers):raise ValueError('KV fingerprints')
def arrays_pair(fresh,light,job):
 paths=[Path(x['arrays']['path']).resolve() for x in (fresh,light)]
 for path,r in zip(paths,(fresh,light)):
  if not path.is_relative_to((DIAG/'arrays').resolve()) or r['arrays']['kind']!='full_arrays' or sha(path)!=r['arrays']['sha256']:raise ValueError('archive SHA/path')
 with np.load(paths[0],allow_pickle=False) as a,np.load(paths[1],allow_pickle=False) as b:
  required={'visual','prefill','prefill_logits',*[f'ds{i}' for i in range(3)],*[f'injected{i}' for i in range(3)]}
  if not required<=set(a.files)&set(b.files) or 'pruned' not in b.files:raise ValueError('archive fields')
  if any(not np.array_equal(a[x],b[x]) for x in set(a.files)&set(b.files)):raise ValueError('archive pair mismatch')
  spatial=fresh['spatial_ids'];start=fresh['image_start'];k=job['budget']
  if not np.array_equal(b['pruned'],a['visual'][spatial]) or not np.array_equal(a['prefill'][0,start:start+k],b['pruned']):raise ValueError('actual pruned/prefill arrays')
  if any(not np.array_equal(a['injected'+str(i)],a['ds'+str(i)][spatial]) for i in range(3)):raise ValueError('actual injected DS')
 return {str(p):sha(p) for p in paths}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=I/'e4c_budget_identity001_review.json'
 if args.out.resolve()!=out:raise ValueError('output path')
 formal=A/'verification002.json';supervised(A/'monitor_verify002',A/'verify.py',formal);f=read(formal)
 if f.get('status')!='passed' or f.get('verifier_sha256')!=sha(A/'verify.py') or f.get('errors')!=[] or f.get('GT_opened') is not False or f.get('scoring_performed') is not False:raise ValueError('formal proof')
 supervised(A/'monitor_snapshot001',A/'snapshot.py');run=supervised(D/'monitor_run001',D/'launch.py')
 if run['command'][-2:]!=['--run-id','001']:raise ValueError('run id')
 plan=read(D/'execution_plan001.json');snapshot=D/'run_001_sources';manifest=snapshot/'manifest.json';m=read(manifest)
 if sha(manifest)!=f['snapshot_manifest_sha256'] or m.get('status')!='passed' or set(m['sources'])!=set(plan['source_sha256']):raise ValueError('source snapshot inventory')
 for name,h in plan['source_sha256'].items():
  item=m['sources'][name];p=Path(item['snapshot']).resolve()
  if item['source']!=name or item['sha256']!=h or not p.is_relative_to(snapshot.resolve()) or sha(p)!=h or sha(name)!=h:raise ValueError('source snapshot differs')
 if len(m['sources'])!=142 or sha(D/'execution_plan001.json')!=f['plan_sha256'] or sha(D/'monitor_run001/started.json')!=f['run_started_sha256'] or sha(D/'monitor_run001/terminal.json')!=f['run_terminal_sha256']:raise ValueError('run provenance')
 launch=read(RUN/'launch.json');ex=read(RUN/'exit.json');completed=read(RUN/'worker_complete.json');pre=read(I/'e4c_budget_execution_prelaunch_review.json');supervised(I/'monitor_e4c_budget_execution_prelaunch',I/'review_e4c_budget_execution_prelaunch.py',I/'e4c_budget_execution_prelaunch_review.json')
 if launch['approval_bundle']['gpu_review_sha256']!=sha(I/'e4c_budget_execution_prelaunch_review.json') or pre.get('reviewer_sha256')!=sha(I/'review_e4c_budget_execution_prelaunch.py') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('gpu_pids_after')!=[] or completed.get('status')!='passed' or completed.get('generations')!=32:raise ValueError('launch/exit proof')
 ident=ex['container_identity'];obj=json.loads(subprocess.check_output(['docker','inspect',ident['container_id']],text=True))[0];state=obj['State'];argv=launch['command'];idx=argv.index(ident['image'])
 if ident['container_id'] not in ident['host_cgroup'] or obj['Id']!=ident['container_id'] or obj['Image']!=ident['image'] or state['Status']!='exited' or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] or obj['Config']['Entrypoint']!=[argv[idx-1]] or obj['Config']['Cmd']!=argv[idx+1:]:raise ValueError('actual GPU container identity')
 refs=module(B/'e4_c_budget_pilot_v2/reference.py').load_references();rows=refs['rows'];token=completed['forced_token'];jobs=[]
 for row in rows:
  for k in (128,256,512,1024):
   for force in ([False,True] if row is rows[0] else [False]):jobs.append({'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':k,'forced':force})
 events=[json.loads(x) for x in (DIAG/'journal.jsonl').read_text().splitlines() if x.strip()];hashes={}
 if len(jobs)!=16 or len(events)!=112 or sha(DIAG/'journal.jsonl')!=f['journal_sha256']:raise ValueError('journal denominator')
 from transformers import AutoTokenizer
 model='/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b';tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
 if not tok.encode(' the',add_special_tokens=False) or tok.encode(' the',add_special_tokens=False)[0]!=token:raise ValueError('forced token identity')
 for i,job in enumerate(jobs):
  seg=events[7*i:7*i+7]
  if [x.get('event') for x in seg]!=['arm_started','raw_generation','arm_completed','arm_started','raw_generation','arm_completed','pair_verified'] or any(x.get('job')!=job for x in seg) or [x.get('side') for x in seg[:6]]!=['fresh']*3+['light']*3:raise ValueError('journal exposure/order')
  fresh,light=seg[2]['record'],seg[5]['record']
  for raw,r in ((seg[1]['raw'],fresh),(seg[4]['raw'],light)):
   if any(raw.get(k)!=r.get(k) for k in ('task','qid','group_id','budget','ds_on','forced','output_ids','eos_ids','generation_scalars')) or tok.decode(r['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=r['answer']:raise ValueError('raw token identity')
  if seg[6].get('comparison',{}).get('status')!='passed':raise ValueError('pair status')
  metadata_pair(fresh,light,job,token);hashes.update(arrays_pair(fresh,light,job))
  if not job['forced']:
   ref=refs['records'][(job['task'],job['qid'],job['budget'])]
   for key in ('processor','ordered_ids','spatial_ids','full_position_ids','prefill_position_ids','adjusted_delta','injected_mask','injected_deepstack','output_ids','answer','eos_ids','generation_scalars'):
    if fresh[key]!=ref[key]:raise ValueError('accepted E4 reference mismatch '+key)
 if len(hashes)!=32 or hashes!=f['array_sha256']:raise ValueError('archive inventory')
 result={'status':'passed','independent_review':True,'pairs':16,'generations':32,'natural_pairs':12,'forced_pairs':4,'forced_decode_steps':32,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'formal_sha256':sha(formal),'formal_started_sha256':sha(A/'monitor_verify002/started.json'),'formal_terminal_sha256':sha(A/'monitor_verify002/terminal.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'snapshot_manifest_sha256':sha(manifest),'reviewer_sha256':sha(__file__),'array_sha256':hashes,'errors':[],'limitations':['only three DEV anchors; identity diagnostic and not latency/quality inference','forced 33 generated tokens represent 32 subsequent decoding calls','formal verifier002 was corrected after verifier001 had read the first partial pair; failed proof and logs retained','source snapshots were produced after run; pre/post source hashing does not rule out a transient change during execution']}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('passed')
if __name__=='__main__':main()
