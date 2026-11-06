"""Independent no-GT review of full random-K256 panel after formal completion."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;I=B/'e1_independent_review';D=B/'e6/random_full_exec_v2';S=B/'e6/random_smoke';O=D/'run_001';SEEDS=(20261104,20261105,20261106)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def term(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor incomplete')
 return s

def choose(image,seed):
 d=int.from_bytes(hashlib.sha256(('eadp-e6-random-v1|'+str(seed)+'|'+image).encode()).digest()[:8],'big');counter=0;pending=[];indices=list(range(1024))
 for i in range(1023,0,-1):
  bound=i+1;limit=2**64-(2**64%bound)
  while True:
   if not pending:
    block=hashlib.sha256(d.to_bytes(8,'big')+counter.to_bytes(8,'big')).digest();counter+=1;pending=[int.from_bytes(block[z:z+8],'big') for z in range(0,32,8)]
   n=pending.pop(0)
   if n<limit:break
  j=n%bound;indices[i],indices[j]=indices[j],indices[i]
 return d,sorted(indices[:256])
def check(r,choice,e1,decode):
 for k in ('task','qid','group_id','image_sha256','message_sha256'):
  if str(r.get(k))!=str(choice.get(k)):raise ValueError('input '+k)
 for k in ('task','qid','group_id','message_sha256'):
  if str(r.get(k))!=str(e1.get(k)):raise ValueError('reference '+k)
 derived,ids=choose(choice['image_sha256'],choice['seed'])
 if choice.get('derived_seed64')!=derived or choice.get('spatial_ids')!=ids or r.get('derived_seed64')!=derived or r.get('seed')!=choice['seed'] or r.get('ordered_ids')!=ids or r.get('spatial_ids')!=ids:raise ValueError('random draw')
 if r.get('selection_mode')!='e6_random_fixed_seed' or r.get('budget')!=256 or r.get('ds_on') is not True or r.get('greedy_calls')!=1 or r.get('extra_author_greedy') is not True or r.get('actual_injection_verified') is not True or r.get('timing_eligible') is not False or r.get('forced') is not False or r.get('protocol')!='natural':raise ValueError('random path')
 for a,b in (('processor','processor'),('main_visual','visual_sha256'),('full_deepstack','deepstack_full_sha256'),('full_position_ids','original_position_ids'),('full_delta','full_rope_delta')):
  if r.get(a)!=e1.get(b):raise ValueError('E1 main/DS '+a)
 start=r['image_start'];n=r['original_length'];pos=r['full_position_ids'];ret=list(range(start))+[start+x for x in ids]+list(range(start+1024,n))
 if r.get('retained_ids')!=ret or len(pos)!=3 or any(len(x)!=1 or len(x[0])!=n for x in pos) or r.get('prefill_position_ids')!=[[[x[0][i] for i in ret]] for x in pos]:raise ValueError('P3 positions')
 delta=max(max(x[0]) for x in pos)+1-n;mask=[False]*len(ret);mask[start:start+256]=[True]*256
 if r.get('full_delta')!=[[delta]] or r.get('adjusted_delta')!=[[delta+768]] or r.get('injected_mask')!=[mask] or len(r.get('full_deepstack',[]))!=3 or len(r.get('selected_deepstack',[]))!=3 or r.get('selected_deepstack')!=r.get('injected_deepstack') or r.get('arrays',{}).get('kind')!='summary_only':raise ValueError('DS/retention')
 out=r.get('output_ids');eos=r.get('eos_ids');trace=r.get('decode_trace')
 if not isinstance(out,list) or not out or len(out)>2048 or any(type(x) is not int for x in out) or eos!=e1['eos_ids'] or r.get('generation_scalars')!=e1['generation_scalars'] or r.get('answer')!=decode(out):raise ValueError('generation identity')
 ended=out[-1] in eos
 if r.get('eos') is not ended or r.get('truncated') is not (not ended and len(out)==2048) or not (ended or len(out)==2048) or any(t in eos for t in out[:-1]) or not isinstance(trace,list) or len(trace)!=len(out):raise ValueError('termination')
 for i,t in enumerate(trace):
  prev=0 if i==0 else len(ret)+i-1;after=len(ret)+i
  if t.get('position_ids')!=(r['prefill_position_ids'] if i==0 else [[[prev+delta+768]]]*3) or t.get('cache_position')!=(list(range(len(ret))) if i==0 else [prev]) or t.get('rope_delta')!=[[delta+768]] or (t.get('layer_cache_lengths') not in ([],[0]*36) if i==0 else t.get('layer_cache_lengths')!=[prev]*36) or t.get('layer_cache_lengths_after')!=[after]*36:raise ValueError('decode position/cache')

def main(out):
 # No full-panel result is accessed before both real supervisors completed.
 rs=term(D/'monitor_run001',D/'launch.py');term(D/'monitor_snapshot001',D/'snapshot.py');term(D/'monitor_verify001',D/'verify.py',D/'verification001.json')
 f=read(D/'verification001.json');plan=read(D/'runtime_plan001.json');vp=read(D/'verification_plan001.json')
 if f.get('status')!='passed' or (f.get('questions'),f.get('generations'))!=(384,1152) or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('errors')!=[] or f.get('verifier_sha256')!=sha(D/'verify.py') or f.get('runtime_plan_sha256')!=sha(D/'runtime_plan001.json') or vp.get('runtime_plan_sha256')!=sha(D/'runtime_plan001.json') or vp.get('verifier_sha256')!=sha(D/'verify.py') or vp.get('contract_sha256')!=sha(D/'verify_contract.py') or vp.get('snapshot_sha256')!=sha(D/'snapshot.py'):raise ValueError('formal proof or frozen verifier changed')
 if f.get('run_started_sha256')!=sha(D/'monitor_run001/started.json') or f.get('run_terminal_sha256')!=sha(D/'monitor_run001/terminal.json'):raise ValueError('run proof changed')
 snap=read(D/'run_001_sources/manifest.json')
 if snap.get('status')!='passed' or snap.get('runtime_plan_sha256')!=sha(D/'runtime_plan001.json') or set(snap.get('sources',{}))!=set(plan['source_sha256']):raise ValueError('snapshot inventory')
 for name,item in snap['sources'].items():
  cp=Path(item['snapshot']).resolve()
  if item.get('sha256')!=plan['source_sha256'][name] or not cp.is_relative_to((D/'run_001_sources').resolve()) or sha(cp)!=item['sha256']:raise ValueError('snapshot bytes')
 launch=read(O/'launch.json');ex=read(O/'exit.json');cid=(ex.get('container_identity') or {}).get('container_id')
 if not cid or (launch.get('controller') or {}).get('pid')!=rs['worker_pid'] or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('client_exited_while_container_running') is not False or ex.get('container_stopped_verified') is not True or ex.get('gpu_pids_after')!=[]:raise ValueError('controller exit')
 doc=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=doc['State']
 if doc['Id']!=cid or st['Running'] is not False or st['Pid']!=0 or st['ExitCode']!=0 or st['OOMKilled'] is not False or doc['HostConfig']['NetworkMode']!='none' or doc['Config']['Cmd']!=[str(D/'worker.py'),'--out',str(O)]:raise ValueError('actual Docker exit')
 for name,digest in f['files_sha256'].items():
  if sha(name)!=digest:raise ValueError('accepted file changed')
 done=read(O/'worker_complete.json');prov=read(O/'completion_provenance.json')
 if done.get('status')!='completed' or (done.get('questions'),done.get('natural_generations'))!=(384,1152) or done.get('GT_opened') is not False or done.get('scoring_performed') is not False or prov.get('status')!='passed' or prov.get('weight_stats_before')!=prov.get('weight_stats_after') or len(prov.get('weight_stats_before',{}))!=4 or prov.get('source_sha256')!=plan['source_sha256']:raise ValueError('completion provenance')
 # Accepted E1 mixed predictions are read only to validate input and method.
 ep=read(B/'e1/e1full001_completion.json');ej=B/'e1/e1full001/raw_predictions.jsonl';choicesfile=S/'selection_manifest001.jsonl';sp=read(S/'plan001.json')
 if ep.get('status')!='passed' or ep.get('GT_opened') is not False or sha(ej)!=ep['files']['journal']['sha256'] or sha(choicesfile)!=sp.get('selection_manifest_sha256'):raise ValueError('E1 or frozen selection changed')
 choices={}
 for x in lines(choicesfile):
  k=(x['task'],str(x['qid']),x['seed'])
  if k in choices or x['seed'] not in SEEDS:raise ValueError('random manifest uniqueness')
  choices[k]=x
 if len(choices)!=1152 or len({(t,q) for t,q,_ in choices})!=384:raise ValueError('selection denominator')
 refs={};keys={(t,q) for t,q,_ in choices}
 for x in lines(ej):
  k=(x.get('task'),str(x.get('qid')))
  if k in keys and x.get('arm')=='P3-D1' and x.get('forced') is False:
   if k in refs:raise ValueError('duplicate E1 reference')
   refs[k]=x
 if len(refs)!=384:raise ValueError('missing E1 reference')
 records=lines(O/'records.jsonl');generated=lines(O/'generated.jsonl');journal=lines(O/'journal.jsonl')
 if (len(records),len(generated),len(journal))!=(1152,1152,2304):raise ValueError('full output denominator')
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
 decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
 seen=set()
 for i,(r,g) in enumerate(zip(records,generated)):
  key=(r['task'],str(r['qid']),r['seed'])
  if key in seen or key not in choices:raise ValueError('duplicate/unexpected arm')
  seen.add(key);expected={'task':key[0],'qid':key[1],'group_id':r['group_id'],'seed':key[2],'forced':False}
  if g.get('scheduled')!=expected or any(g['raw'].get(k)!=r.get(k) for k in ('task','qid','group_id','seed','forced','output_ids','eos_ids','generation_scalars')) or journal[2*i]!={'event':'started',**expected} or journal[2*i+1]!={'event':'recorded',**expected}:raise ValueError('journal/raw mismatch')
  check(r,choices[key],refs[key[:2]],decode)
 if seen!=set(choices):raise ValueError('missing seed arms')
 proof={'status':'passed','independent_review':True,'questions':384,'generations':1152,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'errors':[],'formal_sha256':sha(D/'verification001.json'),'formal_started_sha256':sha(D/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(D/'monitor_verify001/terminal.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'reviewer_sha256':sha(__file__),'files_sha256':f['files_sha256'],'limitations':['Posthoc DEV study after observing earlier DEV results.','Full panel stores runtime summaries rather than reconstructable tensor arrays; actual KV tensor bytes were not archived.','Extra author greedy pass invalidates performance comparisons.','Weight stat equality cannot exclude transient replacement.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.resolve()!=I/'random_full_v2_result_review.json' or a.out.exists():raise ValueError('fixed output required')
 main(a.out)
