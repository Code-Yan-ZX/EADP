"""Independent 222-image/888 fixed32 review; never opens GT or scores."""
import argparse,hashlib,importlib.util,json,os,re,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;R=B/'e4_c_multibudget_references192';A=B/'e4_c_multibudget_references192_accept';OUT=I/'e4c_multibudget_references192_result_review.json'
FIELDS=('task','qid','group_id','image_sha256','message_sha256','budget','ds_on','selection_mode','greedy_calls','processor','main_visual','pruned_visual','image_start','original_length','full_deepstack','ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','injected_mask','injected_deepstack','block_calls','main_merger_calls','ds_calls')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def done(d,script):
 s=read(d/'started.json');t=read(d/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervisor/command not clean')
 return s
def validate_arm(job,rec,panel,raw,tokenizer):
 if any(str(rec.get(k))!=str(job[k]) for k in ('task','qid','group_id','budget','protocol')) or any(raw.get(k)!=rec.get(k) for k in ('task','qid','group_id','budget','ds_on','forced','protocol','output_ids','eos_ids','generation_scalars','min_new_tokens')):raise ValueError('arm/raw identity')
 if panel.get('actual_injection_verified') is not True or any(name not in rec or rec[name]!=panel.get(name) for name in FIELDS) or rec['decode_trace'][0]!=panel['decode_trace'][0]:raise ValueError('independent E4 prefill identity')
 if rec.get('forced') is not False or rec.get('min_new_tokens')!=32 or rec.get('max_new_tokens')!=32 or rec.get('generation_scalars')!=dict(panel['generation_scalars'],max_new_tokens=32) or rec.get('eos_ids')!=panel['eos_ids']:raise ValueError('fixed32 recipe')
 ids=rec['output_ids'];eos=rec['eos_ids'];trace=rec['decode_trace'];seq=rec['original_length']-1024+rec['budget'];delta=rec['adjusted_delta'][0][0]
 if len(ids)!=32 or any(type(x) is not int for x in ids) or not eos or any(x in eos for x in ids[:-1]) or len(trace)!=32 or tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=rec.get('answer'):raise ValueError('output endpoint/decode')
 for i,step in enumerate(trace):
  before=0 if i==0 else seq+i-1;after=seq+i
  if step.get('position_ids')!=(rec['prefill_position_ids'] if i==0 else [[[seq+i-1+delta]]]*3) or step.get('cache_position')!=(list(range(seq)) if i==0 else [seq+i-1]) or step.get('rope_delta')!=[[delta]] or (step.get('layer_cache_lengths') not in ([],[0]*36) if i==0 else step.get('layer_cache_lengths')!=[before]*36) or step.get('layer_cache_lengths_after')!=[after]*36:raise ValueError('32-step position/cache lengths')
 fps=rec.get('cache_fingerprints')
 if not isinstance(fps,list) or len(fps)!=1 or len(fps[0])!=36:raise ValueError('prefill KV fingerprint count')
 for layer in fps[0]:
  if set(layer)!={'key','value'}:raise ValueError('bad KV fingerprint pair')
  for v in layer.values():
   if set(v)!={'dtype','shape','sha256'} or v['dtype']!='torch.bfloat16' or v['shape']!=[1,8,seq,128] or not re.fullmatch('[0-9a-f]{64}',v['sha256']):raise ValueError('KV fingerprint schema')
 if (rec.get('arrays') or {}).get('kind')!='summary_only':raise ValueError('high dimensional archive claim')
def review():
 # Do not read target records until BOTH the worker and formal verifier
 # have actually terminated successfully and formal metadata is bound.
 run=R/'run_001';mon=R/'monitor_run001';rs=done(mon,R/'launch.py');done(A/'monitor_verify001',A/'verify.py');done(A/'monitor_snapshot001',A/'snapshot.py')
 formal=read(A/'verification001.json');vp=read(A/'verification_plan001.json')
 if formal.get('status')!='passed' or (formal.get('inputs'),formal.get('generations'))!=(222,888) or formal.get('GT_opened') is not False or formal.get('scoring_performed') is not False or formal.get('timing_claims') is not False or formal.get('errors')!=[] or formal.get('verifier_sha256')!=sha(A/'verify.py') or formal.get('verification_plan_sha256')!=sha(A/'verification_plan001.json') or any(sha(p)!=h for p,h in vp['source_sha256'].items()):raise ValueError('formal proof/source mismatch')
 ex=read(run/'exit.json');launch=read(run/'launch.json');st=ex.get('last_inspect') or {};cid=(ex.get('container_identity') or {}).get('container_id')
 if rs.get('worker_pid')!=(launch.get('controller') or {}).get('pid') or ex.get('controller')!=launch.get('controller') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('container_stopped_verified') is not True or ex.get('client_exited_while_container_running') is not False or ex.get('gpu_pids_after')!=[] or st.get('id')!=cid or st.get('running') is not False or st.get('pid')!=0:raise ValueError('run not cleanly stopped')
 doc=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=10).stdout)[0]
 if doc['Id']!=cid or doc['State']['Running'] is not False or doc['State']['Pid']!=0 or doc['State']['ExitCode']!=0 or doc['Image']!=st['image'] or doc['HostConfig']['NetworkMode']!='none' or doc['Config']['Cmd']!=[str(R/'worker.py'),'--out',str(run)]:raise ValueError('actual Docker identity')
 snap=R/'run_001_sources/manifest.json';sm=read(snap);plan=read(R/'execution_plan001.json')
 if sm.get('status')!='passed' or sm.get('execution_plan_sha256')!=sha(R/'execution_plan001.json') or set(sm['sources'])!=set(plan['source_sha256']):raise ValueError('snapshot identity')
 for path,digest in plan['source_sha256'].items():
  x=sm['sources'][path];copy=Path(x['snapshot']).resolve()
  if x.get('source')!=path or x.get('sha256')!=digest or not copy.is_relative_to(snap.parent.resolve()) or sha(copy)!=digest:raise ValueError('snapshot source mismatch')
 diag=run/'diagnostic';jp=diag/'journal.jsonl';rp=diag/'records.jsonl'
 for key,path in (('run_started_sha256',mon/'started.json'),('run_terminal_sha256',mon/'terminal.json'),('snapshot_manifest_sha256',snap),('journal_sha256',jp),('records_sha256',rp)):
  if formal.get(key)!=sha(path):raise ValueError('formal target binding '+key)
 # The old E1/E4 mixed prediction records are parsed solely as input identity.
 modpath=R/'references.py'
 if sha(modpath)!=plan['source_sha256'][str(modpath.resolve())]:raise ValueError('reference source changed')
 spec=importlib.util.spec_from_file_location('_independent_888_refs',modpath);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);refs=m.load_references()
 rows=sorted(refs['rows'],key=lambda x:(x['task'],str(x['qid'])));jobs=[(r,k) for r in rows for k in (128,256,512,1024)]
 if len(rows)!=222 or len({(x['task'],x['group_id']) for x in rows})!=222 or len(jobs)!=888:raise ValueError('cohort denominator')
 events=[json.loads(x) for x in jp.read_text().splitlines() if x.strip()];records=[json.loads(x) for x in rp.read_text().splitlines() if x.strip()]
 if len(events)!=3552 or len(records)!=888:raise ValueError('complete journal denominator')
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
 for i,(row,k) in enumerate(jobs):
  job={'order':i,'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':k,'protocol':'fixed32'};segment=events[4*i:4*i+4]
  if [x.get('event') for x in segment]!=['arm_started','raw_generation','arm_captured','arm_completed'] or any(x.get('job')!=job for x in segment) or records[i]!=segment[2]['record']:raise ValueError('journal order/records mismatch')
  validate_arm(job,records[i],refs['records'][(row['task'],str(row['qid']),k)],segment[1]['raw'],tokenizer)
 return {'status':'passed','independent_review':True,'inputs':222,'generations':888,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'predictions_opened':True,'formal_sha256':sha(A/'verification001.json'),'formal_started_sha256':sha(A/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(A/'monitor_verify001/terminal.json'),'run_started_sha256':sha(mon/'started.json'),'run_terminal_sha256':sha(mon/'terminal.json'),'snapshot_manifest_sha256':sha(snap),'records_sha256':sha(rp),'journal_sha256':sha(jp),'reviewer_sha256':sha(__file__),'scope':'independent record/decode and prefill fingerprint review; full KV/DS tensors not archived; existing DEV predictions parsed only for identity','errors':[]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.resolve()!=OUT:raise ValueError('unexpected proof path')
 result=review()
 with OUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
