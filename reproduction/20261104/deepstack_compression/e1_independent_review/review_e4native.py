"""Independent stock-native/K1024 review after experiment acceptance; no GT."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');E=B/'e4_native';R=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(4*1024*1024),b''):h.update(z)
 return h.hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def lines(p):
 b=Path(p).read_bytes()
 if b and not b.endswith(b'\n'):raise ValueError('partial journal')
 return [json.loads(x) for x in b.splitlines()]
def main():
 start=time.time();errors=[];checks=0;verified=0
 def req(v,msg):
  nonlocal checks
  checks+=1
  if not bool(v):errors.append(msg)
 try:
  import torch
  from transformers import AutoTokenizer
  run=E/'native_001';proof=rd(E/'native_001_verification.json');f=rd(run/'frozen.json');done=rd(run/'worker_complete.json');launch=rd(run/'launch.json');ex=rd(run/'exit.json');term=rd(E/'monitor_native001/terminal.json');vterm=rd(E/'monitor_verify_native001/terminal.json')
  req(proof.get('status')=='passed' and proof.get('tasks')==3 and proof.get('natural_pairs')==3 and proof.get('forced_pairs')==3 and proof.get('forced_decode_steps')==32 and proof.get('kv_fingerprints_compared') is True,'formal native proof')
  req(vterm.get('reason')=='completed' and vterm.get('worker_returncode')==0 and vterm.get('remaining_process_group')==[],'formal verifier termination')
  for path,h in proof['files_sha256'].items():req(sha(path)==h,'evidence hash '+path)
  req(proof['verifier_sha256']==sha(E/'native_verify.py'),'native verifier source')
  req(term.get('reason')=='completed' and term.get('worker_returncode')==0 and term.get('remaining_process_group')==[] and term.get('worker_pid')==launch['controller']['pid'],'actual launcher terminal')
  req(ex.get('returncode')==0 and ex.get('container_exit_code')==0 and ex.get('timed_out') is False and ex.get('interrupted') is None and ex.get('container_stopped_verified') is True and ex.get('client_exited_while_container_running') is False and ex.get('gpu_pids_after')==[] and ex['controller']==launch['controller'],'actual container exit')
  for ident in (ex['controller'],ex['container_identity']['host_init']):
   p=Path('/proc')/str(ident['pid'])/'stat';alive=False
   if p.exists() and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==ident['boot_id']:alive=int(p.read_text().rsplit(')',1)[1].split()[19])==ident['start_ticks']
   req(not alive,'original processes exited')
  ins=subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,timeout=10)
  req(ins.returncode==0 and not json.loads(ins.stdout)[0]['State']['Running'] and json.loads(ins.stdout)[0]['State']['ExitCode']==0,'independent docker inspect')
  pre=rd(E/'independent_review_002.json');cpu=rd(E/'cpu_check_002/verification.json');req(pre['status']=='passed' and pre['cpu_proof_sha256']==sha(E/'cpu_check_002/verification.json') and cpu['status']=='passed','prelaunch review binding')
  for path,h in pre['source_sha256'].items():req(sha(path)==h,'reviewed code '+path)
  for path,h in f['code_sha256'].items():req(sha(path)==h,'frozen worker code '+path)
  req(f['stage']=='stock_native_k1024' and len(f['anchors'])==3 and f['gt_read'] is False and f['scoring'] is False and done['n_questions']==3 and done['n_generations']==12 and done['n_pairs']==6,'frozen inventory')
  rows=lines(run/'records.jsonl');events=lines(run/'generated.jsonl');markers=lines(run/'comparisons.jsonl');req(len(rows)==len(events)==12 and len(markers)==6,'record inventories')
  plan=rd(B.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');tokenizer=AutoTokenizer.from_pretrained(plan['recipe']['model_path'],local_files_only=True,trust_remote_code=False);forced_id=tokenizer.encode(' the',add_special_tokens=False)[0]
  mapping={'task':'task','qid':'qid','group_id':'group_id','processor':'processor','main_visual':'main_visual','full_deepstack':'full_deepstack','prefill_embeds':'prefill_embeds','prefill_mask':'prefill_mask','prefill_position_ids':'prefill_position_ids','rope_delta':'adjusted_delta','injected_mask':'injected_mask','injected_deepstack':'injected_deepstack','prefill_logits':'prefill_logits','output_ids':'output_ids','answer':'answer','eos_ids':'eos_ids','eos':'eos','truncated':'truncated','generation_scalars':'generation_scalars','decode_trace':'decode_trace','cache_fingerprints':'cache_fingerprints'}
  for ix in range(6):
   before=len(errors);native,budget=rows[2*ix:2*ix+2];anchor=f['anchors'][ix//2];forced=bool(ix%2);tag=f"{anchor['task']}:{anchor['qid']}:{forced}"
   req(native['path']=='stock_native' and budget['budget']==1024 and budget['ds_on'] is True and native['forced'] is forced and budget['forced'] is forced,tag+' paths')
   for key in ('task','qid','group_id','message_sha256','image_sha256'):req(str(native[key])==str(anchor[key]) and str(budget[key])==str(anchor[key]),tag+' anchor '+key)
   for left,right in mapping.items():req(native[left]==budget[right],tag+' endpoint '+left)
   req(budget['ordered_ids']==budget['spatial_ids']==list(range(1024)) and budget['greedy_calls']==0,tag+' keep all')
   for j,r in enumerate((native,budget)):
    evt=events[ix*2+j];keys=('task','qid','group_id','forced','output_ids','eos_ids','generation_scalars');req(evt.get('path')==('stock_native' if j==0 else 'budget1024') and all(evt[k]==r[k] for k in keys),tag+' journal')
    out=r['output_ids'];eos=r['eos_ids'];end=bool(out and out[-1] in eos);trunc=bool(len(out)==2048 and not end)
    if forced:req(out==[forced_id]*33 and eos==[] and r['eos'] is False and r['truncated'] is False,tag+' forced')
    else:req(bool(out) and len(out)<=2048 and (end or trunc) and r['eos']==end and r['truncated']==trunc,tag+' natural endpoint')
    req(tokenizer.decode(out,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['answer'],tag+' decode')
    positions=r['prefill_position_ids'];n=len(positions[0][0]);delta=(r['rope_delta'] if j==0 else r['adjusted_delta'])[0][0];trace=r['decode_trace'];req(len(trace)==len(out),tag+' call count')
    for step,z in enumerate(trace):
     cp=0 if step==0 else n+step-1
     req(z['rope_delta']==[[delta]] and z['layer_cache_lengths'] in (([],[0]*36) if step==0 else ([cp]*36,)) and z['layer_cache_lengths_after']==[n+step]*36,tag+' KV length step '+str(step))
     req(z['cache_position']==(list(range(n)) if step==0 else [cp]) and z['position_ids']==(positions if step==0 else [[[cp+delta]]]*3),tag+' position '+str(step))
    fp=r['cache_fingerprints'];req(len(fp)==(33 if forced else 1) and all(len(x)==36 and all(set(y)=={'key','value'} for y in x) for x in fp),tag+' KV content fingerprint inventory')
    archive=Path(r['arrays']['path']).resolve();req(archive.parent==(run/'arrays').resolve() and sha(archive)==r['arrays']['sha256'] and proof['array_sha256'].get(str(archive))==r['arrays']['sha256'],tag+' archive')
   with np.load(native['arrays']['path'],allow_pickle=False) as left,np.load(budget['arrays']['path'],allow_pickle=False) as right:
    req(set(left.files)==set(right.files) and all(np.array_equal(left[k],right[k]) for k in left.files),tag+' actual tensor bytes')
    names={'visual':native['main_visual'],'prefill':native['prefill_embeds'],'prefill_logits':native['prefill_logits'],**{'ds'+str(k):v for k,v in enumerate(native['full_deepstack'])},**{'injected'+str(k):v for k,v in enumerate(native['injected_deepstack'])},**{'processor_'+k:v for k,v in native['processor'].items()}}
    for k,m in names.items():req(hashlib.sha256(left[k].tobytes()).hexdigest()==m['sha256'],tag+' tensor hash '+k)
    bits=left['prefill_logits'].tobytes();dtype=native['prefill_logits']['dtype'];finite=np.all((np.frombuffer(bits,dtype=np.uint16)&0x7f80)!=0x7f80) if dtype=='torch.bfloat16' else np.all(np.isfinite(np.frombuffer(bits,dtype={'torch.float16':np.float16,'torch.float32':np.float32,'torch.float64':np.float64}[dtype])))
    req(finite,tag+' finite logits')
   req(markers[ix]=={'task':anchor['task'],'qid':str(anchor['qid']),'group_id':anchor['group_id'],'forced':forced,'status':'passed'},tag+' pair marker')
   if len(errors)==before:verified+=1
  req(not torch.cuda.is_initialized(),'no CUDA')
 except Exception as exc:
  import traceback;traceback.print_exc();errors.append('exception: '+repr(exc))
 result={'status':'passed' if not errors and verified==6 else 'failed','pairs_expected':6,'pairs_verified':verified,'checks':checks,'errors':errors,'formal_proof_sha256':sha(E/'native_001_verification.json'),'formal_verifier_terminal_sha256':sha(E/'monitor_verify_native001/terminal.json'),'reviewer_sha256':sha(__file__),'scope':'three DEV anchors, six actual natural/forced stock-vs-budget pairs; full archived tensors verified; KV content comparison via captured hashes, not raw cache arrays; no GT/scoring','start_unix':start,'end_unix':time.time()}
 with (R/'e4native001_review.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(result));return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
