"""Independent E4 smoke review; consumes completed proof, no GT/scoring/GPU."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from collections import Counter
import numpy as np
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');E=B/'e4';Q=B/'e1';R=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def tensor(a):return hashlib.sha256(a.tobytes()).hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def lines(p):
 raw=Path(p).read_bytes()
 if raw and not raw.endswith(b'\n'):raise ValueError('incomplete JSONL')
 return [json.loads(x) for x in raw.splitlines()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--proof',type=Path,required=True);ap.add_argument('--verify-terminal',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();errors=[];checks=0;verified=0;forced_verified=0;starts=time.time();ends=Counter()
 def req(v,m):
  nonlocal checks
  checks+=1
  if not bool(v):errors.append(m)
 try:
  import torch
  from transformers import AutoTokenizer
  run=a.run.resolve();f=rd(run/'frozen.json');done=rd(run/'worker_complete.json');official=rd(a.proof);ot=rd(a.verify_terminal);launch=rd(run/'launch.json');ex=rd(run/'exit.json');tm=rd(E/'monitor_smoke001/terminal.json')
  req(official.get('status')=='passed' and official.get('all_arms_validated')==256 and official.get('forced_arms_validated')==8 and official.get('forced_decode_steps')==32 and official.get('GT_opened') is False,'official smoke proof')
  req(ot.get('reason')=='completed' and ot.get('worker_returncode')==0 and ot.get('remaining_process_group')==[],'official verifier actual exit')
  for p,d in official['files_sha256'].items():req(sha(p)==d,'official file binding '+p)
  req(official['verifier_sha256']==sha(E/'verify_k256.py'),'official verifier source')
  req(tm.get('reason')=='completed' and tm.get('worker_returncode')==0 and tm.get('remaining_process_group')==[] and tm.get('worker_pid')==launch['controller']['pid'],'actual supervisor exit')
  req(ex.get('returncode')==0 and ex.get('container_exit_code')==0 and ex.get('timed_out') is False and ex.get('interrupted') is None and ex.get('container_stopped_verified') is True and ex.get('client_exited_while_container_running') is False and ex.get('gpu_pids_after')==[],'actual container exit')
  for ident in (ex['container_identity']['host_init'],ex['controller']):
   p=Path('/proc')/str(ident['pid'])/'stat';active=False
   if p.exists() and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==ident['boot_id']:active=int(p.read_text().rsplit(')',1)[1].split()[19])==ident['start_ticks']
   req(not active,'original container/controller exited')
  inspected=subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,timeout=10)
  req(inspected.returncode==0 and not json.loads(inspected.stdout)[0]['State']['Running'] and json.loads(inspected.stdout)[0]['State']['ExitCode']==0,'independent docker inspection')
  review=rd(E/'independent_review_023.json');cpu=rd(E/'cpu_check_023/verification.json');req(review['status']=='passed' and review['independent_review'] is True and review['cpu_proof_sha256']==sha(E/'cpu_check_023/verification.json') and cpu['status']=='passed','prelaunch independent CPU proof')
  for path,digest in review['source_sha256'].items():req(sha(path)==digest,'approved source '+path)
  for path,digest in f['code_sha256'].items():req(sha(path)==digest,'frozen worker source '+path)
  e1proof=Q/'e1full001_completion.json';req(f['e1_acceptance_sha256']==sha(e1proof) and rd(R/'e1full001_review.json')['status']=='passed','accepted E1')
  inp=lines(rd(e1proof)['files']['inputs']['path']);bykey={(r['task'],str(r['qid'])):r for r in inp};e1=lines(Q/'e1full001/raw_predictions.jsonl');ref={(r['task'],str(r['qid']),r['arm']):r for r in e1}
  selected=f['selected'];req(len(selected)==32 and len({r['group_id'] for r in selected})==32 and Counter(r['task'] for r in selected)=={'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10},'smoke stratified inventory')
  natural=lines(run/'records.jsonl');forced=lines(run/'forced_records.jsonl');generated=lines(run/'generated.jsonl');allrows=natural+forced
  expected=[(r['task'],str(r['qid']),k,on) for r in selected for k in (128,256,512,1024) for on in (False,True)]
  req([(r['task'],str(r['qid']),r['budget'],r['ds_on']) for r in natural]==expected and len(forced)==8 and len(generated)==264 and done['n_natural_generations']==256 and done['n_forced_generations']==8,'natural and forced counts/order')
  anchor=selected[0];req(f['forced_anchor']=={k:anchor[k] for k in ('task','qid','group_id')} and [(r['task'],str(r['qid']),r['budget'],r['ds_on']) for r in forced]==[(anchor['task'],str(anchor['qid']),k,on) for k in (128,256,512,1024) for on in (False,True)],'forced anchor inventory')
  plan=rd(B.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');tok=AutoTokenizer.from_pretrained(plan['recipe']['model_path'],local_files_only=True,trust_remote_code=False);forced_token=tok.encode(' the',add_special_tokens=False)[0]
  natural_lookup={(r['task'],str(r['qid']),r['budget'],r['ds_on']):r for r in natural};perquestion={}
  for i,r in enumerate(allrows):
   before=len(errors);tag=f"{r['task']}:{r['qid']}:{r['budget']}:{r['ds_on']}:{r['forced']}";item=bykey[(r['task'],str(r['qid']))];isforced=i>=256;k=r['budget'];on=r['ds_on'];out=r['output_ids'];eos=r['eos_ids']
   req(all(r[x]==item[x] for x in ('task','qid','group_id','message_sha256','image_sha256')) and r['forced'] is isforced,tag+' frozen identity')
   keys=('task','qid','group_id','budget','ds_on','output_ids','eos_ids','generation_scalars','forced');req(generated[i]=={x:r[x] for x in keys},tag+' immediate journal')
   if isforced:req(out==[forced_token]*33 and eos==[] and r['forced_token']==forced_token and not r['eos'] and not r['truncated'],tag+' forced prefix')
   else:
    end=bool(out and out[-1] in eos);trunc=bool(len(out)==2048 and not end);req(bool(out) and len(out)<=2048 and r['eos']==end and r['truncated']==trunc and (end or trunc),tag+' natural endpoint');ends['eos']+=end;ends['truncated']+=trunc
   req(tok.decode(out,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['answer'],tag+' token decode')
   arm='P3-D1' if on else 'P3-D0';e=ref[(r['task'],str(r['qid']),arm)];scalars=dict(e['generation_scalars']);scalars['max_new_tokens']=33 if isforced else 2048
   req(r['generation_scalars']==scalars and r['processor']==e['processor'] and r['main_visual']==e['visual_sha256'] and r['full_deepstack']==e['deepstack_full_sha256'] and r['full_position_ids']==e['original_position_ids'],tag+' accepted E1 original input')
   S=r['image_start'];L=r['original_length'];order=r['ordered_ids'];spatial=r['spatial_ids'];ret=list(range(S))+[S+x for x in spatial]+list(range(S+1024,L));n=len(ret)
   req(k in (128,256,512,1024) and len(order)==len(set(order))==len(spatial)==k and sorted(order)==spatial and all(type(x) is int and 0<=x<1024 for x in order) and r['retained_ids']==ret and r['greedy_calls']==(0 if k==1024 else 1),tag+' budget/selection')
   full=np.asarray(r['full_position_ids'],dtype=np.int64).reshape(3,L);v=full[:,S:S+1024]-S;T,H,W=(len(np.unique(v[j])) for j in range(3));expect=np.stack((np.repeat(np.arange(T),H*W),np.tile(np.repeat(np.arange(H),W),T),np.tile(np.arange(W),T*H))) if T*H*W==1024 else np.empty((3,0))
   req(np.array_equal(v,expect),tag+' original grid geometry')
   delta=int(full.max())+1-L+1024-k;req(r['prefill_position_ids']==full[:,ret][:,None,:].tolist() and r['adjusted_delta']==[[delta]],tag+' P3 positions/delta')
   trace=r['decode_trace'];req(len(trace)==len(out),tag+' total LLM calls')
   for step,z in enumerate(trace):
    cp=0 if step==0 else n+step-1;after=n+step
    req(z['rope_delta']==[[delta]] and z['layer_cache_lengths'] in (([],[0]*36) if step==0 else ([cp]*36,)) and z['layer_cache_lengths_after']==[after]*36,tag+' KV '+str(step))
    req(z['cache_position']==(list(range(n)) if step==0 else [cp]) and z['position_ids']==(r['prefill_position_ids'] if step==0 else [[[cp+delta]]]*3),tag+' decode position '+str(step))
   mask=[[False]*S+[True]*k+[False]*(L-S-1024)];req((r['injected_mask']==mask and r['injected_deepstack']==r['selected_deepstack']) if on else (r['injected_mask'] is None and r['injected_deepstack'] is None),tag+' actual DS mask')
   p=Path(r['arrays']['path']);req(p.parent==run/'arrays' and sha(p)==r['arrays']['sha256'] and official['array_sha256'].get(str(p))==r['arrays']['sha256'],tag+' archive binding')
   with np.load(p,allow_pickle=False) as z:
    for j,key in enumerate(('visual','ds0','ds1','ds2')):
     meta=([r['main_visual']]+r['full_deepstack'])[j];sel=([r['pruned_visual']]+r['selected_deepstack'])[j]
     req(tensor(z[key])==meta['sha256'] and tensor(z[key][spatial])==sel['sha256'],tag+' full/selected '+key)
    pre=z['prefill'];req(tensor(pre)==r['prefill_embeds']['sha256'] and np.array_equal(pre[0,S:S+k],z['visual'][spatial]),tag+' actual prefill')
    if on:
     for j in range(3):req(np.array_equal(z['injected'+str(j)],z['ds'+str(j)][spatial]) and tensor(z['injected'+str(j)])==r['injected_deepstack'][j]['sha256'],tag+' actual DS '+str(j))
    textid=tensor(pre[0,:S])+tensor(pre[0,S+k:]);key=(r['task'],str(r['qid']))
    if key in perquestion:req(perquestion[key]==textid,tag+' retained text')
    else:perquestion[key]=textid
    if r.get('prefill_logits') is not None:
     zlog=z['prefill_logits'];meta=r['prefill_logits'];bits=zlog.tobytes();dtype=meta['dtype'];req(tensor(zlog)==meta['sha256'],tag+' logits hash')
     if dtype=='torch.bfloat16':finite=np.all((np.frombuffer(bits,dtype=np.uint16)&0x7f80)!=0x7f80)
     else:finite=np.all(np.isfinite(np.frombuffer(bits,dtype={'torch.float16':np.float16,'torch.float32':np.float32,'torch.float64':np.float64}[dtype])))
     req(finite,tag+' finite logits')
    if isforced:
     for key in ('input_ids','attention_mask','pixel_values','image_grid_thw'):req(tensor(z['processor_'+key])==r['processor'][key]['sha256'],tag+' raw processor '+key)
   if isforced:
    x=natural_lookup[(r['task'],str(r['qid']),k,on)];fields=('processor','ordered_ids','spatial_ids','retained_ids','main_visual','pruned_visual','full_deepstack','selected_deepstack','injected_mask','injected_deepstack','full_position_ids','full_delta','prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','prefill_logits')
    req(all(r[xk]==x[xk] for xk in fields),tag+' forced versus natural prefill')
   if k==256 and not isforced:
    mapping={'processor':'processor','ordered_ids':'ordered_ids','spatial_ids':'spatial_ids','retained_ids':'retained_ids','main_visual':'visual_sha256','pruned_visual':'pruned_sha256','full_deepstack':'deepstack_full_sha256','selected_deepstack':'selected_deepstack','prefill_position_ids':'prefill_position_ids','adjusted_delta':'rope_delta','prefill_embeds':'prefill_embed_sha256','prefill_mask':'attention_mask_sha256','output_ids':'output_ids','answer':'answer'}
    req(all(r[x]==e[y] for x,y in mapping.items()),tag+' 256 E1 endpoint')
   if len(errors)==before:
    if isforced:forced_verified+=1
    else:verified+=1
  req(not torch.cuda.is_initialized(),'no CUDA initialized')
 except Exception as exc:
  import traceback;traceback.print_exc();errors.append('exception: '+repr(exc))
 result={'status':'passed' if not errors and verified==256 and forced_verified==8 else 'failed','natural_expected':256,'natural_verified':verified,'forced_expected':8,'forced_verified':forced_verified,'checks':checks,'errors':errors,'natural_eos':ends['eos'],'natural_truncated':ends['truncated'],'proof_sha256':sha(a.proof),'verifier_terminal_sha256':sha(a.verify_terminal),'reviewer_sha256':sha(__file__),'scope':'32 DEV smoke questions; 256 natural plus 8 forced, full arrays, all decode/KV, no GT, scores or CUDA; no full-panel reuse approval','start_unix':starts,'end_unix':time.time()}
 with a.out.open('x') as file:json.dump(result,file,indent=2);file.write('\n');file.flush();os.fsync(file.fileno())
 print(json.dumps(result));return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
