"""Independent complete E4 DEV panel metadata review. No GT, scoring or GPU.

Full panel stores tensor summaries rather than reconstructible high-dimensional
arrays. Array reconstruction is confined to separately reviewed E4 smoke/native.
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from collections import Counter
import numpy as np
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');P=B/'e4_panel_v2';E=B/'e1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def lines(p):
 b=Path(p).read_bytes()
 if b and not b.endswith(b'\n'):raise ValueError('partial journal')
 return [json.loads(x) for x in b.splitlines()]
def alive(ident):
 path=Path('/proc')/str(ident['pid'])/'stat'
 return path.exists() and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==ident['boot_id'] and int(path.read_text().rsplit(')',1)[1].split()[19])==ident['start_ticks']
def main():
 ap=argparse.ArgumentParser()
 for name in ('run','proof','verify-started','verify-terminal','run-started','run-terminal','out'):ap.add_argument('--'+name,type=Path,required=True)
 a=ap.parse_args();start=time.time();errors=[];checks=0;verified=0;arms=0;ends=Counter();diff=0
 def req(v,msg):
  nonlocal checks
  checks+=1
  if not bool(v):errors.append(msg)
 try:
  import torch
  from transformers import AutoTokenizer
  run=a.run.resolve();formal=rd(a.proof);vs=rd(a.verify_started);vt=rd(a.verify_terminal);rs=rd(a.run_started);rt=rd(a.run_terminal)
  req(formal.get('status')=='passed' and formal.get('stage')=='full' and formal.get('n_questions')==384 and formal.get('all_arms_validated')==3072 and formal.get('k256_reuse') is False and formal.get('GT_opened') is False and formal.get('scoring_performed') is False,'formal proof')
  req(vt.get('reason')=='completed' and vt.get('worker_returncode')==0 and vt.get('remaining_process_group')==[] and vt.get('worker_pid')==vs.get('worker_pid'),'formal supervisor exit')
  cmd=vs['command'];cwd=Path(vs['cwd']);req([(cwd/x).resolve() for x in cmd if Path(x).name=='panel_verify.py']==[(P/'panel_verify.py').resolve()] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==a.proof.resolve(),'formal supervisor command')
  req(formal.get('verifier_sha256')==sha(P/'panel_verify.py'),'formal verifier source')
  for path,digest in formal['files_sha256'].items():req(sha(path)==digest,'formal evidence '+path)
  for path in (a.run_started,a.run_terminal,run/'launch.json',run/'exit.json',run/'frozen.json',run/'worker_complete.json',run/'records.jsonl',run/'generated.jsonl'):
   req(formal['files_sha256'].get(str(path.resolve()))==sha(path),'required formal evidence '+str(path))
  launch=rd(run/'launch.json');exit=rd(run/'exit.json');frozen=rd(run/'frozen.json');done=rd(run/'worker_complete.json')
  req(rt.get('reason')=='completed' and rt.get('worker_returncode')==0 and rt.get('remaining_process_group')==[] and rt.get('worker_pid')==rs.get('worker_pid')==launch['controller']['pid'],'run supervisor exit')
  req(rs.get('timeout_seconds')==10920 and rs.get('stale_seconds')==1800 and rs.get('snapshot_seconds')==3600 and rs.get('progress_paths')==[str(run/'generated.jsonl')],'bounded supervision')
  rc=rs['command'];rcwd=Path(rs['cwd']);dock=launch['command'];budget=rd(P/'budget_plan.json')
  req([(rcwd/x).resolve() for x in rc if Path(x).name=='panel_launch.py']==[(P/'panel_launch.py').resolve()] and rc.count('--run-id')==1 and run.name=='panel_'+rc[rc.index('--run-id')+1] and dock.count(str(P/'panel_worker.py'))==1 and dock.count('--out')==1 and Path(dock[dock.index('--out')+1]).resolve()==run,'supervisor and Docker commands')
  req(launch['worker_sha256']==sha(P/'panel_worker.py') and launch['launcher_sha256']==sha(P/'panel_launch.py') and launch['budget_plan_sha256']==sha(P/'budget_plan.json') and launch['started_unix']+budget['outer_timeout_seconds']<=budget['budget_end_unix'],'worker sources and deadline')
  req(exit.get('controller')==launch['controller'] and exit.get('returncode')==exit.get('container_exit_code')==0 and exit.get('timed_out') is False and exit.get('interrupted') is None and exit.get('container_stopped_verified') is True and exit.get('client_exited_while_container_running') is False and exit.get('gpu_pids_after')==[],'container exit chain')
  ident=exit['container_identity'];req(not alive(exit['controller']) and not alive(ident['host_init']),'recorded processes stopped')
  inspect=subprocess.run(['docker','inspect',ident['container_id']],capture_output=True,text=True,timeout=10,check=True);state=json.loads(inspect.stdout)[0]
  req(state['Id']==ident['container_id'] and state['State']['Running'] is False and state['State']['ExitCode']==0,'independent Docker inspection')
  req(frozen.get('stage')=='full' and frozen.get('n_questions')==384 and frozen.get('k256_reuse') is False and frozen.get('gt_read') is False and frozen.get('scoring') is False and frozen.get('tensor_retention')=='summary_only_runtime_checked_no_tensor_reconstruction' and done.get('n_questions')==384 and done.get('n_natural_generations')==3072 and not (run/'arrays').exists(),'run protocol and summary limitation')
  review=rd(P/'independent_review_005.json');req(review.get('status')=='passed' and review.get('independent_review') is True and launch['approval_bundle']['review']['sha256']==sha(P/'independent_review_005.json'),'prelaunch review')
  for path,digest in review['source_sha256'].items():req(sha(path)==digest,'approved source '+path)
  for path,digest in frozen['code_sha256'].items():req(sha(path)==digest,'runtime source '+path)
  e1path=E/'e1full001_completion.json';ep=rd(e1path);req(frozen['e1_acceptance_sha256']==sha(e1path) and ep['status']=='passed' and ep['GT_opened'] is False,'E1 input proof')
  panel=lines(ep['files']['inputs']['path']);refs=lines(ep['files']['journal']['path']);reference={(x['task'],str(x['qid']),x['arm']):x for x in refs if x['forced'] is False}
  req(len(panel)==384 and len(reference)==1536 and Counter(x['task'] for x in panel)=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128} and len({x['group_id'] for x in panel})==384,'DEV inventory')
  req(frozen['selected']==[{k:x[k] for k in ('task','qid','group_id','message_sha256','image_sha256')} for x in panel],'frozen selected rows')
  records=lines(run/'records.jsonl');journal=lines(run/'generated.jsonl');expected=[(x['task'],str(x['qid']),k,on) for x in panel for k in (128,256,512,1024) for on in (False,True)]
  req(len(records)==len(journal)==3072 and [(r['task'],str(r['qid']),r['budget'],r['ds_on']) for r in records]==expected,'complete record inventory')
  recipe=rd(ep['files']['plan']['path'])['recipe'];tok=AutoTokenizer.from_pretrained(recipe['model_path'],local_files_only=True,trust_remote_code=False)
  scalars={'do_sample':False,'max_new_tokens':2048,'top_p':recipe['top_p'],'top_k':recipe['top_k'],'temperature':recipe['temperature'],'repetition_penalty':recipe['repetition_penalty']}
  for i,item in enumerate(panel):
   before=len(errors);tag=f"{item['task']}:{item['qid']}";text=None
   req(sha(item['image_path'])==item['image_sha256'] and hashlib.sha256(json.dumps(item['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()==item['message_sha256'],tag+' source image/message')
   for j in range(8):
    r=records[8*i+j];g=journal[8*i+j];k=r['budget'];on=r['ds_on'];ref=reference[(item['task'],str(item['qid']),'P3-D1' if on else 'P3-D0')];t=tag+f':{k}:{on}';prev=len(errors)
    req(all(r[x]==item[x] for x in ('task','qid','group_id','image_sha256','message_sha256')) and r['forced'] is False,t+' input')
    keys=('task','qid','group_id','budget','ds_on','output_ids','eos_ids','generation_scalars','forced');req(g=={x:r[x] for x in keys},t+' generation journal')
    out=r['output_ids'];eos=r['eos_ids'];end=bool(out and out[-1] in eos);trunc=bool(len(out)==2048 and not end)
    req(r['generation_scalars']==scalars and eos==ref['eos_ids'] and r['max_new_tokens']==2048 and 0<len(out)<=2048 and r['eos']==end and r['truncated']==trunc and (end or trunc),t+' generation endpoint');ends['eos']+=end;ends['truncated']+=trunc
    req(tok.decode(out,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['answer'],t+' token decode')
    req(r['processor']==ref['processor'] and r['main_visual']==ref['visual_sha256'] and r['full_deepstack']==ref['deepstack_full_sha256'] and r['full_position_ids']==ref['original_position_ids'],t+' E1 raw input metadata')
    s=r['image_start'];L=r['original_length'];order=r['ordered_ids'];spatial=r['spatial_ids'];ret=list(range(s))+[s+x for x in spatial]+list(range(s+1024,L));n=len(ret)
    req(k in (128,256,512,1024) and 0<=s and s+1024<=L and len(order)==len(set(order))==len(spatial)==k and sorted(order)==spatial and all(type(x) is int and 0<=x<1024 for x in order) and r['retained_ids']==ret and r['selection_mode']==('keep_all' if k==1024 else 'author_greedy') and (k!=1024 or order==list(range(1024))),t+' selected indices')
    full=np.asarray(r['full_position_ids'],dtype=np.int64).reshape(3,L);v=full[:,s:s+1024]-s;T,H,W=(len(np.unique(v[x])) for x in range(3))
    grid=np.stack((np.repeat(np.arange(T),H*W),np.tile(np.repeat(np.arange(H),W),T),np.tile(np.arange(W),T*H))) if T*H*W==1024 else np.empty((3,0))
    delta=int(full.max())+1-L+1024-k
    req(np.array_equal(v,grid) and r['full_delta']==ref['full_rope_delta']==[[int(full.max())+1-L]] and r['prefill_position_ids']==full[:,ret][:,None,:].tolist() and r['adjusted_delta']==[[delta]],t+' original P3 position')
    trace=r['decode_trace'];req(len(trace)==len(out),t+' actual forward count')
    for step,z in enumerate(trace):
     cp=n+step-1
     req(z['rope_delta']==[[delta]] and z['layer_cache_lengths'] in (([],[0]*36) if step==0 else ([cp]*36,)) and z['layer_cache_lengths_after']==[n+step]*36 and z['cache_position']==(list(range(n)) if step==0 else [cp]) and z['position_ids']==(r['prefill_position_ids'] if step==0 else [[[cp+delta]]]*3),t+' actual decode '+str(step))
    mask=[[False]*s+[True]*k+[False]*(L-s-1024)]
    req((r['injected_mask']==mask and r['injected_deepstack']==r['selected_deepstack']) if on else (r['injected_mask'] is None and r['injected_deepstack'] is None),t+' actual DS injection')
    req(r['actual_injection_verified'] is True and r['visual_calls']==r['main_merger_calls']==1 and bool(r['block_calls']) and all(x==1 for x in r['block_calls']) and r['ds_calls']==[1,1,1] and r['greedy_calls']==(0 if k==1024 else 1),t+' observed calls')
    arrays=r['arrays'];tensors=arrays['tensors'];names={'visual','prefill','ds0','ds1','ds2'}|({'injected0','injected1','injected2'} if on else set())
    req(arrays['kind']=='summary_only' and set(tensors)==names and tensors['visual']==r['main_visual'] and tensors['prefill']==r['prefill_embeds'] and all(tensors['ds'+str(x)]==r['full_deepstack'][x] for x in range(3)) and (not on or all(tensors['injected'+str(x)]==r['injected_deepstack'][x] for x in range(3))),t+' summary inventory')
    metas=[r['main_visual']]+r['full_deepstack'];sels=[r['pruned_visual']]+r['selected_deepstack']
    req(all(x['dtype']=='torch.bfloat16' and x['shape']==[1024,4096] for x in metas) and all(x['dtype']=='torch.bfloat16' and x['shape']==[k,4096] for x in sels) and (k!=1024 or metas==sels) and r['prefill_embeds']['shape']==[1,n,4096],t+' feature summary shape')
    maskmeta={'dtype':'torch.int64','shape':[1,n],'sha256':hashlib.sha256(np.ones((1,n),dtype=np.int64).tobytes()).hexdigest()};req(r['prefill_mask']==maskmeta,t+' attention mask')
    textpair=(r['prefill_text_before'],r['prefill_text_after'])
    if text is None:text=textpair
    else:req(text==textpair,t+' retained text')
    if k==256:
     mapping={'ordered_ids':'ordered_ids','spatial_ids':'spatial_ids','retained_ids':'retained_ids','pruned_visual':'pruned_sha256','selected_deepstack':'selected_deepstack','prefill_embeds':'prefill_embed_sha256','prefill_mask':'attention_mask_sha256','prefill_position_ids':'prefill_position_ids','adjusted_delta':'rope_delta'}
     req(all(r[x]==ref[y] for x,y in mapping.items()),t+' K256 E1 prefill')
     diff+=any(r[x]!=ref[x] for x in ('output_ids','answer','eos_ids','eos','truncated'))
    if len(errors)==prev:arms+=1
   if len(errors)==before:verified+=1
  req(not torch.cuda.is_initialized(),'CUDA not initialized')
 except Exception as exc:
  import traceback;traceback.print_exc();errors.append('review exception: '+repr(exc))
 result={'status':'passed' if not errors and verified==384 and arms==3072 else 'failed','n_expected':384,'n_verified':verified,'arms_expected':3072,'arms_verified':arms,'checks':checks,'errors':errors,'natural_eos':ends['eos'],'natural_truncated':ends['truncated'],'k256_e1_output_differences':diff,'GT_opened':False,'scoring_performed':False,'formal_proof_sha256':sha(a.proof),'formal_verifier_started_sha256':sha(a.verify_started),'formal_verifier_terminal_sha256':sha(a.verify_terminal),'reviewer_sha256':sha(__file__),'scope':'384 DEV questions, 3072 natural arms; CPU metadata/journal/token/position/KV and container audit; full high-dimensional tensor reconstruction unavailable','start_unix':start,'end_unix':time.time()}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({k:v for k,v in result.items() if k!='errors'}));return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
