"""Independent E1 DEV review, no reference answers or scoring; metadata only."""
import argparse,hashlib,json,os,time,subprocess
from collections import Counter
from pathlib import Path
import numpy as np
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');E=B/'e1';ROOT=B.parents[2]
ARMS=('P1-D0','P1-D1','P3-D0','P3-D1');TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(4*1024*1024),b''):h.update(z)
 return h.hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def lines(p):
 raw=Path(p).read_bytes()
 if raw and not raw.endswith(b'\n'):raise ValueError('incomplete journal '+str(p))
 return [json.loads(x) for x in raw.splitlines()]
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);opt=a.parse_args();start=time.time();checks=0;errors=[];verified=0;cross=[];ends=Counter()
 def req(ok,msg):
  nonlocal checks
  checks+=1
  if not bool(ok):errors.append(msg)
 try:
  import torch,transformers
  from transformers import AutoTokenizer
  run=E/'e1full001';proofpath=E/'e1full001_completion.json';proof=rd(proofpath);mon=E/'monitorverifye1full001';started=rd(mon/'started.json');terminal=rd(mon/'terminal.json')
  req(proof.get('status')=='passed' and proof.get('natural_questions')==384 and proof.get('natural_records')==1536 and proof.get('tokenizer_decode_checked')==1536 and proof.get('GT_opened') is False and proof.get('scoring_performed') is False,'completion proof')
  req(terminal.get('reason')=='completed' and terminal.get('worker_returncode')==0 and terminal.get('remaining_process_group')==[] and terminal.get('worker_pid')==started.get('worker_pid'),'completion supervisor exit')
  cmd=started['command'];cwd=Path(started['cwd']);req(any((cwd/x).resolve()==(B/'e1_analysis/completion.py') for x in cmd if Path(x).name=='completion.py') and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==proofpath,'completion supervisor command')
  req(proof['verifier_sha256']==sha(E/'e1full001_sources/completion.py'),'completion snapshot')
  for k,v in proof['files'].items():req(sha(v['path'])==v['sha256'],'completion evidence '+k)
  f=rd(run/'frozen.json');launch=rd(E/'e1full001_launch.json');ex=rd(E/'e1full001_exit.json');tm=rd(E/'monitore1full001/terminal.json');done=rd(run/'worker_complete.json')
  req(done.get('n')==384 and f.get('stage')=='full' and f.get('n_questions')==384 and f.get('gt_read') is False and f.get('scoring') is False,'run completion')
  req(tm.get('reason')=='completed' and tm.get('worker_returncode')==0 and tm.get('remaining_process_group')==[] and tm.get('worker_pid')==launch['controller']['pid'],'launcher supervisor exit')
  req(ex.get('returncode')==0 and ex.get('timed_out') is False and ex.get('interrupted') is None and ex.get('container_stopped_verified') is True and ex.get('client_exited_while_container_running') is False and ex.get('gpu_pids_after')==[],'docker exit')
  req(ex['controller']==launch['controller'],'controller identity')
  for ident in (ex['controller'],ex['container_identity']['host_init']):
   stat=Path('/proc')/str(ident['pid'])/'stat';alive=False
   if stat.exists() and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==ident['boot_id']:
    alive=int(stat.read_text().rsplit(')',1)[1].split()[19])==ident['start_ticks']
   req(not alive,'worker process exited')
  inspect=subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,timeout=10)
  req((inspect.returncode==0 and not json.loads(inspect.stdout)[0]['State']['Running']) or (inspect.returncode!=0 and ('No such object' in inspect.stderr or 'No such container' in inspect.stderr)),'container stopped')
  src=E/'e1full001_sources';pre=rd(B/'e1_independent_review/e1full_prelaunch_review.json')
  for path,digest in launch['source_sha256'].items():
   req(sha(src/Path(path).name)==digest,'launch snapshot '+path)
   if path in pre['source_sha256']:req(pre['source_sha256'][path]==digest,'prelaunch peer '+path)
  inputs=lines(proof['files']['inputs']['path']);panel=[x for x in lines(proof['files']['panel']['path']) if x['split']=='DEV']; records=lines(run/'raw_predictions.jsonl');generated=lines(run/'generated_arms.jsonl');quartets=lines(run/'validated_quartets.jsonl')
  req(len(inputs)==len(panel)==384 and len(records)==len(generated)==1536 and len(quartets)==384,'inventory counts')
  req(Counter(x['task'] for x in panel)=={x:128 for x in TASKS} and len({x['group_id'] for x in panel})==384,'DEV task and group inventory')
  req(all(all(str(i[k])==str(p[k]) for k in ('task','qid','group_id')) for i,p in zip(inputs,panel)),'frozen input order')
  plan=rd(proof['files']['plan']['path']);recipe=plan['recipe'];model=Path(recipe['model_path']);tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False);gc=rd(model/'generation_config.json');eos=gc['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
  expected_scalars={'do_sample':False,'max_new_tokens':2048,'top_p':recipe['top_p'],'top_k':recipe['top_k'],'temperature':recipe['temperature'],'repetition_penalty':recipe['repetition_penalty']}
  smoke=lines(E/'e1smoke001/raw_predictions.jsonl');sm={(r['task'],str(r['qid']),r['arm']):r for r in smoke if not r['forced']}
  for i,row in enumerate(inputs):
   before=len(errors);tag=f"{row['task']}:{row['qid']}";q=quartets[i]
   req(q=={'task':row['task'],'qid':row['qid'],'group_id':row['group_id'],'forced':False},tag+' quartet')
   req(sha(row['image_path'])==row['image_sha256'] and hashlib.sha256(json.dumps(row['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()==row['message_sha256'],tag+' image/message')
   baseline=None
   for j,arm in enumerate(ARMS):
    r=records[4*i+j];g=generated[4*i+j];t=tag+':'+arm
    req(all(r[k]==row[k] for k in ('task','qid','group_id','message_sha256')) and r['arm']==arm and r['forced'] is False,t+' identity')
    req(g=={k:r[k] for k in ('task','qid','group_id','arm','forced','output_ids','eos_ids')},t+' journal')
    out=r['output_ids'];end=bool(out and out[-1] in eos);trunc=bool(out and len(out)==2048 and not end)
    req(r['generation_scalars']==expected_scalars and r['eos_ids']==eos and 0<len(out)<=2048 and r['eos']==end and r['truncated']==trunc and (end or trunc),t+' recipe/end')
    req(tokenizer.decode(out,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['answer'],t+' decoder')
    ends['eos']+=end;ends['truncated']+=trunc
    S=r['image_start'];L=r['original_length'];o=r['ordered_ids'];s=r['spatial_ids'];ret=list(range(S))+[S+x for x in s]+list(range(S+1024,L));n=len(ret)
    req(0<=S and S+1024<=L and len(o)==len(set(o))==len(s)==256 and all(type(x) is int and 0<=x<1024 for x in o) and sorted(o)==s and r['retained_ids']==ret,t+' selection and retained')
    full=np.array(r['original_position_ids'],dtype=np.int64).reshape(3,L);v=full[:,S:S+1024]-S
    tv,hv,wv=(np.unique(v[x]) for x in range(3));T,H,W=len(tv),len(hv),len(wv)
    expected=np.stack((np.repeat(np.arange(T),H*W),np.tile(np.repeat(np.arange(H),W),T),np.tile(np.arange(W),T*H))) if T*H*W==1024 else np.empty((3,0))
    req(np.array_equal(v,expected) and np.array_equal(full[:,:S],np.broadcast_to(np.arange(S),(3,S))) and np.array_equal(full[:,S+1024:],np.broadcast_to(np.arange(S+max(T,H,W),S+max(T,H,W)+L-S-1024),(3,L-S-1024))),t+' original mRoPE geometry')
    fd=int(full.max())+1-L;req(r['full_rope_delta']==[[fd]],t+' original delta')
    p=np.broadcast_to(np.arange(n),(3,n)) if arm.startswith('P1') else full[:,ret];req(r['prefill_position_ids']==p[:,None,:].tolist() and r['rope_delta']==[[0 if arm.startswith('P1') else fd+768]],t+' position/delta')
    mask=np.ones((1,n),dtype=np.int64);req(r['attention_mask_sha256']=={'dtype':'torch.int64','shape':[1,n],'sha256':hashlib.sha256(mask.tobytes()).hexdigest()},t+' prefill attention')
    ds=r['selected_deepstack'];req(len(ds)==3 and all(x['shape']==[256,4096] and x['dtype']=='torch.bfloat16' for x in ds),t+' three selected DS')
    im=[[False]*S+[True]*256+[False]*(L-S-1024)]
    req((r['injected_mask']==im and r['injected_deepstack']==ds) if arm.endswith('D1') else (r['injected_mask'] is None and r['injected_deepstack'] is None),t+' injection')
    req(r['visual_sha256']['shape']==[1024,4096] and r['pruned_sha256']['shape']==[256,4096] and len(r['deepstack_full_sha256'])==3 and all(x['shape']==[1024,4096] for x in r['deepstack_full_sha256']) and r['prefill_embed_sha256']['shape']==[1,n,4096] and len(r['logits'])==1,t+' feature/logit shapes')
    common=[r[x] for x in ('processor','image_start','original_length','ordered_ids','spatial_ids','retained_ids','original_position_ids','full_rope_delta','visual_sha256','pruned_sha256','deepstack_full_sha256','selected_deepstack','prefill_embed_sha256','attention_mask_sha256')]
    if baseline is None:baseline=common
    else:req(common==baseline,t+' common prefill')
    old=sm.get((row['task'],str(row['qid']),arm))
    if old is not None:
     fields=('processor','ordered_ids','spatial_ids','retained_ids','original_position_ids','full_rope_delta','visual_sha256','pruned_sha256','deepstack_full_sha256','selected_deepstack','prefill_embed_sha256','attention_mask_sha256','prefill_position_ids','rope_delta','injected_mask','injected_deepstack','output_ids','answer','eos_ids','generation_scalars','logits')
     cross.append({'task':row['task'],'qid':str(row['qid']),'arm':arm,'identical':all(r[k]==old[k] for k in fields)})
   if len(errors)==before:verified+=1
  req(not torch.cuda.is_initialized(),'no CUDA initialized')
 except Exception as exc:
  import traceback;traceback.print_exc();errors.append('review exception: '+repr(exc))
 report={'status':'passed' if not errors and verified==384 else 'failed','n_expected':384,'n_verified':verified,'checks':checks,'errors':errors,'natural_eos':ends['eos'],'natural_truncated':ends['truncated'],'cross_smoke_count':len(cross),'cross_smoke_all_identical':bool(cross) and all(x['identical'] for x in cross),'cross_smoke':cross,'scope':'all 1536 DEV natural arms; CPU metadata/position/journal/token decode and exit; no GT, score, model forward or independent full feature bytes (not archived)','completion_proof_sha256':sha(E/'e1full001_completion.json'),'completion_started_sha256':sha(E/'monitorverifye1full001/started.json'),'completion_terminal_sha256':sha(E/'monitorverifye1full001/terminal.json'),'reviewer_sha256':sha(__file__),'start_unix':start,'end_unix':time.time()}
 with opt.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({k:v for k,v in report.items() if k not in ('cross_smoke','errors')},ensure_ascii=False));return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
