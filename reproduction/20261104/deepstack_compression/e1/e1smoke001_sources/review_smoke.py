"""Independent read-only E1 smoke audit: 32 questions, four arms, no GT or scoring."""
import argparse, hashlib, json, os, statistics, subprocess, time
from pathlib import Path
import numpy as np

ROOT=Path('/media/disk2/YZX/research/EADP_v2')
BASE=ROOT/'reproduction/20261104/deepstack_compression'
E1=BASE/'e1'
PLAN=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
ARMS=('P1-D0','P1-D1','P3-D0','P3-D1')
TASKS=(('DocVQA_VAL',11),('TextVQA_VAL',11),('AI2D_TEST',10))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def tensor_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def safe_sha(p):return sha(p) if Path(p).is_file() else None
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 raw=Path(p).read_bytes()
 if raw and not raw.endswith(b'\n'):raise ValueError('incomplete JSONL tail '+str(p))
 return [json.loads(x) for x in raw.splitlines()]

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
 p.add_argument('--proof',type=Path,required=True);p.add_argument('--verifier-terminal',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args(); errors=[]; checks=0;started=time.time(); timings=[]; ends=[]; verified=0; legacy_endpoints=[]
 def req(ok,msg):
  nonlocal checks
  checks+=1
  if not bool(ok):errors.append(msg)
 try:
  import torch, transformers, safetensors
  from safetensors import safe_open
  from transformers import AutoTokenizer
  run=a.run.resolve();frozen=read(run/'frozen.json');records=lines(run/'raw_predictions.jsonl');generated=lines(run/'generated_arms.jsonl');quartets=lines(run/'validated_quartets.jsonl')
  complete=read(run/'worker_complete.json')
  req(complete.get('n')==32 and len(records)==len(generated)==256 and len(quartets)==64,'complete arm/quad counts')
  launcher=E1/(run.name+'_launch.json');exit_path=E1/(run.name+'_exit.json');term_path=E1/('monitor'+run.name)/'terminal.json'
  launch=read(launcher);ex=read(exit_path);terminal=read(term_path)
  req(terminal.get('reason')=='completed' and terminal.get('worker_returncode')==0 and not terminal.get('remaining_process_group'),'launcher actual exit')
  req(ex.get('returncode')==0 and not ex.get('timed_out') and ex.get('interrupted') is None and ex.get('container_stopped_verified') is True and not ex.get('gpu_pids_after'),'container exit')
  ident=ex['container_identity'];orig=ident['host_init'];stat=Path('/proc')/str(orig['pid'])/'stat';alive=False
  if stat.exists() and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==orig['boot_id']:
   alive=int(stat.read_text().rsplit(')',1)[1].split()[19])==orig['start_ticks']
  req(not alive,'original container process exited')
  inspect=subprocess.run(['docker','inspect',ident['container_id']],capture_output=True,text=True,timeout=10)
  req((inspect.returncode==0 and not json.loads(inspect.stdout)[0]['State']['Running']) or (inspect.returncode!=0 and ('No such object' in inspect.stderr or 'No such container' in inspect.stderr)),'container stopped')
  req(frozen.get('stage')=='smoke' and frozen.get('n_questions')==32 and frozen.get('inputs_sha256')==sha(a.inputs),'frozen input/stage')
  req(frozen['plan_sha256']==sha(PLAN) and frozen['panel_sha256']==sha(BASE/'data_manifest/panel_manifest.jsonl'),'frozen plan and panel')
  input_proof=a.inputs.parent/'independent_verification.json';ip=read(input_proof)
  req(frozen['input_proof_sha256']==sha(input_proof) and ip.get('status')=='passed' and ip.get('n_verified')==384 and ip.get('inputs_sha256')==sha(a.inputs),'input identity proof')
  exclusion=BASE/'data_manifest/anchor_exclusion'
  req(frozen['anchor_exclusion']=={'summary_sha256':sha(exclusion/'summary.json'),'verification_sha256':sha(exclusion/'verification.json')},'anchor exclusion binding')
  req(frozen['weights_identity_sha256']==sha(ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json'),'weight identity manifest')
  for field,name in (('runner_sha256','runner.py'),('contract_sha256','contract.py'),('position_adapter_sha256','position_adapter.py')):
   req(frozen[field]==launch['source_sha256'][str(E1/name)],'frozen source '+field)
  sources=E1/(run.name+'_sources')
  for path,digest in launch['source_sha256'].items():
   original=Path(path)
   snapshot=sources/original.name if original.parent==E1 else original
   req(snapshot.exists() and sha(snapshot)==digest,'executed source '+path)
  all_rows=lines(a.inputs);sample=[]
  for task,n in TASKS:sample.extend([r for r in all_rows if r['task']==task][:n])
  req(len(sample)==32 and len({r['group_id'] for r in sample})==32,'sample inventory')
  panel={(x['task'],str(x['qid']),x['group_id']) for x in lines(BASE/'data_manifest/panel_manifest.jsonl') if x['split']=='DEV'}
  req(all((r['task'],str(r['qid']),r['group_id']) in panel for r in sample),'DEV only')
  e0=read(BASE/'e0_identity/anchor16_verification.json');req(e0.get('status')=='passed' and e0.get('n_verified')==16 and frozen['e0_proof_sha256']==sha(BASE/'e0_identity/anchor16_verification.json'),'E0 identity binding')
  prior=read(E1/'legacy004_verification.json');peer=read(BASE/'e1_independent_review/legacy004_review.json');peer_terminal=BASE/'e1_independent_review/monitor_legacy004/terminal.json'
  req(prior.get('status')=='passed' and prior.get('n_verified')==16 and peer.get('status')=='passed' and peer.get('n_expected')==16 and not peer.get('errors'),'legacy P1 identity proof')
  legacy_rows=lines(E1/'legacy004/comparisons.jsonl')
  legacy_lookup={(str(x['qid']),x['forced']):x['legacy'] for x in legacy_rows}
  req(frozen['legacy_gate']=={'proof_sha256':sha(E1/'legacy004_verification.json'),'review_sha256':sha(BASE/'e1_independent_review/legacy004_review.json'),'review_terminal_sha256':sha(peer_terminal)},'legacy proof binding')
  pt=read(peer_terminal);req(pt.get('reason')=='completed' and pt.get('worker_returncode')==0 and not pt.get('remaining_process_group'),'legacy peer actual exit')
  config=read(PLAN);recipe=config['recipe'];model=Path(recipe['model_path']);mc=read(model/'config.json');gc=read(model/'generation_config.json')
  eos=gc['eos_token_id'];eos=[eos] if isinstance(eos,int) else list(eos);tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
  forced_id=tokenizer.encode(' the',add_special_tokens=False)[0];merge=mc['vision_config']['spatial_merge_size']
  weights=read(ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json')['files']
  for ent in weights:
   st=Path(ent['path']).stat();actual=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
   req(frozen['weight_stats_before'].get(ent['path'])==actual,'weight stat '+ent['path'])
  weight_index=read(model/'model.safetensors.index.json')['weight_map'];key='model.language_model.embed_tokens.weight'
  with safe_open(str(model/weight_index[key]),framework='pt',device='cpu') as sf:
   embedding=sf.get_tensor(key)
   for sample_index,row in enumerate(sample):
    baseline_by_condition={}; sample_errors=len(errors)
    req(sha(row['image_path'])==row['image_sha256'],'image hash '+str(row['qid']))
    req(hashlib.sha256(json.dumps(row['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()==row['message_sha256'],'prompt hash '+str(row['qid']))
    text_cache=None
    for fi,forced in enumerate((False,True)):
     q=quartets[sample_index*2+fi]
     req(q=={'task':row['task'],'qid':row['qid'],'group_id':row['group_id'],'forced':forced},'quartet '+str(row['qid']))
     baseline=None
     for ai,arm in enumerate(ARMS):
      i=(sample_index*2+fi)*4+ai;rec=records[i];gen=generated[i];tag=f"{row['task']}:{row['qid']}:{arm}:{forced}"
      req(all(rec[k]==row[k] for k in ('task','qid','group_id','message_sha256')) and rec['arm']==arm and rec['forced']==forced,tag+' identity')
      expected_gen={k:rec[k] for k in ('task','qid','group_id','arm','forced','output_ids','eos_ids')}
      req(gen==expected_gen,tag+' generated journal')
      scalars={'do_sample':False,'max_new_tokens':33 if forced else 2048,'top_p':recipe['top_p'],'top_k':recipe['top_k'],'temperature':recipe['temperature'],'repetition_penalty':recipe['repetition_penalty']}
      req(rec['generation_scalars']==scalars,tag+' generation scalars')
      output=rec['output_ids'];req(tokenizer.decode(output,skip_special_tokens=True,clean_up_tokenization_spaces=False)==rec['answer'],tag+' output decode')
      if forced:req(output==[forced_id]*33 and rec['forced_token']==forced_id and rec['eos_ids']==[],tag+' forced tokens')
      else:
       ended=bool(output and output[-1] in eos);truncated=bool(len(output)>=2048 and not ended)
       req(rec['eos_ids']==eos and rec['eos']==ended and rec['truncated']==truncated and (ended or truncated),tag+' EOS/truncation')
       ends.append({'task':row['task'],'arm':arm,'eos':ended,'truncated':truncated})
      req(rec['wall_seconds']>0 and rec['wall_seconds_through_archive']>=rec['wall_seconds'],tag+' timing')
      timings.append({'task':row['task'],'arm':arm,'forced':forced,'generation_observed_seconds':rec['wall_seconds'],'through_archive_seconds':rec['wall_seconds_through_archive']})
      path=Path(rec['array_path']);expected_name=f"{row['task']}_{row['qid']}_{arm}_{'forced' if forced else 'natural'}.npz"
      req(path.parent==run and path.name==expected_name and sha(path)==rec['array_sha256'],tag+' NPZ hash')
      with np.load(path,allow_pickle=False) as z:
       req(sorted(z.files)==rec['array_keys'],tag+' NPZ keys')
       entries={**{'processor_'+k:v for k,v in rec['processor'].items()},'visual_main':rec['visual_sha256'],'pruned_main':rec['pruned_sha256'],
                'prefill_inputs_embeds':rec['prefill_embed_sha256'],'prefill_attention_mask':rec['attention_mask_sha256']}
       entries.update({'visual_ds_'+str(j):v for j,v in enumerate(rec['deepstack_full_sha256'])})
       entries.update({'selected_ds_'+str(j):v for j,v in enumerate(rec['selected_deepstack'])})
       entries.update({'logits_'+str(j):v for j,v in enumerate(rec['logits'])})
       if rec['injected_deepstack'] is not None:entries.update({'prefill_ds_'+str(j):v for j,v in enumerate(rec['injected_deepstack'])})
       for name,meta in entries.items():
        req(name in z.files and tensor_sha(z[name])==meta['sha256'],tag+' tensor hash '+name)
        if name.startswith('logits_'):
         bits=z[name].tobytes()
         if meta['dtype']=='torch.bfloat16':finite=np.all((np.frombuffer(bits,dtype=np.uint16)&0x7f80)!=0x7f80)
         elif meta['dtype']=='torch.float32':finite=np.all(np.isfinite(np.frombuffer(bits,dtype=np.float32)))
         else:finite=False
         req(finite,tag+' finite logits '+name)
       req(len(rec['logits'])==(33 if forced else 1),tag+' logits count')
       ids=np.frombuffer(z['processor_input_ids'].tobytes(),dtype=np.int64);att=np.frombuffer(z['processor_attention_mask'].tobytes(),dtype=np.int64)
       grid=np.frombuffer(z['processor_image_grid_thw'].tobytes(),dtype=np.int64);req(len(grid)==3 and np.all(att==1),tag+' processor geometry')
       t,h,w=map(int,grid);req(h%merge==w%merge==0,tag+' merge geometry');h//=merge;w//=merge
       image_positions=np.flatnonzero(ids==mc['image_token_id']);req(t*h*w==1024 and len(image_positions)==1024,tag+' visual count')
       start=int(image_positions[0]);req(np.array_equal(image_positions,np.arange(start,start+1024)) and rec['image_start']==start and rec['original_length']==len(ids),tag+' contiguous visual')
       coords=np.stack([np.repeat(np.arange(t),h*w),np.tile(np.repeat(np.arange(h),w),t),np.tile(np.arange(w),t*h)])+start
       full=np.empty((3,len(ids)),dtype=np.int64);full[:,:start]=np.arange(start);full[:,start:start+1024]=coords
       suffix=len(ids)-start-1024;full[:,start+1024:]=np.arange(start+max(t,h,w),start+max(t,h,w)+suffix)
       original=np.asarray(rec['original_position_ids'],dtype=np.int64);req(np.array_equal(original[:,0,:],full),tag+' independent original mRoPE')
       ordered=rec['ordered_ids'];spatial=rec['spatial_ids'];req(len(ordered)==len(set(ordered))==len(spatial)==256 and sorted(ordered)==spatial and all(0<=x<1024 for x in ordered),tag+' selection')
       retained=list(range(start))+[start+x for x in spatial]+list(range(start+1024,len(ids)))
       req(rec['retained_ids']==retained,tag+' text/visual positions retained')
       selected=z['visual_main'][np.asarray(spatial,dtype=np.int64)];req(np.array_equal(selected,z['pruned_main']),tag+' main selection')
       pre=z['prefill_inputs_embeds'][0];req(np.array_equal(pre[start:start+256],selected),tag+' main prefill')
       if text_cache is None:
        left=embedding[torch.from_numpy(ids[:start].copy())].contiguous().view(torch.uint8).numpy()
        right=embedding[torch.from_numpy(ids[start+1024:].copy())].contiguous().view(torch.uint8).numpy();text_cache=(left,right)
       req(np.array_equal(pre[:start],text_cache[0]) and np.array_equal(pre[start+256:],text_cache[1]),tag+' nonvisual embeddings')
       mask=np.frombuffer(z['prefill_attention_mask'].tobytes(),dtype=np.int64);req(len(mask)==len(retained) and np.all(mask==1),tag+' attention mask')
       full_delta=int(full.max())+1-len(ids);req(rec['full_rope_delta']==[[full_delta]],tag+' full delta')
       expect_pos=np.broadcast_to(np.arange(len(retained)),(3,len(retained))) if arm.startswith('P1') else full[:,retained]
       actual_pos=np.frombuffer(z['prefill_position_ids'].tobytes(),dtype=np.int64).reshape(3,-1)
       req(np.array_equal(actual_pos,expect_pos) and rec['prefill_position_ids']==expect_pos[:,None,:].tolist(),tag+' prefill positions')
       delta=0 if arm.startswith('P1') else full_delta+768;req(rec['rope_delta']==[[delta]],tag+' rope delta')
       cp=np.frombuffer(z['prefill_cache_position'].tobytes(),dtype=np.int64);req(np.array_equal(cp,np.arange(len(retained))),tag+' prefill cache position')
       for j in range(3):
        ds=z['visual_ds_'+str(j)][np.asarray(spatial,dtype=np.int64)];req(np.array_equal(ds,z['selected_ds_'+str(j)]),tag+' selected DS '+str(j))
        if arm.endswith('D1'):req(np.array_equal(ds,z['prefill_ds_'+str(j)]),tag+' injected DS '+str(j))
        else:req('prefill_ds_'+str(j) not in z.files,tag+' omitted DS '+str(j))
       if arm.endswith('D1'):
        actual_mask=np.frombuffer(z['prefill_visual_pos_masks'].tobytes(),dtype=np.bool_);desired=np.zeros(len(retained),dtype=np.bool_);desired[start:start+256]=True
        req(np.array_equal(actual_mask,desired) and rec['injected_mask']==[desired.tolist()] and len(rec['injected_deepstack'])==3,tag+' visual DS mask')
       else:req('prefill_visual_pos_masks' not in z.files and rec['injected_mask'] is None and rec['injected_deepstack'] is None,tag+' omitted mask')
       if arm=='P1-D0' and row['task']=='DocVQA_VAL' and (str(row['qid']),forced) in legacy_lookup:
        old=legacy_lookup[(str(row['qid']),forced)];op=Path(old['array_path'])
        same={'processor':old['processor']==rec['processor'],'ordered_ids':old['ordered_ids']==ordered,
              'prefill_embed':old['llm'][0]['inputs_embeds']==rec['prefill_embed_sha256'],
              'prefill_attention':old['llm'][0]['attention_mask']==rec['attention_mask_sha256'],
              'position':old['llm'][0]['position_ids']==rec['prefill_position_ids'] and old['llm'][0]['rope_delta']==rec['rope_delta'],
              'output':old['output_ids'][0]==output and old['answer']==rec['answer'],'logits':old['logits']==rec['logits'],
              'old_archive_hash':sha(op)==old['array_sha256']}
        with np.load(op,allow_pickle=False) as oldz:
         common=('processor_input_ids','processor_attention_mask','processor_pixel_values','processor_image_grid_thw',
                 'visual_main','visual_ds_0','visual_ds_1','visual_ds_2','pruned_main','prefill_inputs_embeds',
                 'prefill_attention_mask','prefill_position_ids','prefill_cache_position')
         same['archive_inputs']=all(np.array_equal(z[k],oldz[k]) for k in common)
         same['archive_logits']=all(np.array_equal(z['logits_'+str(j)],oldz['logits_'+str(j)]) for j in range(33 if forced else 1))
        legacy_endpoints.append({'qid':str(row['qid']),'forced':forced,'identical':all(same.values()),'checks':same})
       signatures=[tensor_sha(z[x]) for x in ('processor_input_ids','processor_attention_mask','processor_pixel_values','processor_image_grid_thw','visual_main','visual_ds_0','visual_ds_1','visual_ds_2','pruned_main','prefill_inputs_embeds','prefill_attention_mask')]+[str(ordered),str(spatial),str(retained)]
       if baseline is None:baseline=signatures
       else:req(signatures==baseline,tag+' same intervened input')
       if forced:
        trace=rec['decode_trace'];req(len(trace)==33,tag+' 33 forward calls')
        for step,entry in enumerate(trace):
         expected_cache=0 if step==0 else len(retained)+step-1;after=len(retained)+step
         req(entry['rope_delta']==[[delta]] and entry['layer_cache_lengths']==[expected_cache]*36 and entry['layer_cache_lengths_after']==[after]*36,tag+' per-layer cache '+str(step))
         if step==0:req(entry['cache_position']==list(range(len(retained))) and entry['position_ids']==expect_pos[:,None,:].tolist(),tag+' prefill trace')
         else:req(entry['cache_position']==[expected_cache] and entry['position_ids']==[[[expected_cache+delta]]]*3,tag+' decode position '+str(step))
      baseline_by_condition[forced]=baseline
     if fi==1:req(baseline_by_condition[False]==baseline_by_condition[True],'natural/forced same prefill '+str(row['qid']))
    if len(errors)==sample_errors:verified+=1
  req(not torch.cuda.is_initialized(),'CPU audit did not initialize CUDA')
  # Smoke costs include observation and array capture, not controlled performance timings.
  task_means={}
  for task,_ in TASKS:
   vals=[x['through_archive_seconds'] for x in timings if x['task']==task and not x['forced']]
   task_means[task]=sum(vals)/len(vals) if vals else None
  e1_hours=sum(task_means[task]*128*4 for task,_ in TASKS)/3600 if all(v is not None for v in task_means.values()) else None
  arm_stats={}
  for task,_ in TASKS:
   for arm in ARMS:
    vals=[x['through_archive_seconds'] for x in timings if x['task']==task and x['arm']==arm and not x['forced']]
    arm_stats[task+'|'+arm]={'n':len(vals),'mean_seconds':statistics.mean(vals),'median_seconds':statistics.median(vals),'min_seconds':min(vals),'max_seconds':max(vals)} if vals else None
  observed_natural=sum(x['through_archive_seconds'] for x in timings if not x['forced'])
  observed_forced=sum(x['through_archive_seconds'] for x in timings if x['forced'])
  duration=ex['finished_unix']-launch['started_unix']
  # The other verifier runs independently; bind its completed proof only after our full audit.
  deadline=time.monotonic()+1200
  while not (a.proof.is_file() and a.verifier_terminal.is_file()) and time.monotonic()<deadline:
   time.sleep(10)
  proof=read(a.proof);verify_terminal=read(a.verifier_terminal)
  req(proof.get('status')=='passed' and proof.get('n_verified')==32 and not proof.get('errors'),'smoke verifier proof')
  req(verify_terminal.get('reason')=='completed' and verify_terminal.get('worker_returncode')==0 and not verify_terminal.get('remaining_process_group'),'smoke verifier terminal')
 except Exception as exc:
  import traceback;traceback.print_exc();errors.append('review exception: '+repr(exc));task_means={};e1_hours=None;duration=None
 report={'status':'passed' if not errors and verified==32 else 'failed','n_expected':32,'n_verified':verified,'checks':checks,'errors':errors,
  'legacy_endpoint_comparisons':legacy_endpoints,'legacy_endpoint_all_identical':bool(legacy_endpoints) and all(x['identical'] for x in legacy_endpoints),
  'legacy_reuse_scope':'cross-run identities are descriptive; E1 arms were newly generated, and any mismatch prohibits reusing that prior endpoint',
  'natural_outputs':len(ends),'natural_eos':sum(x['eos'] for x in ends),'natural_truncated':sum(x['truncated'] for x in ends),
  'task_natural_through_archive_mean_seconds':task_means,'task_arm_natural_through_archive_stats':arm_stats if 'arm_stats' in locals() else {},
  'observed_natural_through_archive_seconds':observed_natural if 'observed_natural' in locals() else None,
  'observed_forced_through_archive_seconds':observed_forced if 'observed_forced' in locals() else None,
  'estimated_384_e1_panel_gpu_hours':e1_hours,'observed_launcher_seconds':duration,
  'budget_scope':'E1 estimate only, uses 32-image observed natural generation with diagnostic copies/archive; not TTFT, E4/E2/E3 budgets or a guarantee',
  'run':str(a.run.resolve()),'runner_sha256':frozen.get('runner_sha256') if 'frozen' in locals() else None,
  'source_snapshot':str(E1/(a.run.name+'_sources')),
  'frozen_sha256':safe_sha(a.run/'frozen.json'),'raw_predictions_sha256':safe_sha(a.run/'raw_predictions.jsonl'),
  'generated_arms_sha256':safe_sha(a.run/'generated_arms.jsonl'),'validated_quartets_sha256':safe_sha(a.run/'validated_quartets.jsonl'),
  'worker_complete_sha256':safe_sha(a.run/'worker_complete.json'),'launch_sha256':safe_sha(E1/(a.run.name+'_launch.json')),
  'container_exit_sha256':safe_sha(E1/(a.run.name+'_exit.json')),'supervisor_terminal_sha256':safe_sha(E1/('monitor'+a.run.name)/'terminal.json'),
  'run_proof_sha256':safe_sha(a.proof),'verifier_terminal_sha256':safe_sha(a.verifier_terminal),
  'reviewer_sha256':sha(__file__),'start_unix':started,'end_unix':time.time(),'scope':'read-only CPU, no model forward, GT, scoring or CUDA'}
 with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(report,ensure_ascii=False));return not errors
if __name__=='__main__':raise SystemExit(0 if main() else 1)
