"""Prepared, review-gated, single DEV PriorTR identity smoke. Never scores."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from admission import BASE,E1,admit,sha
HERE=Path(__file__).resolve().parent; INSTALL=HERE.parent/'install001'; PLAN=HERE/'smoke_single_plan002.json'

def admit_review(path):
 proof=json.loads(Path(path).read_text())
 if proof.get('status')!='passed' or proof.get('independent_review') is not True:raise ValueError('independent source review not passed')
 cpu=HERE/'cpu_validation004.json';ct=HERE/'monitor_cpu_validation004/terminal.json';validation=json.loads(cpu.read_text());terminal=json.loads(ct.read_text())
 if (proof.get('cpu_proof_sha256')!=sha(cpu) or proof.get('cpu_terminal_sha256')!=sha(ct) or validation.get('status')!='passed' or validation.get('returncode')!=0 or terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[]):raise ValueError('CPU test proof/terminal not passed')
 started=json.loads((HERE/'monitor_cpu_validation004/started.json').read_text());command=started.get('command',[]);cwd=Path(started.get('cwd','/'))
 if terminal.get('worker_pid')!=started.get('worker_pid') or [(cwd/x).resolve() for x in command if Path(x).name=='cpu_validation.py']!=[(HERE/'cpu_validation.py').resolve()]:raise ValueError('CPU test supervisor did not run this verifier')
 if validation.get('stdout_sha256')!=sha(HERE/'cpu_validation004.stdout') or validation.get('stderr_sha256')!=sha(HERE/'cpu_validation004.stderr'):raise ValueError('CPU test logs changed')
 required=('admission.py','preprocess.py','trace.py','smoke_worker.py','smoke_launch.py','smoke_verify.py','test_trace.py','test_env_contract.py','test_launch.py','env_contract.py','cpu_validation.py')
 for name in required:
  if proof.get('source_sha256',{}).get(name)!=sha(HERE/name) or validation.get('source_sha256_before',{}).get(name)!=sha(HERE/name) or validation.get('source_sha256_after',{}).get(name)!=sha(HERE/name):raise ValueError('review/source mismatch: '+name)
 if proof.get('plan_sha256')!=sha(PLAN) or validation.get('plan_sha256')!=sha(PLAN) or proof.get('processor_sha256')!=sha(HERE/'processor_check002.json') or validation.get('processor_proof_sha256')!=sha(HERE/'processor_check002.json'):raise ValueError('review proof inputs changed')
 from env_contract import verify_env
 fingerprint=verify_env()
 if proof.get('env_fingerprint_sha256')!=fingerprint or validation.get('env_fingerprint_sha256')!=fingerprint:raise ValueError('reviewed environment fingerprint mismatch')
 terminal=json.loads((INSTALL/'monitor_fingerprint002/terminal.json').read_text())
 if terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[]:raise ValueError('runtime fingerprint supervisor not complete')
 provenance=INSTALL/'provenance.json'
 if proof.get('environment_provenance_sha256')!=sha(provenance) or validation.get('environment_provenance_sha256')!=sha(provenance):raise ValueError('independent environment proof changed')
 bundle=json.loads(provenance.read_text())
 if bundle.get('priortr_commit')!='e442a4c14af1b333067a6b8310c9111cad922033' or bundle.get('transformers_commit')!='f8f2834e1a82970f93e06f9bb076a361ca2d70c0':raise ValueError('wrong fixed sources')
 for name,digest in bundle['original_source_sha256'].items():
  if sha(INSTALL/'source'/name)!=digest:raise ValueError('isolated priortr source changed: '+name)
 return {'path':str(Path(path).resolve()),'sha256':sha(path)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--review-proof',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if not a.execute:raise ValueError('disabled without explicit execute')
 review=admit_review(a.review_proof);proof,rows=admit();plan=json.loads(PLAN.read_text())
 if plan['selected']!=rows[0] or plan['cpu_processor_proof_sha256']!=sha(HERE/'processor_check002.json'):raise ValueError('frozen sample differs')
 check=json.loads((HERE/'processor_check002.json').read_text());term=json.loads((HERE/'monitor_processor002/terminal.json').read_text())
 if check.get('status')!='passed' or check.get('n')!=8 or term.get('worker_returncode')!=0 or term.get('remaining_process_group')!=[]:raise ValueError('processor gate failed')
 weight_path=BASE.parents[1]/'20261028/qwen/qwen_weight_identity.json';weights=json.loads(weight_path.read_text());stats={}
 for item in weights['files']:
  path=Path(item['path']);st=path.stat()
  if st.st_size!=item['bytes'] or sha(path)!=item['expected_lfs_sha256']:raise ValueError('weight mismatch')
  stats[str(path)]=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
 out=a.out.resolve()
 if not out.is_dir() or any((out/x).exists() for x in ('frozen.json','generated.jsonl','records.jsonl','arrays')):raise ValueError('output invalid')
 (out/'arrays').mkdir();sources={name:sha(HERE/name) for name in ('admission.py','preprocess.py','trace.py','smoke_worker.py','smoke_launch.py','smoke_verify.py','test_trace.py','test_env_contract.py','test_launch.py','env_contract.py','cpu_validation.py')}
 frozen={'sample_plan_sha256':sha(PLAN),'review':review,'processor_proof_sha256':sha(HERE/'processor_check002.json'),'e1_proof_sha256':sha(E1/'e1full001_completion.json'),'weights_identity_sha256':sha(weight_path),'env_fingerprint_sha256':sha(INSTALL/'env_fingerprint.json'),'environment_provenance_sha256':sha(INSTALL/'provenance.json'),'weight_stats_before':stats,'source_sha256':sources,'gt_read':False,'scoring':False,'instrumented_timing':True}
 with (out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 import torch,numpy as np,transformers,visual_token_pruning
 import transformers.models.qwen3_vl.modeling_qwen3_vl as actual_qwen
 if Path(sys.prefix).resolve()!=(INSTALL/'env').resolve() or torch.__version__!='2.10.0+cu128' or transformers.__version__!='5.2.0.dev0' or Path(visual_token_pruning.__file__).resolve()!=(INSTALL/'source/visual_token_pruning/__init__.py').resolve() or Path(actual_qwen.__file__).resolve()!=(INSTALL/'source/qwen3/modeling_qwen3_vl.py').resolve():raise ValueError('wrong runtime environment')
 from transformers import AutoProcessor
 from transformers.generation.logits_process import LogitsProcessor,LogitsProcessorList
 from visual_token_pruning.config import VTRConfig
 from visual_token_pruning.model.vtr_qwen3_vl import VTRQwen3VLForConditionalGeneration
 from preprocess import prepare
 from trace import Trace
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 recipe=json.loads((BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json').read_text())['recipe'];processor=AutoProcessor.from_pretrained(recipe['model_path'],local_files_only=True,trust_remote_code=False)
 reference=[r for r in json.loads((HERE/'processor_check002.json').read_text())['checks'] if r['task']==rows[0]['task'] and str(r['qid'])==str(rows[0]['qid'])]
 if len(reference)!=1 or not all(reference[0]['matches'].values()):raise ValueError('selected processor not verified')
 model=VTRQwen3VLForConditionalGeneration.from_pretrained(recipe['model_path'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
 if model.dtype!=torch.bfloat16 or model.config._attn_implementation!='sdpa':raise ValueError('unexpected dtype or attention backend')
 model.vtr_config=VTRConfig(enabled=False);model.vtr_strategy=model._create_strategy('priortr')
 # Two stock cases run before the one-time in-place public class swap.
 patched=False
 class Forced(LogitsProcessor):
  def __init__(self,idx):self.idx=idx
  def __call__(self,input_ids,scores):scores.fill_(-float('inf'));scores[:,self.idx]=0;return scores
 force=processor.tokenizer.encode(' the',add_special_tokens=False)[0]
 for case in plan['cases']:
  if case['name'] not in ('stock','stock_forced') and not patched:
   model._replace_text_model();patched=True
  enabled=case['enabled'];cfg={'enabled':enabled,'strategy':'priortr','prune_layer':3}
  for key in ('keep_ratio','keep_tokens'):
   if key in case:cfg[key]=case[key]
  model.vtr_config=VTRConfig(**cfg);model.vtr_strategy=model._create_strategy('priortr')
  inputs=prepare(processor,rows[0]['message']).to('cuda').to(torch.bfloat16);original_length=inputs['input_ids'].shape[1]
  def met(t):
   c=t.detach().contiguous().cpu();return {'dtype':str(c.dtype),'shape':list(c.shape),'sha256':hashlib.sha256(c.view(torch.uint8).numpy().tobytes()).hexdigest()}
  if {k:met(inputs[k]) for k in reference[0]['actual']}!=reference[0]['actual']:raise ValueError('processor changed before generation')
  forced=case['generation']=='forced_33';max_new=33 if forced else 2048
  kwargs={'do_sample':False,'max_new_tokens':max_new,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0,'use_cache':True}
  if forced:kwargs.update(min_new_tokens=33,logits_processor=LogitsProcessorList([Forced(force)]))
  with torch.inference_mode(),Trace(model,forced=forced) as trace:
   generated=model.generate(**inputs,**kwargs)
   ids=generated[0,original_length:].detach().cpu().tolist();raw={'case':case['name'],'ids':ids,'forced':forced,'time':time.time()}
   with (out/'generated.jsonl').open('a') as f:f.write(json.dumps(raw,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
  for key in ('input_ids','attention_mask','pixel_values','image_grid_thw'):trace.save('processor_'+key,inputs[key])
  arrays={k:v.view(torch.uint8).numpy() for k,v in trace.arrays.items()};archive=out/'arrays'/(case['name']+'.npz')
  with archive.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
  eos_ids=model.generation_config.eos_token_id
  eos_ids=[eos_ids] if isinstance(eos_ids,int) else list(eos_ids or [])
  rec={'case':case['name'],'ids':ids,'eos_ids':eos_ids,'eos':bool(ids and ids[-1] in eos_ids),'truncated':bool(not forced and len(ids)>=2048 and (not ids or ids[-1] not in eos_ids)),'decoded':processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),'forced':forced,'force_token':force if forced else None,'generated_kwargs':{k:kwargs[k] for k in ('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty','use_cache')},'forced_processor':{'class':type(kwargs['logits_processor'][0]).__name__,'token_id':force,'min_new_tokens':kwargs['min_new_tokens']} if forced else None,'calls':trace.calls,'layers':trace.layers,'injections':trace.injections,'prunes':trace.prunes,'arrays':{'path':str(archive),'sha256':sha(archive),'tensors':{k:met(v) for k,v in trace.arrays.items()}},'rope_delta':None if model.model.rope_deltas is None else model.model.rope_deltas.detach().cpu().tolist(),'rope_offset':model._rope_position_offset}
  with (out/'records.jsonl').open('a') as f:f.write(json.dumps(rec,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
 for path,before in stats.items():
  st=Path(path).stat()
  if [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]!=before:raise ValueError('weight changed')
 if any(sha(HERE/name)!=h for name,h in sources.items()):raise ValueError('source changed')
 admit_review(a.review_proof)
 with (out/'worker_complete.json').open('x') as f:json.dump({'cases':len(plan['cases']),'completed_unix':time.time()},f);f.write('\n')
if __name__=='__main__':main()
