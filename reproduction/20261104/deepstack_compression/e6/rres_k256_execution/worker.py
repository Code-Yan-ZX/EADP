"""Three-anchor stock 512px model diagnostic; no quality scores or timings."""
import argparse,hashlib,importlib.util,json,os,socket,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];P=B/'e6/rres_k256_pilot_v2';ROOT=B.parents[2]
def atomic(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def event(path,value):
 with Path(path).open('a') as f:json.dump(value,f);f.write('\n');f.flush();os.fsync(f.fileno())
def checked_generation(model,reference):
 original=model.generate
 def generate(*args,**kwargs):
  if {k:kwargs.get(k) for k in reference['generation_scalars']}!=reference['generation_scalars']:raise ValueError('E1 generation scalar mismatch')
  eos=kwargs.get('eos_token_id',model.generation_config.eos_token_id);eos=[eos] if isinstance(eos,int) else list(eos or [])
  if eos!=reference['eos_ids']:raise ValueError('E1 EOS mismatch')
  return original(*args,**kwargs)
 return generate
def import_gate():
 # A full GPU gate will be pinned before any execution proof is frozen.
 path=D/'gpu_gate.py';expected='75e5966eb593eb2e6f0c156975c4850aba82a770cf7c99fb568bd87e1e3128ae'
 if hashlib.sha256(path.read_bytes()).hexdigest()!=expected or '_rres_gpu_gate' in sys.modules:raise ValueError('GPU gate source')
 spec=importlib.util.spec_from_file_location('_rres_gpu_gate',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve()
 if out!=D/'run001/observations' or not out.is_dir():raise ValueError('unplanned run directory')
 gate=import_gate();approval=gate.admit_gpu();deadline=time.monotonic()+60
 while True:
  try:launch=gate.read(D/'run001/launch.json');identity=gate.read(D/'run001/container_identity.json');break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>deadline:raise TimeoutError('container identity publication')
   time.sleep(.2)
 cid=identity['container_id']
 if len(cid)!=64 or socket.gethostname()!=cid[:12] or cid not in Path('/proc/self/cgroup').read_text() or launch['approval']!=approval:raise ValueError('runtime container identity')
 from transformers import AutoConfig
 import torch
 sys.path[:0]=[str(P),str(B/'e4_c_latency'),str(B/'e4'),str(ROOT/'_reproduction/author_eadp/Qwen_vl'),str(ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit')]
 import arm,decode_observer,resolution,vlmeval.vlm.qwen3_vl.model_fixed_res as module
 from input_preflight import load_accepted_e1_references
 if torch.cuda.is_initialized() or module.QWEN3_FIXED_RESOLUTION!=1024:raise ValueError('unexpected initial runtime state')
 refs=gate.read(P/'run_preflight001/processor_probe.json');anchors=refs['anchors'];assert len(anchors)==3
 rows,_=load_accepted_e1_references(B/'e1/e1full001_completion.json',B/'e1/monitorverifye1full001/started.json',B/'e1/monitorverifye1full001/terminal.json');rows={(r['task'],str(r['qid'])):r for r in rows}
 completion=gate.read(B/'e1/e1full001_completion.json');journal=completion['files']['journal'];reference_path=Path(journal['path'])
 if gate.sha(reference_path)!=journal['sha256']:raise ValueError('accepted E1 journal hash')
 controls={}
 with reference_path.open() as f:
  for line in f:
   item=json.loads(line)
   if item.get('arm')=='P3-D1' and item.get('forced') is False:
    key=(item['task'],str(item['qid']))
    if key in controls:raise ValueError('duplicate E1 control')
    controls[key]={'generation_scalars':item['generation_scalars'],'eos_ids':item['eos_ids']}
 if len(controls)!=384:raise ValueError('incomplete E1 controls')
 recipe=gate.read(ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')['recipe'];recipe={k:v for k,v in recipe.items() if k not in ('visual_token_num','alpha','beta')}
 started=[];recorded=[];events=out/'progress.jsonl'
 try:
  wrapper=module.Qwen3VLChatFixedRes(**recipe)
  if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('model dtype/attention backend')
  for ref in anchors:
   key=(ref['task'],str(ref['qid']));row=rows[key]
   if any(str(row[k])!=str(ref[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) or gate.sha(row['image_path'])!=ref['image_sha256']:raise ValueError('input identity')
   name=key[0]+'_'+key[1];event(events,{'event':'arm_started','task':key[0],'qid':key[1]});started.append(name)
   original=wrapper.model.generate;had='generate' in wrapper.model.__dict__;saved=wrapper.model.__dict__.get('generate');wrapper.model.generate=checked_generation(wrapper.model,controls[key])
   try:record=arm.run_arm(torch,wrapper,module,row,ref,out/(name+'_generated.json'),decode_observer)
   finally:
    if had:wrapper.model.generate=saved
    else:delattr(wrapper.model,'generate')
   if wrapper.model.generate!=original:raise ValueError('generation method not restored')
   record['generation_scalars']=controls[key]['generation_scalars'];atomic(out/(name+'.json'),record)
   event(events,{'event':'arm_recorded','task':key[0],'qid':key[1],'status':'completed'});recorded.append(name)
  if module.QWEN3_FIXED_RESOLUTION!=1024 or len(started)!=3 or len(recorded)!=3:raise ValueError('resolution or denominator')
  atomic(out/'worker_complete.json',{'status':'completed','expected':3,'started':started,'recorded':recorded,'GT_opened':False,'scoring_performed':False,'existing_DEV_predictions_opened':True,'timing_claims':False,'resolution_restored':True,'errors':[]})
 except BaseException as exc:
  atomic(out/'worker_failure.json',{'status':'failed','started':started,'recorded':recorded,'failure_type':type(exc).__name__,'failure':str(exc),'traceback':traceback.format_exc(),'GT_opened':False,'scoring_performed':False});raise
if __name__=='__main__':main()
