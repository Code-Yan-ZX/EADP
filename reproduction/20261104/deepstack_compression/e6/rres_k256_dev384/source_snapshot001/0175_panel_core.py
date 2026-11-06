"""Runtime panel loop; callers provide admitted model, inputs and actual arm."""
import hashlib,json,os,traceback
from pathlib import Path
from design import panel
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def write(path,value):
 with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
def event(path,value):
 with Path(path).open('a') as f:json.dump(value,f,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
def key(row):return (row['task'],str(row['qid']))
RECIPE={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0}
def checked_generation(model,reference):
 scalars=reference.get('generation_scalars');expected_eos=[151645,151643]
 if not isinstance(scalars,dict) or set(scalars)!=set(RECIPE) or any(type(scalars[k]) is not type(v) or scalars[k]!=v for k,v in RECIPE.items()) or reference.get('eos_ids')!=expected_eos or any(type(x) is not int for x in reference['eos_ids']):raise ValueError('E1 generation reference')
 original=model.generate
 def generate(*args,**kwargs):
  if {k:kwargs.get(k) for k in reference['generation_scalars']}!=reference['generation_scalars']:raise ValueError('generation scalar mismatch')
  eos=kwargs.get('eos_token_id',model.generation_config.eos_token_id);eos=[eos] if type(eos) is int else list(eos or [])
  if eos!=reference['eos_ids']:raise ValueError('EOS mismatch')
  return original(*args,**kwargs)
 return generate
def run(rows,anchors,low_refs,controls,torch,wrapper,module,arm,decode,out,verify_image=sha):
 anchors=panel(anchors);out=Path(out)
 if not out.is_dir() or len(rows)!=384 or len(low_refs)!=384 or len(controls)!=384:raise ValueError('incomplete panel')
 index={key(r):r for r in rows};refs={key(r):r for r in low_refs};expected=[key(a) for a in anchors]
 if len(index)!=384 or len(refs)!=384 or set(index)!=set(expected) or set(refs)!=set(expected) or set(controls)!=set(expected):raise ValueError('panel identity inventory')
 started=[];recorded=[];journal=out/'progress.jsonl'
 try:
  for a in anchors:
   k=key(a);r=index[k];ref=refs[k];c=controls[k];name=k[0]+'_'+k[1]
   if any(str(r.get(x))!=str(a[x]) or str(ref.get(x))!=str(a[x]) for x in ('task','qid','group_id','image_sha256','message_sha256')) or ref.get('low_visual_tokens')!=256:raise ValueError('input identity')
   event(journal,{'event':'started','task':k[0],'qid':k[1],'group_id':a['group_id']});started.append(name)
   if verify_image(r['image_path'])!=a['image_sha256']:raise ValueError('image identity')
   inner=wrapper.model.model
   if not hasattr(inner,'rope_deltas'):raise ValueError('missing rope state')
   inner.rope_deltas=None
   original=wrapper.model.generate;had='generate' in wrapper.model.__dict__;saved=wrapper.model.__dict__.get('generate');wrapper.model.generate=checked_generation(wrapper.model,c)
   try:result=arm.run_arm(torch,wrapper,module,r,ref,out/(name+'_generated.json'),decode)
   finally:
    if had:wrapper.model.generate=saved
    else:delattr(wrapper.model,'generate')
   if wrapper.model.generate!=original or result.get('visual_tokens')!=256 or result.get('hooks_and_methods_restored') is not True or result.get('processor')!=ref['low_processor'] or not (out/(name+'_generated.json')).is_file():raise ValueError('arm identity, raw output or hook restoration')
   if any(str(result.get(x))!=str(a[x]) for x in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('result identity')
   result.update(status='completed',generation_scalars=c['generation_scalars'],rope_reset_before_generation=True);write(out/(name+'.json'),result)
   event(journal,{'event':'recorded','task':k[0],'qid':k[1],'group_id':a['group_id'],'status':'completed'});recorded.append(name)
  if len(started)!=384 or len(recorded)!=384 or module.QWEN3_FIXED_RESOLUTION!=1024:raise ValueError('incomplete panel or resolution not restored')
  return {'status':'completed','expected':384,'started':started,'recorded':recorded,'GT_opened':False,'scoring_performed':False,'existing_DEV_predictions_opened':True,'timing_claims':False,'resolution_restored':True,'errors':[]}
 except BaseException as exc:
  write(out/'worker_failure.json',{'status':'failed','expected':384,'started':started,'recorded':recorded,'failure_type':type(exc).__name__,'failure':str(exc),'traceback':traceback.format_exc(),'GT_opened':False,'scoring_performed':False});raise
