"""Unadmitted GPU identity continuation. No runnable entrypoint.

The caller must provide a separately reviewed launch/identity/cleanup gate,
freeze this file, and supervise the retained Docker before calling this code.
This is an instrumented endpoint diagnostic, never a latency measurement.
"""
import json,os,time,types
from contextlib import contextmanager
from pathlib import Path

def append_journal(path,record):
 with Path(path).open('a') as f:
  f.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

def write_complete(path,record):
 with Path(path).open('x') as f:
  json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def force_code(runner):
 """Find the one frozen runner implementation of ForceToken.__call__."""
 found=[]
 def visit(code):
  if code.co_name=='__call__' and 'forced_token' in code.co_freevars:found.append(code)
  for item in code.co_consts:
   if isinstance(item,types.CodeType):visit(item)
 visit(runner.__code__)
 if len(found)!=1:raise ValueError('ambiguous frozen ForceToken implementation')
 return found[0]

@contextmanager
def admitted_generate(wrapper,panel,forced,token=None,processor_code=None):
 """Validate actual kwargs at the last boundary before model.generate."""
 original=wrapper.model.generate
 names=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')
 expected=dict(panel['generation_scalars'])
 if set(expected)!=set(names) or expected['max_new_tokens']!=2048:raise ValueError('invalid accepted generation recipe')
 if forced:expected['max_new_tokens']=33
 def checked(*args,**kwargs):
  if {name:kwargs.get(name) for name in names}!=expected:raise ValueError('actual generation recipe mismatch before generate')
  eos=kwargs.get('eos_token_id',wrapper.model.generation_config.eos_token_id)
  eos=[eos] if isinstance(eos,int) else list(eos or [])
  if eos!=([] if forced else panel['eos_ids']):raise ValueError('actual EOS mismatch before generate')
  if forced:
   processors=kwargs.get('logits_processor')
   if kwargs.get('min_new_tokens')!=33 or processors is None or len(processors)!=1:raise ValueError('actual forced control mismatch before generate')
   fn=getattr(type(processors[0]),'__call__',None);code=getattr(fn,'__code__',None);closure=getattr(fn,'__closure__',None)
   if type(token) is not int or processor_code is None or code is not processor_code or not closure:raise ValueError('actual forced token/processor mismatch before generate')
   captured=dict(zip(code.co_freevars,(x.cell_contents for x in closure))).get('forced_token')
   if type(captured) is not int or captured!=token:raise ValueError('actual forced token/processor mismatch before generate')
  return original(*args,**kwargs)
 wrapper.model.generate=checked
 try:yield
 finally:wrapper.model.generate=original

def run_after_future_admission(torch,wrapper,native,refs,out,generate,forced_token,
                               fresh_runner,light_runner,cache_fingerprints,core,compare,sha):
 """Use only after future reviewed admission; GPU scheduling belongs to root."""
 out=Path(out)
 fixed=wrapper.processor.tokenizer.encode(' the',add_special_tokens=False)
 if not fixed or type(forced_token) is not int or int(fixed[0])!=forced_token:raise ValueError('forced token differs from fixed tokenizer')
 if not out.is_dir() or any((out/name).exists() for name in ('journal.jsonl','arrays','worker_complete.json')):raise ValueError('reused output')
 (out/'arrays').mkdir();journal=out/'journal.jsonl'
 def emit(record):append_journal(journal,record)
 def archive_factory(job,side):
  import numpy as np
  name=f"{job['task']}_{job['qid']}_k{job['budget']}_{'forced' if job['forced'] else 'natural'}_{side}.npz"
  def archive(tensors):
   dst=out/'arrays'/name
   data={k:v.detach().contiguous().cpu().view(torch.uint8).numpy() for k,v in tensors.items()}
   with dst.open('xb') as f:np.savez(f,**data);f.flush();os.fsync(f.fileno())
   return {'kind':'full_arrays','path':str(dst),'sha256':sha(dst)}
  return archive
 def fresh(row,budget,token,raw,archive):
  wrapper.visual_token_num=budget;wrapper.pruner.visual_token_num=budget
  panel=refs['records'][(row['task'],str(row['qid']),budget)]
  with admitted_generate(wrapper,panel,token is not None,token,force_code(fresh_runner) if token is not None else None),cache_fingerprints(wrapper,token is not None) as kv:
   result=fresh_runner(torch,wrapper,native,row,budget,True,generate,raw,archive,token,capture_prefill_logits=True)
  result['cache_fingerprints']=kv.fingerprints
  return result
 def light(row,budget,token,raw,archive):
  wrapper.visual_token_num=budget;wrapper.pruner.visual_token_num=budget
  panel=refs['records'][(row['task'],str(row['qid']),budget)]
  with admitted_generate(wrapper,panel,token is not None,token,force_code(light_runner) if token is not None else None):
   return light_runner(torch,wrapper,native,row,budget,generate,raw,archive,token)
 result=core.run(refs['rows'],refs['records'],forced_token,fresh,light,archive_factory,emit,compare.compare_metadata,compare.compare_archives)
 if result.get('generations')!=32 or result.get('natural_pairs')!=12 or result.get('forced_pairs')!=4:raise ValueError('incomplete diagnostic')
 write_complete(out/'worker_complete.json',dict(result,finished_unix=time.time(),GT_opened=False,scoring_performed=False,timing_claims=False))
 return result

if __name__=='__main__':raise SystemExit('GPU worker locked: no admitted launcher or independent prelaunch approval')
