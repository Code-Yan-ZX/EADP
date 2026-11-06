import importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('budget_worker_candidate_test',HERE/'worker_candidate.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

class Tests(unittest.TestCase):
 def test_actual_two_runner_force_implementations_resolve(self):
  e4=HERE.parent/'e4';sys.path.insert(0,str(e4))
  try:
   import budget_runner,budget_arm
   for runner in (budget_runner.run_arm,budget_arm.run_arm):
    code=worker.force_code(runner);self.assertEqual(code.co_name,'__call__');self.assertIn('forced_token',code.co_freevars)
  finally:sys.path.remove(str(e4))
 def test_real_journal_is_durable_before_later_failure(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);wrapper=types.SimpleNamespace(pruner=types.SimpleNamespace(),model=types.SimpleNamespace(generate=lambda **kw:None,generation_config=types.SimpleNamespace(eos_token_id=9)),processor=types.SimpleNamespace(tokenizer=types.SimpleNamespace(encode=lambda *_a,**_kw:[3])));seen=[]
   recipe={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1}
   class KV:
    def __init__(self,w,forced):self.fingerprints=['forced' if forced else 'natural']
    def __enter__(self):return self
    def __exit__(self,*args):pass
   def fresh(_torch,w,_native,_row,k,ds,_generate,raw,_archive,token,capture_prefill_logits=False):
    seen.append((w.visual_token_num,w.pruner.visual_token_num,k,ds,capture_prefill_logits));raw({'output_ids':[7]});raise ValueError('later failure')
   def run(_rows,_refs,_token,fresh,_light,_archive,emit,_compare,_arrays):
    fresh({'task':'D','qid':'1'},512,None,lambda x:emit({'event':'raw','output':x}),lambda _:None)
   with self.assertRaisesRegex(ValueError,'later failure'):
    worker.run_after_future_admission(None,wrapper,None,{'rows':[],'records':{('D','1',512):{'generation_scalars':recipe,'eos_ids':[9]}}},out,None,3,fresh,None,KV,types.SimpleNamespace(run=run),types.SimpleNamespace(compare_metadata=None,compare_archives=None),None)
   self.assertEqual(seen,[(512,512,512,True,True)])
   self.assertEqual(json.loads((out/'journal.jsonl').read_text())['output']['output_ids'],[7])
   self.assertFalse((out/'worker_complete.json').exists())
 def test_bad_recipe_and_eos_never_call_real_generate(self):
  calls=[];original=lambda **kw:calls.append(kw)
  w=types.SimpleNamespace(model=types.SimpleNamespace(generate=original,generation_config=types.SimpleNamespace(eos_token_id=9)))
  recipe={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1}
  panel={'generation_scalars':recipe,'eos_ids':[9]}
  with worker.admitted_generate(w,panel,False):
   for key in recipe:
    bad=dict(recipe);bad[key]='wrong'
    with self.assertRaisesRegex(ValueError,'before generate'):w.model.generate(**bad)
   with self.assertRaisesRegex(ValueError,'EOS mismatch'):w.model.generate(**recipe,eos_token_id=[10])
   self.assertEqual(calls,[])
   w.model.generate(**recipe,eos_token_id=[9])
  self.assertEqual(len(calls),1);self.assertIs(w.model.generate,original)
 def test_forced_recipe_requires_control_before_generate(self):
  calls=[];w=types.SimpleNamespace(model=types.SimpleNamespace(generate=lambda **kw:calls.append(kw),generation_config=types.SimpleNamespace(eos_token_id=9)))
  recipe={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1}
  def runner(forced_token):
   class ForceToken:
    def __call__(self,_ids,scores):return forced_token
   return ForceToken()
  code=worker.force_code(runner)
  with worker.admitted_generate(w,{'generation_scalars':recipe,'eos_ids':[9]},True,7,code):
   kw=dict(recipe,max_new_tokens=33,eos_token_id=[],min_new_tokens=33)
   with self.assertRaisesRegex(ValueError,'forced control'):w.model.generate(**kw)
   with self.assertRaisesRegex(ValueError,'forced token/processor'):w.model.generate(**kw,logits_processor=[runner(8)])
   Fake=type('ForceToken',(),{'__call__':lambda *_args:None})
   with self.assertRaisesRegex(ValueError,'forced token/processor'):w.model.generate(**kw,logits_processor=[Fake()])
   self.assertEqual(calls,[])
   w.model.generate(**kw,logits_processor=[runner(7)])
  self.assertEqual(len(calls),1)
 def test_runner_nesting_rejects_before_underlying_generate_for_both_arms(self):
  recipe={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1}
  for side in ('fresh','light'):
   for corrupted in recipe:
    with self.subTest(side=side,field=corrupted),tempfile.TemporaryDirectory() as d:
     calls=[];original=lambda **kw:calls.append(kw)
     model=types.SimpleNamespace(generate=original,generation_config=types.SimpleNamespace(eos_token_id=9));w=types.SimpleNamespace(model=model,pruner=types.SimpleNamespace(),processor=types.SimpleNamespace(tokenizer=types.SimpleNamespace(encode=lambda *_a,**_kw:[3])))
     class KV:
      def __init__(self,*_args):self.fingerprints=[]
      def __enter__(self):return self
      def __exit__(self,*_args):pass
     def nested(*_args,**_kwargs):
      previous=model.generate
      def capture(**kwargs):return previous(**kwargs)
      model.generate=capture
      try:
       bad=dict(recipe);bad[corrupted]='invalid';return model.generate(**bad,eos_token_id=[9])
      finally:model.generate=previous
     def run(_rows,_refs,_token,fresh,light,_archive,_emit,_compare,_arrays):
      return (fresh if side=='fresh' else light)({'task':'D','qid':'1'},256,None,lambda _:None,lambda _:None)
     refs={'rows':[],'records':{('D','1',256):{'generation_scalars':recipe,'eos_ids':[9]}}}
     with self.assertRaisesRegex(ValueError,'recipe mismatch before generate'):
      worker.run_after_future_admission(None,w,None,refs,Path(d),None,3,nested,nested,KV,types.SimpleNamespace(run=run),types.SimpleNamespace(compare_metadata=None,compare_archives=None),None)
     self.assertEqual(calls,[]);self.assertIs(model.generate,original)
if __name__=='__main__':unittest.main()
