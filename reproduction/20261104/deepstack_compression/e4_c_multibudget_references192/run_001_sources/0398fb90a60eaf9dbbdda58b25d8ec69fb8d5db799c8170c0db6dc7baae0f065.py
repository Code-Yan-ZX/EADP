import copy,json,sys,tempfile,types,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent
sys.path.append(str(BASE/'e4_c_multibudget_fixed32'))
import importlib.util
spec=importlib.util.spec_from_file_location('_fixed_test_core',BASE/'e4_c_multibudget_fixed32/core.py');fixed_core=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed_core)
import test_fixed
spec=importlib.util.spec_from_file_location('_references_core',Path(__file__).resolve().parent/'core.py');bulk=importlib.util.module_from_spec(spec);spec.loader.exec_module(bulk)
class Tests(unittest.TestCase):
 def sample(self,k):
  x,p=test_fixed.Tests().sample(k);recipe={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0}
  p['generation_scalars']=recipe;x['generation_scalars']=dict(recipe,max_new_tokens=32);x['answer']='x';x['arrays']={'kind':'summary_only','tensors':{}};x['cache_fingerprints']=x['cache_fingerprints'][:1]
  return x,p
 def test_valid_reference_all_budgets(self):
  for k in bulk.BUDGETS:
   x,p=self.sample(k);self.assertTrue(bulk.validate(x,p,fixed_core))
 def test_shared_bad_prefill_early_eos_and_decode_cache_reject(self):
  for kind in ('prefill','eos','cache','bad_kv'):
   x,p=self.sample(128)
   if kind=='prefill':x['ordered_ids']=[12]*128
   if kind=='eos':x['output_ids'][4]=9
   if kind=='cache':x['decode_trace'][30]['layer_cache_lengths_after'][1]+=1
   if kind=='bad_kv':x['cache_fingerprints'][0][1]['value']['sha256']='z'*64
   with self.assertRaises(ValueError):bulk.validate(x,p,fixed_core)
 def test_schedule_222_inputs_and_reject_duplicates(self):
  rows=[{'task':'D','qid':str(i),'group_id':str(i),'split':'DEV'} for i in range(222)]
  self.assertEqual(len(bulk.schedule(rows)),888);rows[-1]['group_id']=rows[0]['group_id']
  with self.assertRaisesRegex(ValueError,'222 DEV'):bulk.schedule(rows)
 def test_generation_failure_is_logged_and_partial_output_retained(self):
  rows=[{'task':'D','qid':str(i),'group_id':str(i),'split':'DEV','image_sha256':'i','message_sha256':'m'} for i in range(222)]
  records={(r['task'],r['qid'],k):dict(r,budget=k,ds_on=True) for r in rows for k in bulk.BUDGETS}
  def failure(_torch,_wrapper,_native,_row,_k,_on,_generate,raw,_archive,**_kw):raw({'output_ids':[1]*32});raise ValueError('after generation')
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'after generation'):bulk.run(None,types.SimpleNamespace(pruner=types.SimpleNamespace()),None,{'rows':rows,'records':records},d,None,failure,fixed_core,None)
   events=[json.loads(x)['event'] for x in (Path(d)/'journal.jsonl').read_text().splitlines()]
   self.assertEqual(events,['arm_started','raw_generation','arm_failed']);self.assertFalse((Path(d)/'worker_complete.json').exists())
if __name__=='__main__':unittest.main()
