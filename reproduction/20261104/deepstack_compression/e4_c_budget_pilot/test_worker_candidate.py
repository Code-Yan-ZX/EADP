import importlib.util,json,tempfile,types,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('budget_worker_candidate_test',HERE/'worker_candidate.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

class Tests(unittest.TestCase):
 def test_real_journal_is_durable_before_later_failure(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);wrapper=types.SimpleNamespace(pruner=types.SimpleNamespace());seen=[]
   class KV:
    def __init__(self,w,forced):self.fingerprints=['forced' if forced else 'natural']
    def __enter__(self):return self
    def __exit__(self,*args):pass
   def fresh(_torch,w,_native,_row,k,ds,_generate,raw,_archive,token,capture_prefill_logits=False):
    seen.append((w.visual_token_num,w.pruner.visual_token_num,k,ds,capture_prefill_logits));raw({'output_ids':[7]});raise ValueError('later failure')
   def run(_rows,_refs,_token,fresh,_light,_archive,emit,_compare,_arrays):
    fresh({'task':'D','qid':'1'},512,None,lambda x:emit({'event':'raw','output':x}),lambda _:None)
   with self.assertRaisesRegex(ValueError,'later failure'):
    worker.run_after_future_admission(None,wrapper,None,{'rows':[],'records':{}},out,None,3,fresh,None,KV,types.SimpleNamespace(run=run),types.SimpleNamespace(compare_metadata=None,compare_archives=None),None)
   self.assertEqual(seen,[(512,512,512,True,True)])
   self.assertEqual(json.loads((out/'journal.jsonl').read_text())['output']['output_ids'],[7])
   self.assertFalse((out/'worker_complete.json').exists())
if __name__=='__main__':unittest.main()
