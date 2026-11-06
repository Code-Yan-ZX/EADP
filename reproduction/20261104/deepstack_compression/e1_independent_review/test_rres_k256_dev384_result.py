"""Exercise independent per-record logic on three previously accepted diagnostics."""
import copy,importlib.util,json,unittest
from pathlib import Path
from transformers import AutoTokenizer
I=Path(__file__).resolve().parent;B=I.parent
p=I/'review_rres_k256_dev384_result.py';spec=importlib.util.spec_from_file_location('_rres384_review_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class RecordTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.anchors=json.loads((B/'e6/rres_k256_pilot_v2/run_preflight001/processor_probe.json').read_text())['anchors']
  cls.path=B/'e6/rres_k256_execution_v2/run001/observations';t=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False);cls.decode=staticmethod(lambda ids:t.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False))
 def test_accepted_three_and_negative_controls(self):
  self.assertEqual(len(self.anchors),3)
  for a in self.anchors:
   name=a['task']+'_'+str(a['qid']);r=json.loads((self.path/(name+'.json')).read_text());raw=json.loads((self.path/(name+'_generated.json')).read_text())
   r['status']='completed';r['rope_reset_before_generation']=True;m.check_record(a,r,raw,self.decode)
   x=copy.deepcopy(r);x['delta']=[[99999]]
   with self.assertRaises(ValueError):m.check_record(a,x,raw,self.decode)
   x=copy.deepcopy(raw);x['qid']='wrong'
   with self.assertRaises(ValueError):m.check_record(a,r,x,self.decode)
   x=copy.deepcopy(r);x['decode_trace'][0]['layer_cache_lengths_after']=[0]*36
   with self.assertRaises(ValueError):m.check_record(a,x,raw,self.decode)
if __name__=='__main__':unittest.main()
