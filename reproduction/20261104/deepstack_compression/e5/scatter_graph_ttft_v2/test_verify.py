"""Synthetic CPU checks; do not load actual measured records."""
import importlib.util,unittest
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('scatter_ttft_formal_test',HERE/'verify.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
class TestFormal(unittest.TestCase):
 def test_schedule_is_fixed_and_balanced(self):
  anchors=[{'task':str(i),'qid':str(i)} for i in range(3)];jobs=v.schedule(anchors)
  self.assertEqual(len(jobs),36);self.assertEqual(jobs,v.schedule(anchors))
  counts=Counter((j['phase'],j['task'],j['method']) for j in jobs)
  self.assertTrue(all(counts[(phase,str(i),method)]==n for phase,n in [('warmup',1),('measured',5)] for i in range(3) for method in ('original','graph')))
 def test_failed_requirement_rejected(self):
  with self.assertRaises(ValueError):v.req(False,'synthetic missing evidence')
if __name__=='__main__':unittest.main(verbosity=2)
