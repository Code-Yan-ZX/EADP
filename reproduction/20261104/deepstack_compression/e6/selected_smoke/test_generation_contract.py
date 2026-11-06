from types import SimpleNamespace
import unittest
from generation_contract import controls
class ControlsTests(unittest.TestCase):
 def test_resolved_defaults_and_overrides(self):
  config=SimpleNamespace(eos_token_id=[2],num_beams=1)
  c=controls(config,{'do_sample':False,'max_new_tokens':2048},False)
  self.assertEqual(c['eos_token_id'],[2]);self.assertEqual(c['num_beams'],1);self.assertIs(c['do_sample'],False)
 def test_forced_requires_actual_min_eos_and_processor(self):
  cfg=SimpleNamespace(eos_token_id=[2]);kw={'min_new_tokens':33,'eos_token_id':[],'logits_processor':['force']}
  self.assertEqual(controls(cfg,kw,True)['min_new_tokens'],33)
  for field in kw:
   bad=dict(kw);del bad[field]
   with self.assertRaises(ValueError):controls(cfg,bad,True)
 def test_natural_extra_processor_is_rejected(self):
  with self.assertRaises(ValueError):controls(SimpleNamespace(),{'logits_processor':['unexpected']},False)
if __name__=='__main__':unittest.main()
