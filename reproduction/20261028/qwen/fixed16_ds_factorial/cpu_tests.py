"""CPU-only tests for safe prefill-argument transformation and gate boundaries."""
import copy,json,unittest
from types import SimpleNamespace
from pathlib import Path
import common as c
from intervention import change

class FakeTensor:
 def __init__(self,*shape,dtype='torch.bfloat16',device='cpu',true_count=1024):
  self.shape=shape;self.dtype=dtype;self.device=device;self.true_count=true_count
 def sum(self):return SimpleNamespace(item=lambda:self.true_count)
def tensor(*shape,**kwargs):return FakeTensor(*shape,**kwargs)

class Tests(unittest.TestCase):
 def setUp(self):
  self.emb=tensor(1,1056,4096);self.mask=tensor(1,1056,dtype='torch.bool');self.ds=[tensor(1024,4096) for _ in range(3)]
  self.pos=tensor(3,1,1056);self.cache=tensor(1056);self.attn=tensor(1,1056)
 def make(self,on):return dict(inputs_embeds=self.emb,position_ids=self.pos,cache_position=self.cache,attention_mask=self.attn,
  visual_pos_masks=self.mask if on else None,deepstack_visual_embeds=self.ds if on else None)
 def test_endpoints_unchanged_by_identity(self):
  for arm,on in [('plain',True),('eadp1024',False)]:
   raw=self.make(on);actual,applied,error=change(arm,raw,self.mask,self.ds)
   self.assertIs(actual,raw);self.assertFalse(applied);self.assertIsNone(error)
 def test_plain_off_changes_exactly_two_fields(self):
  raw=self.make(True);result,applied,error=change('plain_ds_off',raw,self.mask,self.ds)
  self.assertTrue(applied);self.assertIsNone(error);self.assertIsNone(result['visual_pos_masks']);self.assertIsNone(result['deepstack_visual_embeds'])
  self.assertIs(raw['visual_pos_masks'],self.mask);self.assertIs(raw['deepstack_visual_embeds'],self.ds)
  for key in ('inputs_embeds','position_ids','cache_position','attention_mask'):self.assertIs(result[key],raw[key])
 def test_eadp_on_uses_real_supplied_tensors_only(self):
  raw=self.make(False);result,applied,error=change('eadp1024_ds_on',raw,self.mask,self.ds)
  self.assertTrue(applied);self.assertIsNone(error);self.assertIs(result['visual_pos_masks'],self.mask)
  self.assertIs(result['deepstack_visual_embeds'],self.ds)
  for key in ('inputs_embeds','position_ids','cache_position','attention_mask'):self.assertIs(result[key],raw[key])
 def test_bad_shape_missing_or_preexisting_does_not_modify(self):
  for arm,raw,mask,ds in [('eadp1024_ds_on',self.make(False),None,self.ds),('eadp1024_ds_on',self.make(False),tensor(1,4),self.ds),
    ('eadp1024_ds_on',self.make(True),self.mask,self.ds),('plain_ds_off',self.make(False),self.mask,self.ds),
    ('eadp1024_ds_on',self.make(False),self.mask,[tensor(1024,4095)]*3),
    ('eadp1024_ds_on',self.make(False),tensor(1,1056,dtype='torch.bool',true_count=1023),self.ds),
    ('eadp1024_ds_on',self.make(False),tensor(1,1056,dtype='torch.bool',device='other'),self.ds),
    ('eadp1024_ds_on',self.make(False),self.mask,[tensor(1024,4096,dtype='torch.float32')]*3)]:
   result,applied,error=change(arm,raw,mask,ds);self.assertIs(result,raw);self.assertFalse(applied);self.assertTrue(error)
 def test_base_endpoints_are_verified_before_variants(self):
  code=(c.HERE/'worker_draft.py').read_text();self.assertLess(code.index("'endpoints_verified.json'"),code.index("('plain_ds_off','plain')"))
  self.assertIn("'output_ids','output_suffix_ids'",code);self.assertIn("'prefill_cache_position','rope_deltas'",code)
  self.assertIn("status='raw_preserved'",code);self.assertIn('continuation_%04d_',code)
 def test_prior_native_score_proofs_and_no_new_execution(self):
  old=c.read(c.PREV/'execution/independent_verification.json');self.assertTrue(old['success']);self.assertEqual(old['n_native_generations'],32)
  proof=c.read(c.PREV/'root_terminal_process_proof.json');self.assertEqual(proof['root_owned_exec_actual_exit_code'],0)
  score=c.read(c.FULL/'score_terminal_process_proof.json');self.assertEqual(score['root_owned_score_exec_actual_exit_code'],0)
  self.assertFalse((c.HERE/'execution').exists());self.assertFalse((c.HERE/'root_gpu_allocation.json').exists())
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 if c.PLAN.exists():c.save(c.HERE/'cpu_tests.json',dict(success=result.wasSuccessful(),n_tests=result.testsRun,
  plan_sha256=c.sha(c.PLAN),CPU_only=True,model_loaded=False,GPU_started=False,GT_read=False,scoring_started=False))
 raise SystemExit(0 if result.wasSuccessful() else 1)
