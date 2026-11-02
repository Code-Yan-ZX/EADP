"""CPU structural and boundary tests; no model, scoring, or GPU."""
import ast, json, sys, unittest
import common as c
sys.path.insert(0,str(c.PANEL))
from intervention import change
from types import SimpleNamespace
class Tensor:
 def __init__(self,shape,dtype='torch.bfloat16',device='cpu',count=1024):self.shape=shape;self.dtype=dtype;self.device=device;self.count=count
 def sum(self):return SimpleNamespace(item=lambda:self.count)
class Tests(unittest.TestCase):
 def test_intervention_fail_safe_and_two_fields(self):
  e=Tensor((1,1056,4096)); m=Tensor((1,1056),'torch.bool'); ds=[Tensor((1024,4096)) for _ in range(3)]
  on=dict(inputs_embeds=e,position_ids=object(),attention_mask=object(),cache_position=object(),visual_pos_masks=m,deepstack_visual_embeds=ds)
  off,applied,error=change('plain_ds_off',on,m,ds);self.assertTrue(applied);self.assertIsNone(error)
  self.assertEqual({k for k in on if on[k] is not off[k]},{'visual_pos_masks','deepstack_visual_embeds'})
  new,applied,error=change('eadp1024_ds_on',off,m,ds);self.assertTrue(applied);self.assertIs(new['visual_pos_masks'],m);self.assertIs(new['deepstack_visual_embeds'],ds)
  bad,applied,error=change('eadp1024_ds_on',off,Tensor((1,1056),'torch.bool',count=1023),ds);self.assertIs(bad,off);self.assertFalse(applied);self.assertTrue(error)
 def test_manifest_and_fixed16_link(self):
  items=c.rows(c.BASE/'input_manifest.jsonl');panel=c.read(c.PANEL/'plan.json')['panel'];self.assertEqual(len(items),5349);self.assertEqual(len(panel),16)
  for item in panel:self.assertEqual(items[item['position']]['index'],item['index'])
 def test_preservation_and_order(self):
  s=(c.HERE/'worker.py').read_text();ast.parse(s)
  self.assertLess(s.index('write(ps,pred)'),s.index("assert failure is None and isinstance(response,str)"))
  self.assertLess(s.index("out/'endpoints_verified.json'"),s.index("for item in items:"))
 def test_score_gate_fails_closed(self):
  from contracts import require_score_gate
  with self.assertRaises((AssertionError,KeyError)):require_score_gate({}, {}, 'a','b','c')
 def test_prior_proofs(self):
  f=c.read(c.PANEL/'execution/independent_verification.json');self.assertTrue(f['success']);self.assertEqual(f['n_native_generations'],64)
  b=c.read(c.BASE/'score/score_report.json');self.assertTrue(b['success']);self.assertEqual(b['n'],5349)
 def test_unstarted(self):
  self.assertFalse((c.HERE/'execution').exists());self.assertFalse((c.HERE/'root_gpu_allocation.json').exists())
if __name__=='__main__':
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Tests), unittest.defaultTestLoader.loadTestsFromName('test_verify_generation')])
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 if c.PLAN.exists():c.save(c.HERE/'cpu_tests.json',dict(success=result.wasSuccessful(),n_tests=result.testsRun,plan_sha256=c.sha(c.PLAN),CPU_only=True,model_loaded=False,GPU_started=False,GT_read=False,scoring_started=False))
 raise SystemExit(0 if result.wasSuccessful() else 1)
