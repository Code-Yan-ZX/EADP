"""CPU tests of input/score boundaries and failure preservation; no model/GPU."""
import ast,io,json,unittest
from pathlib import Path
import common as c
from contracts import preserve_raw,check_observation,require_score_gate

class Tests(unittest.TestCase):
 def test_complete_order_and_no_answers(self):
  x=c.rows(c.HERE/'input_manifest.jsonl');self.assertEqual(len(x),5349)
  self.assertEqual([r['position'] for r in x],list(range(5349)))
  self.assertEqual(len({r['index'] for r in x}),5349)
  self.assertTrue(all('answer' not in r and 'prediction' not in r for r in x))
  self.assertEqual(len({r['image_path'] for r in x}),1286)
 def test_raw_response_saved_before_reject(self):
  stream=io.StringIO();item={'position':0,'index':'1','processor_expected':{'x':1}}
  preserve_raw(stream,item,'real response',None)
  with self.assertRaises(AssertionError):check_observation(item,'real response',None,{'observer_errors':['bad']})
  self.assertEqual(json.loads(stream.getvalue())['prediction'],'real response')
  stream=io.StringIO();preserve_raw(stream,item,None,'generation failed')
  self.assertEqual(json.loads(stream.getvalue())['native_failure'],'generation failed')
 def test_empty_native_answer_remains_valid(self):
  item={'processor_expected':{'x':1}};obs={'observer_errors':[],'actual_processor':{'x':1},'decode_matches':True,'num_image_tokens':1024}
  check_observation(item,'',None,obs)
 def test_score_needs_actual_exit_and_all_proof_links(self):
  p=dict(root_owned_exec_actual_exit_code=0,all_original_processes_exited=True,plan_sha256='p',release_sha256='r',generation_verification_sha256='v')
  r=dict(success=True,native_container_exited=True,plan_sha256='p',generation_verification_sha256='v')
  require_score_gate(p,r,'p','r','v')
  for key,value in [('root_owned_exec_actual_exit_code',1),('all_original_processes_exited',False),('release_sha256','wrong')]:
   changed=dict(p);changed[key]=value
   with self.assertRaises(AssertionError):require_score_gate(changed,r,'p','r','v')
 def test_direct_original_methods_and_budget(self):
  tree=ast.parse((c.HERE/'worker.py').read_text());calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
  self.assertIn('Qwen3VLChatEADP',calls)
  text=(c.HERE/'worker.py').read_text();self.assertIn('Qwen3VLChatFixedRes.generate_inner_transformers_fixed',text)
  self.assertIn('Qwen3VLChatCDPruner.generate_inner_transformers_fixed',text)
  self.assertIn("recipe['visual_token_num']=1024",text)
 def test_predecessor_completed(self):
  report=c.read(c.PANEL/'execution/independent_verification.json');root=c.read(c.PANEL/'root_terminal_process_proof.json')
  self.assertTrue(report['success']);self.assertEqual(report['n_native_generations'],32)
  self.assertTrue(root['all_original_processes_exited']);self.assertEqual(root['root_owned_exec_actual_exit_code'],0)
 def test_score_provenance_gate_before_any_GT_access(self):
  code=(c.HERE/'score_terminal.py').read_text()
  gate=code.index("verification=c.read(");raw_hash=code.index("complete['predictions'][arm]")
  gt_hash=code.index("plan['score_only_sha256']");gt_read=code.index("canonical=c.rows(")
  self.assertLess(gate,raw_hash);self.assertLess(raw_hash,gt_hash);self.assertLess(gt_hash,gt_read)
  self.assertIn("'original_exited'",code);self.assertIn("independent_token_redecodes",code)
 def test_post_process_contract(self):
  self.assertIn('wrapper.post_process is False',(c.HERE/'worker.py').read_text())
  self.assertIn("runtime['post_process'] is False",(c.HERE/'verify_generation.py').read_text())
 def test_no_auto_launch(self):
  self.assertFalse((c.HERE/'execution').exists());self.assertFalse((c.HERE/'root_gpu_allocation.json').exists())
if __name__=='__main__':
 import sys
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));assert 'torch' not in sys.modules
 if c.PLAN.exists():c.save(c.HERE/'cpu_tests.json',dict(success=result.wasSuccessful(),tests=result.testsRun,plan_sha256=c.sha(c.PLAN),CPU_only=True,GPU_started=False,model_loaded=False))
 sys.exit(0 if result.wasSuccessful() else 1)
