"""CPU integration/negative gates; no GPU model construction."""
import ast,copy,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
N=Path(__file__).resolve().parent

def mod(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=mod(N/'followup_controller.py','test_controller');w=mod(N/'followup_worker.py','test_worker');p=mod(N/'followup_publisher.py','test_publisher');plan=c.read(c.PLAN)
class Gates(unittest.TestCase):
 def test_scope(self):
  self.assertEqual(len(plan['jobs']),5);self.assertEqual(sum(j['n']for j in plan['jobs'].values()),29854);self.assertEqual(len(plan['groups']),3);self.assertEqual(c.read(c.STATE)['registered_jobs'],[])
 def test_original_dependency(self):
  self.assertTrue(c.dependencies()['success']);self.assertEqual(c.dependencies()['published_groups_confirmed'],55)
 def test_sources(self):c.verify(plan)
 def test_source_reject(self):
  bad=copy.deepcopy(plan);bad['worker_sha256']='0'*64
  with self.assertRaises(ValueError):c.verify(bad)
 def test_reused_fairness(self):
  for g in c.GROUPS:self.assertTrue(c.fairness(g,plan['jobs'])['success'])
 def test_pair_mismatch_reject(self):
  jobs=copy.deepcopy(plan['jobs']);jobs['next_gqa_E128']['parameters']['beta']=1
  with self.assertRaises(ValueError):c.fairness(c.GROUPS[0],jobs)
 def test_first_context_and_memory(self):
  job=plan['jobs']['next_gqa_E128'];gpu=dict(entries=[],free_mib=44257)
  self.assertTrue(c.OLD.launch_gate(job,gpu,set(),{},True)['ready'])
  self.assertFalse(c.OLD.launch_gate(job,gpu,{101},{101:True},True)['ready'])
  entry=dict(pid=101,memory_mib=19000,alive=True,command='owned');gpu['entries']=[entry]
  self.assertFalse(c.OLD.launch_gate(job,gpu,{101},{101:False},True)['ready'])
  self.assertTrue(c.OLD.launch_gate(job,gpu,{101},{101:True},True)['ready'])
  gpu['free_mib']=22499;self.assertFalse(c.OLD.launch_gate(job,gpu,{101},{101:True},True)['ready'])
 def test_two_reservations(self):
  self.assertFalse(c.OLD.launch_gate(plan['jobs']['next_gqa_E128'],dict(entries=[],free_mib=44000),{1,2},{1:True,2:True},True)['ready'])
 def test_unknown_zero_context(self):
  entry=dict(pid=99,memory_mib=0,alive=True,command='unknown runner');gpu=dict(entries=[entry],free_mib=44000);job=plan['jobs']['next_gqa_E128']
  self.assertFalse(c.OLD.launch_gate(job,gpu,set(),{},True)['ready']);entry['command']='/usr/local/bin/ollama runner --model x';self.assertTrue(c.OLD.launch_gate(job,gpu,set(),{},True)['ready'])
 def test_both_first_records_required(self):
  tree=ast.parse((N/'followup_controller.py').read_text());source=ast.get_source_segment((N/'followup_controller.py').read_text(),next(n for n in ast.walk(tree)if isinstance(n,ast.DictComp)and 'first_json' in ast.unparse(n)))
  self.assertIn('all(',source);self.assertIn("'prediction','runtime'",source)
 def test_scoring_AST_unchanged(self):
  old=ast.parse((c.C/'legacy_streamwait_worker.py').read_text());new=ast.parse((N/'followup_worker.py').read_text())
  for name in ['scoring','empty_gate_patch','complete_answer_scorer','guidance','verify_images']:
   get=lambda tree:ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name),include_attributes=False)
   self.assertEqual(get(old),get(new),name)
 def test_existing_outputs_protected(self):
  job=copy.deepcopy(plan['jobs']['next_gqa_E128'])
  with tempfile.TemporaryDirectory()as td:
   path=Path(td)/'pred';path.write_text('partial');job['output_paths']['prediction']=str(path)
   with self.assertRaises(FileExistsError):c.OLD.PRIOR.protected(job)
 def test_actual_model_snapshot(self):
  from types import SimpleNamespace
  class Fake:
   alpha=.5;beta=2.;visual_token_num=128;config=SimpleNamespace(_attn_implementation='sdpa')
   def get_model(self):return self
   def parameters(self):return iter([SimpleNamespace(dtype='torch.float16')])
   def get_vision_tower(self):return SimpleNamespace()
  s=w.model_object_snapshot(Fake());self.assertEqual((s['alpha'],s['beta'],s['visual_token_num'],s['dtype']),(.5,2.,128,'torch.float16'))
 def test_official_real_GQA_math(self):
  old=c.read(c.C/'legacy_streamwait_continuation.plan.json')['jobs']['next_gqa_LRMAIN025'];proto=c.read(plan['jobs']['next_gqa_E128']['control_protocol']);s=w.scoring(proto,Path(old['output_paths']['prediction']));saved=c.read(old['output_paths']['score']);self.assertEqual(s['per_question'],saved['per_question']);self.assertEqual(s['value'],saved['value'])
 def test_official_real_Viz_math_and_release(self):
  old=c.read(c.C/'legacy_streamwait_continuation.plan.json')['jobs']['next_vizwiz_LRMAIN025'];proto=c.read(plan['jobs']['next_vizwiz_AZ128']['control_protocol']);s=w.scoring(proto,Path(old['output_paths']['prediction']));saved=c.read(old['output_paths']['score']);self.assertEqual(s['per_question'],saved['per_question']);self.assertEqual(s['value'],saved['value'])
  orig=c.read(old['control_protocol']);v=p.vizwiz_released_audit([dict(protocol=orig,score=saved)]);self.assertEqual(v['rows'][0]['n'],4319);self.assertGreater(v['rows'][0]['released_min_matches_over3'],v['rows'][0]['official_loo'])
 def test_no_GPU_import(self):self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main(verbosity=2)
