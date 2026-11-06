import unittest
from types import SimpleNamespace
from unittest.mock import patch
import score_gate as g
class GateTests(unittest.TestCase):
 def test_missing_proofs_reject_before_open(self):
  with patch.object(g.Path,'is_file',return_value=False),patch.object(g,'read',side_effect=AssertionError('unexpected file read')):
   with self.assertRaisesRegex(ValueError,'completed batch2'):g.gate(SimpleNamespace(run=g.RUN,proof=g.PROOF,verify_started=g.FS,verify_terminal=g.FT,run_started=g.RS,run_terminal=g.RT,review=g.REVIEW,review_started=g.IS,review_terminal=g.IT))
 def test_acceptance_fields_and_corruption(self):
  f={'status':'passed','stage':'oracle_batch2_full','n_questions':128,'new_generations':768,'scoring_performed':False,'reference_sources_opened_by_verifier':False,'verifier_sha256':'v'}
  r={'status':'passed','independent_review':True,'n_questions':128,'new_generations':768,'scoring_performed':False,'reference_sources_opened':False,'errors':[],'formal_proof_sha256':'f','reviewer_sha256':'r'}
  g.formal_fields(f,'v');g.review_fields(r,'f','r')
  for k,v in [('n_questions',127),('new_generations',767),('status','failed'),('scoring_performed',True),('verifier_sha256','bad')]:
   with self.subTest(k=k),self.assertRaises(ValueError):g.formal_fields({**f,k:v},'v')
  for k,v in [('independent_review',False),('n_questions',127),('formal_proof_sha256','bad'),('reviewer_sha256','bad'),('errors',['bad'])]:
   with self.subTest(k=k),self.assertRaises(ValueError):g.review_fields({**r,k:v},'f','r')
if __name__=='__main__':unittest.main()

class AnalysisAdmissionTests(unittest.TestCase):
 def test_missing_source_review_blocks_before_open(self):
  import analysis_gate
  with patch.object(g.Path,'is_file',return_value=False),patch.object(g.Path,'read_text',side_effect=AssertionError('unexpected read')):
   with self.assertRaisesRegex(ValueError,'scorer locked'):analysis_gate.validate()
 def test_frozen_second_look_plan_and_sample(self):
  import json,hashlib
  p=json.loads((g.HERE/'plan.json').read_text());sample=(g.B/'sample_002.json').read_bytes();first=json.loads((g.ROOT/'m3_oracle/sample_001.json').read_text())['examples'];second=json.loads(sample)['examples']
  self.assertEqual(hashlib.sha256(sample).hexdigest(),p['sample_sha256']);self.assertEqual((p['questions'],p['new_generations'],len(second)),(128,768,128));self.assertEqual(len(p['contrasts']),4);self.assertIn('Second exploratory look',p['look']);self.assertFalse({x['group_id'] for x in first}&{x['group_id'] for x in second})

class FullSyntheticGate(unittest.TestCase):
 def test_complete_passes_and_missing_question_rejected(self):
  from types import SimpleNamespace
  files={str(g.RUN/'panel'/f'question_{i}.jsonl'):'digest' for i in range(128)}
  formal={'status':'passed','stage':'oracle_batch2_full','n_questions':128,'new_generations':768,'scoring_performed':False,'reference_sources_opened_by_verifier':False,'verifier_sha256':'digest','python_version':'v','executable_sha256':'digest','approval_sha256':'digest','result_files_sha256':files,**{k:'digest' for k in ('run_start_sha256','run_terminal_sha256','launch_sha256','exit_sha256','panel_events_sha256','worker_complete_sha256')}}
  review={'status':'passed','independent_review':True,'n_questions':128,'new_generations':768,'scoring_performed':False,'reference_sources_opened':False,'errors':[],'formal_proof_sha256':'digest','reviewer_sha256':'digest',**{k:'digest' for k in ('formal_started_sha256','formal_terminal_sha256','frozen_sha256','host_receipt_sha256','source_manifest_sha256')}}
  src={'/synthetic/source':'digest'};acc={'/synthetic/verifier':'digest'};manifest={'source_sha256':src,'acceptance_source_sha256':acc,**{k:'digest' for k in ('acceptance_cpu_proof_sha256','acceptance_cpu_started_sha256','acceptance_cpu_terminal_sha256')}}
  objects={str(g.PROOF):formal,str(g.REVIEW):review,str(g.ROOT/'m3_oracle_verify_v2/environment_002_v2.json'):{'python_version':'v','executable_sha256':'digest'},str(g.FS):{'cwd':'/','command':['/python']},str(g.B/'runtime_001_sources/manifest.json'):manifest,str(g.V/'cpu_check_003/verification.json'):{'status':'passed','source_sha256_before':acc,'source_sha256_after':acc},str(g.B/'runtime_approval.json'):{'cpu_proof':'/synthetic/cpu'},'/synthetic/cpu':{'status':'passed','source_sha256_before':src,'source_sha256_after':src},str(g.B/'execution_plan_002.json'):{},str(g.RS):{},str(g.RT):{},str(g.RUN/'launch.json'):{},str(g.RUN/'exit.json'):{}}
  def fake_read(p):return objects[str(p)]
  args=SimpleNamespace(run=g.RUN,proof=g.PROOF,verify_started=g.FS,verify_terminal=g.FT,run_started=g.RS,run_terminal=g.RT,review=g.REVIEW,review_started=g.IS,review_terminal=g.IT)
  fake_spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None));fake_verifier=SimpleNamespace(check_terminal=lambda *x:None)
  with patch.object(g.Path,'is_file',return_value=True),patch.object(g,'read',side_effect=fake_read),patch.object(g.batch_gate,'sha',return_value='digest'),patch.object(g.batch_gate,'supervised'),patch.object(g.batch_gate,'validate',return_value={'approval_sha256':'digest'}),patch.object(g.importlib.util,'spec_from_file_location',return_value=fake_spec),patch.object(g.importlib.util,'module_from_spec',return_value=fake_verifier):
   self.assertEqual(g.gate(args),formal);formal['result_files_sha256']=dict(list(files.items())[:-1])
   with self.assertRaisesRegex(ValueError,'incomplete question'):g.gate(args)
