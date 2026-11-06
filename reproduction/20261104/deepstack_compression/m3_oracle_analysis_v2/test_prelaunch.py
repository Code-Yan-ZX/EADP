import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import prelaunch_gate,score_gate_v2 as gate,score_driver
ROOT=Path(__file__).resolve().parent.parent;V2=ROOT/'m3_oracle_verify_v2';RUN=ROOT/'m3_oracle_exec/runtime_001';REV=ROOT/'e1_independent_review'
def args():
 return SimpleNamespace(run=RUN,proof=V2/'runtime_001_verification_author.json',verify_started=V2/'monitor_verify002/started.json',verify_terminal=V2/'monitor_verify002/terminal.json',run_started=ROOT/'m3_oracle_exec/monitor_runtime001/started.json',run_terminal=ROOT/'m3_oracle_exec/monitor_runtime001/terminal.json',review=REV/'m3oracle_panel_review.json',review_started=REV/'monitor_m3oracle_panel001/started.json',review_terminal=REV/'monitor_m3oracle_panel001/terminal.json')
class GateTests(unittest.TestCase):
 def test_denied_before_scorer_import(self):
  with patch.object(prelaunch_gate,'validate',side_effect=ValueError('no review')),patch.object(score_driver.importlib.util,'spec_from_file_location') as loader:
   with self.assertRaisesRegex(ValueError,'no review'):score_driver.main()
   loader.assert_not_called()
 def test_driver_injects_v2_gate_in_isolated_stub(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'m3_oracle_verify_v2').mkdir();(root/'m3_oracle_analysis').mkdir();out=root/'out'
   (root/'m3_oracle_verify_v2/acceptance_binding_002.json').write_text(json.dumps({'python_version':sys.version,'executable_sha256':prelaunch_gate.sha(sys.executable)}))
   (root/'m3_oracle_analysis/score_oracle.py').write_text("from pathlib import Path\nimport sys\nscore_gate=None\ndef main():\n assert score_gate.__name__=='score_gate_v2'\n p=Path(sys.argv[sys.argv.index('--out')+1]);p.mkdir()\n (p/'scores.jsonl').write_text('synthetic')\n (p/'analysis.json').write_text('{}')\n return 0\n")
   with patch.object(score_driver,'ROOT',root),patch.object(score_driver.sys,'argv',['driver','--out',str(out)]),patch.object(prelaunch_gate,'validate',return_value={'review_sha256':'r','cpu_sha256':'c','source_sha256':{}}):
    self.assertEqual(score_driver.main(),0)
   self.assertEqual(json.loads((out/'v2_driver_proof.json').read_text())['status'],'completed')
 def test_completed_formal_and_review_chain_without_parsing_predictions(self):
  proof=gate.gate(args());self.assertEqual(proof['n_questions'],128);self.assertEqual(proof['new_generations'],768)
 def test_wrong_prediction_or_verifier_paths_rejected(self):
  for field in ('run','proof','verify_terminal','review'):
   a=args();setattr(a,field,ROOT/'wrong')
   with self.subTest(field=field),self.assertRaises(ValueError):gate.gate(a)
 def test_tampered_environment_and_unfinished_review_rejected(self):
  original=gate.read
  def changed(path):
   obj=copy.deepcopy(original(path))
   if Path(path)==V2/'environment_002_v2.json':obj['recorded_version_counts']={'fake':128}
   return obj
  with patch.object(gate,'read',side_effect=changed),self.assertRaisesRegex(ValueError,'environment'):
   gate.gate(args())
  a=args();a.review_terminal=REV/'wrong'
  with self.assertRaises(ValueError):gate.gate(a)
 def test_missing_independent_review_flag_rejected(self):
  original=gate.read
  def changed(path):
   obj=copy.deepcopy(original(path))
   if Path(path)==REV/'m3oracle_panel_review.json':obj['independent_review']=False
   return obj
  with patch.object(gate,'read',side_effect=changed),self.assertRaisesRegex(ValueError,'independent oracle review'):
   gate.gate(args())
if __name__=='__main__':unittest.main()
