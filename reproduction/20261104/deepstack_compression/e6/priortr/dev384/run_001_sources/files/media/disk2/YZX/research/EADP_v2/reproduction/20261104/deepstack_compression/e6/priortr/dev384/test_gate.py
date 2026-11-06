import unittest
import gate
class GateTests(unittest.TestCase):
 def test_no_execution_without_review(self):
  for preflight in (False,True):
   with self.assertRaises(ValueError):gate.execution_gate(preflight=preflight)
if __name__=='__main__':unittest.main()

class PriorDev32EvidenceTests(unittest.TestCase):
 def test_accepted_chain(self):
  import gate
  self.assertEqual(set(gate.dev32_evidence()),{'formal','review','run_frozen','formal_terminal','review_terminal'})
 def test_replaced_or_missing_artifact_rejected(self):
  from unittest.mock import patch
  import gate
  orig=gate.sha;target=(gate.HERE.parent/'dev32/run_001/data/generated.jsonl').resolve()
  def corrupt(path):
   return '0'*64 if __import__('pathlib').Path(path).resolve()==target else orig(path)
  with patch.object(gate,'sha',side_effect=corrupt):
   with self.assertRaisesRegex(ValueError,'artifacts missing or changed'):gate.dev32_evidence()
  def missing(path):
   if __import__('pathlib').Path(path).resolve()==target:raise FileNotFoundError(str(path))
   return orig(path)
  with patch.object(gate,'sha',side_effect=missing):
   with self.assertRaises(FileNotFoundError):gate.dev32_evidence()
