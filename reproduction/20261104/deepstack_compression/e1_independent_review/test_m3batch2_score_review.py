import tempfile,unittest
from pathlib import Path
from e1_independent_review import review_m3batch2_scores as r
class RecomputeTests(unittest.TestCase):
 def test_generated_only_does_not_parse_reference_payload(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'events';p.write_bytes(b'{"event":"target_encoded", malformed}\n{"event":"generation_completed","source":"MAIN"}\n');self.assertEqual(r.generated_only(p),[{'event':'generation_completed','source':'MAIN'}])
 def test_seven_arms_and_four_frozen_contrasts(self):
  plan=r.read(r.A/'plan.json');self.assertEqual(list(r.ARMS),plan['arms']);self.assertEqual(list(r.CONTRASTS),plan['contrasts']);self.assertEqual(plan['bootstrap']['replicates'],10000)
if __name__=='__main__':unittest.main()
