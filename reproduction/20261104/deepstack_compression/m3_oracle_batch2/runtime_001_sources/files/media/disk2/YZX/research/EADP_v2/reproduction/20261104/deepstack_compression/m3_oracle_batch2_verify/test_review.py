import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT/'e1_independent_review'))
import review_m3batch2_panel as independent
class ReviewTests(unittest.TestCase):
 def test_proposals_match_frozen_contract(self):
  p=ROOT/'m3_oracle/proposals.py';sp=importlib.util.spec_from_file_location('frozen_teacher_proposals_test',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
  streams=[[(i*7+j*53)%113 for i in range(1024)] for j in range(4)];greedy=list(range(256));self.assertEqual(independent.proposals('55',greedy,streams),m.proposals('55',greedy,streams))
 def test_reference_token_payload_skipped_before_json_parsing(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'q';p.write_bytes(b'{"event":"question_started"}\n{"event":"target_encoded", invalid payload}\n{"event":"question_completed"}\n');self.assertEqual(independent.question_events(p),[{'event':'question_started'},{'event':'target_encoded'},{'event':'question_completed'}])
if __name__=='__main__':unittest.main()
