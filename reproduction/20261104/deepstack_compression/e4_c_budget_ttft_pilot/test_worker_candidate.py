import json,tempfile,types,unittest
from pathlib import Path
import contract,worker_candidate
class Tests(unittest.TestCase):
 def test_partial_output_durable_and_no_silent_resume(self):
  rows=[{'task':str(i),'qid':str(i),'group_id':'g'+str(i),'split':'DEV'} for i in range(3)];jobs=contract.schedule(rows)
  refs={(r['task'],r['qid'],k):{} for r in rows for k in contract.BUDGETS}
  def fail(*args):args[-1]({'output_ids':[1,2]});raise ValueError('later failure')
  with tempfile.TemporaryDirectory() as d:
   journal=Path(d)/'journal.jsonl'
   with self.assertRaisesRegex(ValueError,'later failure'):
    worker_candidate.execute_schedule(rows,refs,jobs,contract,types.SimpleNamespace(execute=fail),None,None,None,None,None,None,None,journal)
   events=[json.loads(x)['event'] for x in journal.read_text().splitlines()];self.assertEqual(events,['started','raw_generation','failed'])
   with self.assertRaises(FileExistsError):
    worker_candidate.execute_schedule(rows,refs,jobs,contract,types.SimpleNamespace(execute=fail),None,None,None,None,None,None,None,journal)
