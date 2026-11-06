import copy,importlib.util,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('_budget_ttft_background_test',HERE/'background_overlap.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
class Tests(unittest.TestCase):
 def test_windows_and_missing_or_foreign_namespace(self):
  rows=[{'event':'snapshot','query_start_ns':100,'query_end_ns':110,'time_namespace':'time:[1]','returncode':0},{'event':'snapshot_logging_complete','query_start_ns':100,'logging_end_ns':120}]
  self.assertEqual(b.windows(rows,'time:[1]')[0]['query_and_first_fsync_window'],[100,120])
  with self.assertRaisesRegex(ValueError,'missing'):b.windows(rows[:-1],'time:[1]')
  with self.assertRaisesRegex(ValueError,'namespace'):b.windows(rows,'time:[2]')
 def test_overlap_and_second_fsync_unknown(self):
  jobs=[{'task':'D','qid':str(i),'group_id':str(i),'budget':256,'phase':'measure','order':i} for i in range(48)];events=[]
  for job in jobs:
   boundary={'t0_before_image_read_ns':90,'first_token_selected_ns':105,'generation_complete_ns':130,'time_namespace':'time:[1]'}
   events.extend([{'event':'started','job':job},{'event':'raw_generation','job':job},{'event':'completed','job':job,'payload':{'boundaries':boundary}}])
  obs=[{'query_window':[100,110],'query_and_first_fsync_window':[100,120]}];a=b.matches(jobs,events,obs,'time:[1]')
  self.assertEqual(a[0]['ttft_ps_query_overlap'],[0]);self.assertEqual(a[0]['second_fsync_overlap'],'unknown')
  events[2]['payload']['boundaries']['time_namespace']='time:[2]'
  with self.assertRaisesRegex(ValueError,'namespace'):b.matches(jobs,events,obs,'time:[1]')
