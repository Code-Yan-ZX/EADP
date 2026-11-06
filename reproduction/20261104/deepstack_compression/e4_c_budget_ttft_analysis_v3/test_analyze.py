import copy,importlib.util,types,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
spec=importlib.util.spec_from_file_location('_ttft_analysis_test',HERE/'analyze.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
spec=importlib.util.spec_from_file_location('_ttft_analysis_contract',BASE/'e4_c_budget_ttft_pilot_v3/contract.py');contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
class Tests(unittest.TestCase):
 def sample(self):
  rows=[{'task':str(i),'qid':str(i),'group_id':'g'+str(i),'split':'DEV'} for i in range(3)];jobs=contract.schedule(rows);events=[];annotations=[]
  for job in jobs:
   payload={'boundaries':{'t0_before_image_read_ns':100,'first_token_selected_ns':100+job['budget']},'output_ids':[1]}
   events += [{'event':'started','job':job},{'event':'raw_generation','job':job},{'event':'completed','job':job,'payload':payload}]
   annotations.append({'order':job['order'],'ttft_health_overlap_indices':[0] if job['order']==0 else []})
  return jobs,events,annotations
 def test_paired_ratios_and_all_samples(self):
  jobs,events,ann=self.sample();x=a.summarize(jobs,events,ann);self.assertEqual(len(x['samples']),48);self.assertEqual(len(x['cells']),12)
  cell=next(c for c in x['cells'] if c['budget']==128);self.assertEqual(cell['paired_ratios_to_k256'],[.5]*3)
  self.assertEqual(x['samples'][0]['direct_health_overlap_indices'],[0])
 def test_nonpositive_or_missing_refused(self):
  jobs,events,ann=self.sample();bad=copy.deepcopy(events);bad[2]['payload']['boundaries']['first_token_selected_ns']=100
  with self.assertRaisesRegex(ValueError,'nonpositive'):a.summarize(jobs,bad,ann)
  with self.assertRaisesRegex(ValueError,'incomplete'):a.summarize(jobs,events[:-1],ann)
 def test_no_proofs_blocks_before_journal(self):
  from unittest.mock import patch
  import tempfile
  with tempfile.TemporaryDirectory() as d,patch.object(a,'ACCEPT',Path(d)/'no_formal'),patch.object(a,'REV',Path(d)/'no_review'):
   with self.assertRaisesRegex(ValueError,'before reading timings'):a.admit()
 def test_no_source_review_blocks_before_journal(self):
  from unittest.mock import patch
  import tempfile
  with tempfile.TemporaryDirectory() as d,patch.object(a,'HERE',Path(d)/'no_source'),patch.object(a,'REV',Path(d)/'no_review'):
   with self.assertRaisesRegex(ValueError,'before reading timings'):a.admit_source()
