import unittest
import coverage
class Tests(unittest.TestCase):
 def test_direct_overlaps_and_incomplete_coverage(self):
  jobs=[{'order':i} for i in range(24)];records=[{'start_monotonic_ns':20,'first_token_monotonic_ns':30,'boundaries':{'time_namespace':'ns'}} for _ in jobs]
  events=[{'event':'snapshot','index':0,'time_namespace':'ns','ps':{'query_start_ns':10,'query_end_ns':15,'returncode':0},'docker':{'query_start_ns':20,'query_end_ns':25,'returncode':0}},{'event':'logging_complete','index':0,'time_namespace':'ns','first_fsync_end_ns':28},{'event':'observer_end','time_namespace':'ns','end_ns':40}]
  r=coverage.classify(jobs,records,events);self.assertEqual(r['observer_uptime'],'covered');self.assertEqual(r['annotations'][0]['windows']['docker']['overlap_indices'],[0]);self.assertEqual(r['annotations'][0]['windows']['ps']['overlap_indices'],[])
  events[-1]['time_namespace']='other'
  with self.assertRaises(ValueError):coverage.classify(jobs,records,events)
if __name__=='__main__':unittest.main()
