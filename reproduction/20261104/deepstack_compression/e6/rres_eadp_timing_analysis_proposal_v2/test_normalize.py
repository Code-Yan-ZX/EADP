import copy,unittest
import normalize
class Tests(unittest.TestCase):
 def test_journal_full_sequence_and_failure(self):
  p={'arm':'eadp','requests':[{'qid':'1'},{'qid':'2'}]};j=[]
  for i,r in enumerate(p['requests']):j.extend([{'event':'started','index':i,'request':r},{'event':'completed','index':i,'request':r,'arm':'eadp'}])
  self.assertEqual(len(normalize.completed_records(p,j)),2)
  bad=copy.deepcopy(j);bad[3]['event']='failed'
  with self.assertRaises(ValueError):normalize.completed_records(p,bad)
  with self.assertRaises(ValueError):normalize.completed_records(p,j[:-1])
 def test_observer_ps_docker_fsync(self):
  j=[{'event':'snapshot','index':0,'time_namespace':'n','ps':{'query_start_ns':1,'query_end_ns':2,'returncode':0},'docker':{'query_start_ns':3,'query_end_ns':4,'returncode':0}}, {'event':'logging_complete','index':0,'time_namespace':'n','first_fsync_end_ns':5},{'event':'observer_end','time_namespace':'n','end_ns':20,'samples':1,'reason':'stop'}]
  x=normalize.observer(j,'mono');self.assertEqual(x['ps_query']['windows'],[[1,2]]);self.assertEqual(x['docker_query']['windows'],[[3,4]]);self.assertEqual(x['first_fsync']['windows'],[[4,5]])
  bad=copy.deepcopy(j);bad[0]['docker']['returncode']=1;self.assertIsNone(normalize.observer(bad,'mono')['ps_query']['coverage_end_ns'])
  bad=copy.deepcopy(j);bad[-1]['reason']='duration_expired'
  with self.assertRaises(ValueError):normalize.observer(bad,'mono')
 def test_health_overlap_only_without_coverage(self):
  x=normalize.supervisor_health([{'time_namespace':'n','health_snapshot_window_monotonic_ns':[10,20]}],'mono');self.assertIsNone(x['coverage_start_ns'])
  self.assertIsNone(normalize.supervisor_health([{'utc':'2026-11-06T06:30:29Z','gpu':{'returncode':0}}],'mono'))
  audit=normalize.automatic_health_audit([{'event':'monitor_started'},{'event':'health_failed','error':'timeout'},{'event':'health_completed','returncode':1}],{'status':'captured'})
  self.assertEqual(audit['failure_count'],2);self.assertEqual(audit['failure_overlap'],'unknown');self.assertEqual(audit['snapshot_metadata']['status'],'captured')
  x=normalize.automatic_health([{'event':'manual_health_check','time_namespace':'n','start_monotonic_ns':11,'end_monotonic_ns':12}],'mono');self.assertIsNone(x['coverage_end_ns'])
  self.assertIsNone(normalize.automatic_health([],'mono'))
