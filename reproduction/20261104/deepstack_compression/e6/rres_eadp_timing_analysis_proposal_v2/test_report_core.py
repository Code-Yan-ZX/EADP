import unittest
import report_core
from test_metrics import fixtures

def journals(schedule,records):
 out=[]
 for process,recs in zip(schedule['processes'],records):
  events=[]
  for rec in recs:events.extend([{'event':'started','index':rec['index'],'request':rec['request']},rec])
  out.append(events)
 return out

def observer():
 return [{'event':'snapshot','index':0,'time_namespace':'n','ps':{'query_start_ns':1,'query_end_ns':2,'returncode':0},'docker':{'query_start_ns':3,'query_end_ns':4,'returncode':0}}, {'event':'logging_complete','index':0,'time_namespace':'n','first_fsync_end_ns':5},{'event':'observer_end','time_namespace':'n','end_ns':999999999999,'samples':1,'reason':'stop'}]
class Tests(unittest.TestCase):
 def test_all_jobs_are_annotated_and_clock_mismatch_unknown(self):
  s,r=fixtures();clocks=[{'time_namespace':'n','monotonic_clock':'mono','perf_counter_clock':'mono'} for _ in range(12)]
  x=report_core.analyze(s,journals(s,r),clocks,observer(),'mono',[],[],[])
  self.assertEqual(len(x['per_job']),380);self.assertEqual(len(x['per_image']),32)
  self.assertFalse(x['per_job'][0]['windows']['observer_ps_query']['recorded_overlap'])
  self.assertEqual(x['per_job'][0]['windows']['manual_check']['recorded_overlap'],'unknown')
  clocks[0]['perf_counter_clock']='different';y=report_core.analyze(s,journals(s,r),clocks,observer(),'mono',[],[],[])
  self.assertEqual(y['per_job'][0]['windows']['observer_docker_query']['recorded_overlap'],'unknown')
 def test_missing_journal_event_cannot_form_report(self):
  s,r=fixtures();j=journals(s,r);j[0].pop()
  with self.assertRaises(ValueError):report_core.analyze(s,j,[{'time_namespace':'n'}]*12,observer(),'mono',[],[],[])
