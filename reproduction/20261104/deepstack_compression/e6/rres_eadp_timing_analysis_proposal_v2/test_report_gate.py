import unittest
import report
class Tests(unittest.TestCase):
 def test_real_entry_execute_rejects_before_any_result_file_read(self):
  class G:
   def admit(self,plan):raise ValueError('missing independent acceptance')
   def verify_files(self,receipt):raise AssertionError('file hashing before admission')
  touched=[]
  def read(path):touched.append(path);raise AssertionError('timing opened before admission')
  with self.assertRaisesRegex(ValueError,'missing independent'):report.execute({},G(),read,read)
  self.assertEqual(touched,[])
 def test_full_synthetic_entry_12_processes_and_380_requests(self):
  import copy,sys
  from test_metrics import fixtures
  from test_report_core import journals,observer
  import numpy
  schedule,records=fixtures();js=journals(schedule,records)
  event=observer()
  for v in event:v['monotonic_clock']='mono'
  p={'numpy_version':numpy.__version__,'schedule_path':'schedule','journal_paths':['j'+str(i) for i in range(12)],
     'worker_completion_paths':['c'+str(i) for i in range(12)],'observer_path':'observer','run_health_path':'run_health',
     'observer_health_path':'observer_health','automatic_health_windows_snapshot':'automatic','automatic_health_events_snapshot':'events','automatic_health_snapshot_metadata':'metadata'}
  objects={'schedule':schedule,'metadata':{'status':'captured'},**{'c'+str(i):{'time_namespace':'n','monotonic_clock':'mono','perf_counter_clock':'mono'} for i in range(12)}}
  line_map={**{'j'+str(i):js[i] for i in range(12)},'observer':event,'run_health':[{'utc':'old schema'}],'observer_health':[],'automatic':[],'events':[{'event':'health_failed','error':'timeout'}]}
  class G:
   def admit(self,plan):return {'formal_sha256':'formal','independent_sha256':'independent','files_sha256':{}}
   def verify_files(self,receipt):return True
   def sha(self,path):return 'sha'
  saved={name:sys.modules.pop(name,None) for name in ('metrics','windows','normalize','report_core')}
  try:
   result=report.execute(p,G(),lambda name:copy.deepcopy(objects[name]),lambda name:copy.deepcopy(line_map[name]))
   self.assertEqual(result['result']['counts']['measured'],320);self.assertEqual(len(result['result']['per_job']),380)
   self.assertEqual(result['result']['automatic_health_audit']['failure_count'],1)
   self.assertEqual(result['result']['per_job'][0]['windows']['run_supervisor_health']['recorded_overlap'],'unknown')
  finally:
   for name,mod in saved.items():
    sys.modules.pop(name,None)
    if mod is not None:sys.modules[name]=mod
