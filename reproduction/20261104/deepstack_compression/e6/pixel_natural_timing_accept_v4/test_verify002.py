import copy,unittest,hashlib,struct
from unittest.mock import patch
import verify002 as verify
class Tests(unittest.TestCase):
 def sample(self,arm='pixel_t0'):
  job={'arm':arm,'task':'a','qid':'1','group_id':'g','order':0,'block':0};ref={'processor':{'input_ids':{'shape':[1,2],'dtype':'torch.int64','sha256':hashlib.sha256(struct.pack('<qq',4,5)).hexdigest()}},'tokens':[6,9],'eos_ids':[9],'selected_count':3 if arm.startswith('pixel_') else 256}
  def x(event,**kw):return {'event':event,'job':job,**kw}
  def result(obs):
   return {'processor_verified':True,'tokens':[6,9],'eos_ids':[9],'recipe':verify.pilot_contract.RECIPE,'eos':True,'truncated':False,'output_tokens':2,'dynamic_k':ref['selected_count'] if obs else None,'count_observer_enabled':obs,
     'boundaries':{'t0_before_image_read_ns':10,'first_token_selected_ns':11,'generation_complete_ns':12,'first_criteria_length':3 if arm.startswith('pixel_') else 1,'time_namespace':'time:[1]','clock_implementation':'clock','criteria_calls':2,'explicit_sync_boundaries':['before_t0','first_token','generation_end']}}
  mem={'baseline':{'allocated_bytes':10,'reserved_bytes':20},'end':{'allocated_bytes':12,'reserved_bytes':22},'max_allocated_bytes':13,'max_reserved_bytes':25,'scope':'not NVML'}
  raw=[[4,5,6,9]] if arm.startswith('pixel_') else [[6,9]]
  rows=[x('model_loading'),x('model_ready',setup_wall_ns=1),x('started',phase='warmup'),x('raw_generation',phase='warmup',ids=raw),x('completed',phase='warmup',result=result(False),memory=mem),x('started',phase='measured'),x('raw_generation',phase='measured',ids=raw),x('completed',phase='measured',result=result(True),memory=mem),x('job_complete')]
  return job,rows,ref
 def test_pixel_and_eadp(self):
  for arm in ('pixel_t0','pixel_t045','eadp_k256'):
   j,r,ref=self.sample(arm);self.assertEqual(len(verify.verify_job(j,r,ref)),2)
 def test_missing_raw_or_incorrect_k(self):
  j,r,ref=self.sample();r.pop(3)
  with self.assertRaises(ValueError):verify.verify_job(j,r,ref)
  j,r,ref=self.sample();r[7]['result']['dynamic_k']=4
  with self.assertRaises(ValueError):verify.verify_job(j,r,ref)
 def test_wrong_token_boundary_and_allocator(self):
  j,r,ref=self.sample();r[7]['result']['boundaries']['first_criteria_length']=2
  with self.assertRaises(ValueError):verify.verify_job(j,r,ref)
  j,r,ref=self.sample();r[7]['memory']['max_allocated_bytes']=1
  with self.assertRaises(ValueError):verify.verify_job(j,r,ref)
 def test_source_mismatch_refused_before_real_data(self):
  with patch.object(verify,'read',return_value={'sha256':{'/frozen':'expected'}}),patch.object(verify,'sha',return_value='changed'):
   with self.assertRaisesRegex(ValueError,'verification source'):verify.accept()
 def test_missing_health_clock_records_unknown_without_dropping(self):
  start={'worker_pid':1,'snapshot_seconds':3600,'poll_seconds':5};term={'worker_pid':1,'worker_returncode':0,'reason':'completed','remaining_process_group':[]}
  snaps=[{'utc':'a'},{'event':'terminal_snapshot','utc':'b'}]
  records=[{'t0_ns':2,'first_token_ns':3,'generation_end_ns':4,'time_namespace':'t','clock_implementation':'clock_gettime(CLOCK_MONOTONIC)'}]
  result=verify.annotate_health(records,snaps,start,term)
  self.assertEqual(len(result),1);self.assertIsNone(result[0]['ttft_health_window_indices']);self.assertIn('unknown',result[0]['health_overlap_status'])
 def test_health_overlap_and_unknown_clock_fail_closed(self):
  start={'worker_pid':1,'snapshot_seconds':3600,'poll_seconds':5};terminal={'worker_pid':1,'worker_returncode':0,'reason':'completed','remaining_process_group':[]}
  snaps=[{'health_snapshot_window_monotonic_ns':[1,2],'time_namespace':'t'},{'event':'terminal_snapshot','health_snapshot_window_monotonic_ns':[20,21],'time_namespace':'t'}]
  records=[{'t0_ns':2,'first_token_ns':3,'generation_end_ns':4,'time_namespace':'t','clock_implementation':'clock_gettime(CLOCK_MONOTONIC)'}]
  x=verify.annotate_health(records,snaps,start,terminal);self.assertEqual(x[0]['ttft_health_window_indices'],[0]);self.assertEqual(x[0]['generation_health_window_indices'],[0])
  snaps[1]['time_namespace']='other'
  with self.assertRaises(ValueError):verify.annotate_health(records,snaps,start,terminal)

if __name__=='__main__':unittest.main(verbosity=2)
