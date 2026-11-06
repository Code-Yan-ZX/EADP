import copy,unittest
from contract import TASKS,MODES,PROTOCOLS,make_schedule,validate_schedule,validate_measurement,validate_process_memory_summary
class ScheduleTests(unittest.TestCase):
 def setUp(self):self.rows=[{'task':task,'qid':str(i),'group_id':task+str(i),'split':'DEV'} for task in TASKS for i in range(128)]
 def test_exact_counts_group_blocks_and_determinism(self):
  a=make_schedule(self.rows);self.assertEqual(a,make_schedule(list(reversed(self.rows))));self.assertTrue(validate_schedule(a));self.assertEqual(len(a['schedule']),5940)
  self.assertEqual(sum(r['phase']=='measure' for r in a['schedule']),5760);self.assertEqual(sum(r['phase']=='warmup' for r in a['schedule']),180)
 def test_invalid_scope_duplicate_missing_or_mixed_groups_rejected(self):
  bad=copy.deepcopy(self.rows);bad[0]['split']='FINAL'
  with self.assertRaises(ValueError):make_schedule(bad)
  bad=copy.deepcopy(self.rows);bad[1]['group_id']=bad[0]['group_id']
  with self.assertRaises(ValueError):make_schedule(bad)
  a=make_schedule(self.rows);a['schedule'].pop()
  with self.assertRaises(ValueError):validate_schedule(a)
  a=make_schedule(self.rows);a['cohort'][TASKS[0]]['warmup'][0]=a['cohort'][TASKS[0]]['measured'][0]
  with self.assertRaises(ValueError):validate_schedule(a)
 def test_excluded_old_anchors_are_not_sampled(self):
  old={self.rows[0]['group_id'],self.rows[130]['group_id']};obj=make_schedule(self.rows,old)
  self.assertFalse(old & {r['group_id'] for r in obj['schedule']})
class MicrobenchScheduleTests(unittest.TestCase):
 def test_32_images_20_repetitions_per_component(self):
  from collections import Counter
  from microbench_contract import make_microbench_schedule,COMPONENTS
  rows=[{'task':task,'qid':str(i),'group_id':task+str(i),'split':'DEV'} for task in TASKS for i in range(128)]
  timing=make_schedule(rows);plan=make_microbench_schedule(timing)
  self.assertEqual(len(plan['images']),32);self.assertEqual(len(plan['jobs']),1920);self.assertEqual(plan,make_microbench_schedule(timing))
  self.assertEqual(Counter(x['component'] for x in plan['jobs']),{x:640 for x in COMPONENTS})
  self.assertEqual(Counter(task for task,_ in plan['images']),{'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10})
  self.assertTrue(all(g in timing['cohort'][task]['measured'] for task,g in plan['images']))
class MeasurementTests(unittest.TestCase):
 def setUp(self):self.record={'mode':'C-ON','protocol':'natural','batch_size':1,'phase':'measure','profiler_enabled':False,'tensor_hashing_in_window':False,'per_layer_sync_in_window':False,'fresh_process_per_config':True,'measurement_role':'memory_isolated','exclusive_gpu':True,'t0_before_image_read_ns':1000000000,'logits_ready_ns':None,'first_token_selected_ns':1005000000,'generation_complete_ns':1012000000,'token_selection_materialized':True,'boundary_sync_complete':True,'output_tokens':6,'scored':False,'memory_only':{'allocated_peak_bytes':100,'reserved_peak_bytes':150,'kv_cache_bytes':20}}
 def test_ttft_includes_first_token_and_ns_to_ms(self):
  x=validate_measurement(self.record);self.assertEqual(x,{'ttft_ms':5.0,'first_logits_ms':None,'end_to_end_ms':12.0})
 def test_rejects_protocol_contamination_and_wrong_times(self):
  for key,value in [('profiler_enabled',True),('tensor_hashing_in_window',True),('per_layer_sync_in_window',True),('fresh_process_per_config',False),('exclusive_gpu',False),('first_token_selected_ns',999999999),('token_selection_materialized',False),('boundary_sync_complete',False),('output_tokens',0),('logits_ready_ns',1004000000)]:
   bad=dict(self.record);bad[key]=value
   with self.assertRaises(ValueError,msg=key):validate_measurement(bad)
 def test_nvml_peak_belongs_to_isolated_process_not_sample(self):
  summary={'type':'summary','pid':123,'start_ticks':456,'gpu_uuid':'GPU-a','samples':8,'target_observed':True,'other_gpu_process_observed':False,'nvml_process_sampled_peak_bytes':1024}
  out=validate_process_memory_summary(summary,'C-ON',123,456,'GPU-a');self.assertEqual(out['scope'],'whole isolated process; MiB-rounded sampled maximum')
  for key,value in [('pid',124),('start_ticks',457),('gpu_uuid','GPU-b'),('samples',0),('target_observed',False),('other_gpu_process_observed',True),('nvml_process_sampled_peak_bytes',None)]:
   bad=dict(summary);bad[key]=value
   with self.assertRaises(ValueError,msg=key):validate_process_memory_summary(bad,'C-ON',123,456,'GPU-a')
 def test_bad_memory_peak_rejected(self):
  bad=copy.deepcopy(self.record);bad['memory_only']['reserved_peak_bytes']=50
  with self.assertRaises(ValueError):validate_measurement(bad)
 def test_fixed32_must_be_exact_and_never_scored(self):
  bad=dict(self.record,protocol='fixed32',output_tokens=31)
  with self.assertRaises(ValueError):validate_measurement(bad)
  bad.update(output_tokens=32,scored=True)
  with self.assertRaises(ValueError):validate_measurement(bad)
  bad['scored']=False;validate_measurement(bad)
if __name__=='__main__':unittest.main()
