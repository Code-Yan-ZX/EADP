import copy,unittest
import numpy as np
from analysis_core import TASKS,MODES,PROTOCOLS,analyze_task,build_records,validate_windows,summarize_windows
class AnalysisTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.inputs=[];cls.jobs=[]
  for task in TASKS:
   for i in range(128):cls.inputs.append({'task':task,'qid':str(i),'group_id':task+str(i),'split':'DEV','image_sha256':'a'*64,'message_sha256':'b'*64})
   for phase,indices in [('warmup',range(64,74)),('measure',range(64))]:
    for i in indices:
     for protocol in PROTOCOLS:
      for rep in (range(5) if phase=='measure' else [0]):
       for mode in MODES:cls.jobs.append({'order':len(cls.jobs),'task':task,'qid':str(i),'group_id':task+str(i),'phase':phase,'protocol':protocol,'replicate':rep,'mode':mode})
  cls.events=[];cls.formal=[];cls.manual=[]
  for j in cls.jobs:
   b={'t0_before_image_read_ns':100,'first_token_selected_ns':200,'generation_complete_ns':300,'time_namespace':'time:1','clock_implementation':'clock_gettime(CLOCK_MONOTONIC)','first_criteria_length':1,'explicit_sync_boundaries':['before_t0','first_token','generation_end']}
   v={'mode':j['mode'],'protocol':j['protocol'],'boundaries':b,'output_tokens':32 if j['protocol']=='fixed32' else 2,'eos':True,'truncated':False}
   g={'status':'generated','job':j,'result':v};done={'status':'completed','job':j};cls.events.extend([g,done]);cls.formal.extend([{**g,'monitor_overlap':False,'monitor_snapshot_indices':[]},done]);cls.manual.append({'job':j,'manual_health_overlap':False,'manual_health_indices':[],'manual_nvml_overlap':False,'manual_nvml_indices':[]})
  cls.schedule={'schedule':cls.jobs};cls.windows=validate_windows([{'time_namespace':'time:1','health_snapshot_window_monotonic_ns':[400,500]}],[{'time_namespace':'time:1','manual_health_window_monotonic_ns':[600,700],'nvml_window_monotonic_ns':None,'gpu_check':'no additional NVML query'}],'time:1')
 def test_complete_inventory(self):
  result=build_records(self.schedule,self.inputs,self.events,self.formal,self.manual,'time:1',self.windows);self.assertEqual(len(result),5940);self.assertEqual(sum(r['phase']=='measure' for r in result),5760)
 def test_changed_qid_or_bad_time_or_unknown_window_fails(self):
  for kind in ('qid','time','window','order'):
   events=list(self.events);manual=list(self.manual);j=copy.deepcopy(events[0]);m=copy.deepcopy(manual[0])
   if kind=='qid':j['job']['qid']='bad'
   if kind=='time':j['result']['boundaries']['first_token_selected_ns']=100
   if kind=='window':m['manual_health_overlap']=None
   if kind=='order':j['job']['order']=9
   events[0]=j;manual[0]=m
   with self.subTest(kind=kind),self.assertRaises(ValueError):build_records(self.schedule,self.inputs,events,self.formal,manual,'time:1',self.windows)
 def test_multiple_windows_boundaries_and_per_stratum_counts(self):
  windows=validate_windows([{'time_namespace':'time:1','health_snapshot_window_monotonic_ns':[10,20]},{'time_namespace':'time:1','health_snapshot_window_monotonic_ns':[30,40]}],[{'time_namespace':'time:1','manual_health_window_monotonic_ns':[15,35],'nvml_window_monotonic_ns':[20,25]},{'time_namespace':'time:1','manual_health_window_monotonic_ns':[50,60],'nvml_window_monotonic_ns':None,'gpu_check':'no additional NVML query'}],'time:1')
  row={'phase':'measure','task':TASKS[0],'mode':MODES[0],'protocol':PROTOCOLS[0],'monitor_snapshot_indices':[0,1],'manual_health_indices':[0],'manual_nvml_indices':[0]}
  out=summarize_windows([row],windows);self.assertEqual(len(out),5);self.assertEqual([x['n_overlap'] for x in out],[1,1,1,1,0]);self.assertTrue(all(len(x['counts'])==36 for x in out))
 def test_forged_window_index_and_missing_nvml_status_rejected(self):
  formal=list(self.formal);formal[0]={**formal[0],'monitor_overlap':True,'monitor_snapshot_indices':[0]}
  with self.assertRaisesRegex(ValueError,'annotation mismatch'):build_records(self.schedule,self.inputs,self.events,formal,self.manual,'time:1',self.windows)
  with self.assertRaises(ValueError):validate_windows([{'time_namespace':'time:1','health_snapshot_window_monotonic_ns':[1,2]}],[{'time_namespace':'time:1','manual_health_window_monotonic_ns':[1,2]}],'time:1')
 def test_pooled_quantile_paired_bootstrap_and_ratios(self):
  rows=[]
  for g,base in [('g0',1.),('g1',3.)]:
   for protocol in PROTOCOLS:
    for mode,scale in [('C-ON',1.),('C-DROP',2.),('C-SKIP',3.)]:
     for rep in range(5):rows.append({'group_id':g,'phase':'measure','protocol':protocol,'mode':mode,'replicate':rep,'ttft_ms':base*scale,'end_to_end_ms':base*scale*2,'output_tokens':32,'eos':True,'truncated':False})
  draws=np.array([[0,0],[0,1],[1,0],[1,1]])
  x=analyze_task(rows,draws);self.assertEqual(x,analyze_task(rows,draws));m=x['protocols']['natural']['metrics']['ttft_ms_median'];self.assertEqual(m['mode_quantiles_ms']['C-ON'],2.)
  c=m['paired_contrasts']['C-DROP_minus_C-ON'];self.assertEqual(c['ratio'],2.);self.assertEqual(c['ratio_95_ci'],[2.,2.]);self.assertEqual(c['relative_change_pct'],100.);self.assertEqual(c['difference_ms'],2.)
  self.assertEqual(x['n_image_groups'],2);self.assertEqual(x['measurements_per_mode_protocol'],10)
 def test_missing_or_nonpositive_time_fails(self):
  rows=[{'group_id':'g','phase':'measure','protocol':p,'mode':m,'replicate':i,'ttft_ms':1.,'end_to_end_ms':1.,'output_tokens':1,'eos':True,'truncated':False} for p in PROTOCOLS for m in MODES for i in range(5)]
  with self.assertRaises(ValueError):analyze_task(rows[:-1],np.array([[0]]))
  rows[0]['ttft_ms']=0.
  with self.assertRaises(ValueError):analyze_task(rows,np.array([[0]]))
if __name__=='__main__':unittest.main()
