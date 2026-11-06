import copy,unittest
import verify
class Tok:
 def decode(self,ids,**kw):return 'answer'
class Tests(unittest.TestCase):
 def sample(self,protocol='natural'):
  job={'task':'D','qid':'1','group_id':'g','budget':128,'protocol':protocol};ids=[1,9] if protocol=='natural' else [1]*32
  scalars={'do_sample':False,'max_new_tokens':2048 if protocol=='natural' else 32,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0}
  rec={'task':'D','qid':'1','group_id':'g','budget':128,'protocol':protocol,'output_ids':ids,'answer':'answer','generation_scalars':scalars,'min_new_tokens':None if protocol=='natural' else 32,'eos_ids':[9],'eos':protocol=='natural','truncated':False,'selector_calls':1,'boundaries':{'t0_before_image_read_ns':10,'first_token_selected_ns':20,'generation_complete_ns':30,'first_criteria_length':1,'criteria_calls':len(ids),'clock_implementation':'clock_gettime(CLOCK_MONOTONIC)','explicit_sync_boundaries':['before_t0','first_token','generate_complete'],'time_namespace':'time:[1]'}}
  raw={k:rec[k] for k in ('task','qid','group_id','budget','protocol','output_ids','generation_scalars','min_new_tokens','eos_ids')};raw['generation_complete_ns']=30
  events=[{'event':'job_started','job':job},{'event':'raw_generation','job':job,'raw':raw},{'event':'job_completed','job':job,'record':rec}]
  return job,rec,events
 def test_natural_and_fixed_reference(self):
  for protocol in ('natural','fixed32'):
   job,rec,events=self.sample(protocol);self.assertEqual(verify.job_events(events,job,rec,rec,Tok())[0],[10,20,30])
 def test_zero_time_or_raw_late_mismatch_is_rejected(self):
  job,rec,events=self.sample();bad=copy.deepcopy(events);bad[-1]['record']['boundaries']['first_token_selected_ns']=10
  with self.assertRaisesRegex(ValueError,'boundary'):verify.job_events(bad,job,rec,rec,Tok())
  bad=copy.deepcopy(events);bad[1]['raw']['generation_complete_ns']=31
  with self.assertRaisesRegex(ValueError,'boundary'):verify.job_events(bad,job,rec,rec,Tok())
 def test_common_wrong_job_identity_rejected(self):
  job,rec,events=self.sample();bad=copy.deepcopy(events);bad[1]['raw']['qid']='2';bad[2]['record']['qid']='2'
  with self.assertRaisesRegex(ValueError,'job identity'):verify.job_events(bad,job,rec,rec,Tok())
 def test_wrong_fixed_reference_and_early_eos_are_rejected(self):
  job,rec,events=self.sample('fixed32');ref=copy.deepcopy(rec);ref['output_ids']=[2]*32
  with self.assertRaisesRegex(ValueError,'reference'):verify.job_events(events,job,rec,ref,Tok())
  bad=copy.deepcopy(events);bad[1]['raw']['output_ids'][0]=9;bad[2]['record']['output_ids'][0]=9;ref=copy.deepcopy(bad[2]['record'])
  with self.assertRaisesRegex(ValueError,'reference'):verify.job_events(bad,job,ref,ref,Tok())
if __name__=='__main__':unittest.main()
