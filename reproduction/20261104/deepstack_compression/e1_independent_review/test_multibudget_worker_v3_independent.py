import copy,unittest
import review_e4c_multibudget_worker_v3_result as v
NS='time:[1]';UTC='2026-11-05T00:00:00+00:00'
def case(protocol='natural',budget=256):
 job={'task':'DocVQA_VAL','qid':'1','group_id':'g','budget':budget,'protocol':protocol,'order':0,'phase':'measure','replicate':0};ids=[1]*31+[9] if protocol=='fixed32' else [9];ctrl={'max_new_tokens':32 if protocol=='fixed32' else 2048}
 rec={k:job[k] for k in ('task','qid','group_id','budget','protocol')};rec.update(output_ids=ids,answer='ok',generation_scalars=ctrl,min_new_tokens=32 if protocol=='fixed32' else None,eos_ids=[9],eos=True,truncated=False,selector_calls=0 if budget==1024 else 1,boundaries={'t0_before_image_read_ns':10,'first_token_selected_ns':20,'generation_complete_ns':30,'time_namespace':NS,'first_criteria_length':1,'criteria_calls':len(ids),'clock_implementation':'clock_gettime(CLOCK_MONOTONIC)','explicit_sync_boundaries':['before_t0','first_token','generate_complete']})
 raw={k:rec[k] for k in ('task','qid','group_id','budget','protocol','output_ids','generation_scalars','min_new_tokens','eos_ids')};raw['generation_complete_ns']=30
 return job,[{'event':'job_started','job':job},{'event':'raw_generation','job':job,'raw':raw},{'event':'job_completed','job':job,'record':rec}],copy.deepcopy(rec),copy.deepcopy(rec)
def observer():
 return ([{'event':'snapshot','index':0,'time_namespace':NS,'query_start_ns':1,'query_end_ns':2,'query_start_utc':UTC,'query_end_utc':UTC,'returncode':0,'stdout':''},{'event':'logging_complete','index':0,'time_namespace':NS,'query_start_ns':1,'first_fsync_end_ns':3,'first_fsync_end_utc':UTC},{'event':'watch_end','time_namespace':NS,'reason':'stop_marker','samples':1,'end_ns':4,'end_utc':UTC}],[{'time_namespace':NS,'health_snapshot_window_monotonic_ns':[2,3]}],[])
class Tests(unittest.TestCase):
 def test_both_protocols(self):
  for protocol in ('natural','fixed32'):
   job,events,panel,fixed=case(protocol);self.assertEqual(v.check_job(job,events,panel,fixed,lambda x:'ok',NS),(10,20,30))
 def test_tamper_rejected(self):
  for kind in ('missing','wrong_job','raw','reference','tokens','clock','namespace','decode','recipe'):
   job,events,panel,fixed=case('fixed32')
   if kind=='missing':events.pop()
   if kind=='wrong_job':events[0]['job']=dict(job,qid='2')
   if kind=='raw':events[1]['raw']['output_ids']=[9]
   if kind=='reference':fixed['answer']='other'
   if kind=='tokens':events[2]['record']['output_ids'][0]=9;events[1]['raw']['output_ids'][0]=9;fixed['output_ids'][0]=9
   if kind=='clock':events[2]['record']['boundaries']['first_token_selected_ns']=9
   if kind=='namespace':events[2]['record']['boundaries']['time_namespace']='other'
   if kind=='recipe':events[2]['record']['min_new_tokens']=31;events[1]['raw']['min_new_tokens']=31
   with self.subTest(kind=kind),self.assertRaises((ValueError,KeyError)):v.check_job(job,events,panel,fixed,lambda x:'bad' if kind=='decode' else 'ok',NS)
 def test_observation_missing_failed_or_utc_coverage(self):
  s,h,m=observer();self.assertTrue(v.validate_observer(s,h,m,NS,UTC,UTC)['manual_missing_is_unknown'])
  for kind in ('missing','failed','namespace','stop','manual','health','uncovered'):
   s,h,m=copy.deepcopy(observer());end=UTC
   if kind=='missing':s=[]
   if kind=='failed':s[0]['returncode']=1
   if kind=='namespace':s[1]['time_namespace']='other'
   if kind=='stop':s[-1]['reason']='duration_expired'
   if kind=='manual':m=[{'event':'manual_health_check','time_namespace':'other'}]
   if kind=='health':h=[]
   if kind=='uncovered':end='2026-11-05T00:00:01+00:00'
   with self.subTest(kind=kind),self.assertRaises((ValueError,KeyError)):v.validate_observer(s,h,m,NS,UTC,end)
 def test_rebuild_denominators_and_frozen_order(self):
  cohort={};identities=[]
  for task,q in zip(v.TASKS,v.QUOTAS):
   measured=[f'{task}-m{i}' for i in range(64)];warm=[f'{task}-w{i}' for i in range(10)];cohort[task]={'measured':measured,'warmup':warm};chosen=sorted(measured,key=lambda g:v.rank('pilot',task,g))[:q]
   identities.extend({'task':task,'qid':g,'group_id':g} for g in chosen+warm)
  groups,jobs=v.rebuild(cohort,identities);self.assertEqual((len(identities),len(jobs),sum(x['phase']=='measure' for x in jobs)),(62,1520,1280));self.assertEqual([x['order'] for x in jobs],list(range(1520)))
if __name__=='__main__':unittest.main()
