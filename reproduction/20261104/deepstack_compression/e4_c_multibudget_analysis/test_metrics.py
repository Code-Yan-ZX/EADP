import copy,unittest
from metrics import summarize,TASKS,KS,PROTOCOLS
class Tests(unittest.TestCase):
 def fixture(self):
  jobs=[];events=[]
  def add(task,group,q,k,prot,rep,phase):
   job={'phase':phase,'task':task,'qid':q,'group_id':group,'budget':k,'protocol':prot,'replicate':rep,'order':len(jobs)};r={'task':task,'qid':q,'group_id':group,'budget':k,'protocol':prot,'output_ids':[9],'generation_scalars':{},'min_new_tokens':None,'eos_ids':[9],'eos':True,'truncated':False,'boundaries':{'t0_before_image_read_ns':100,'first_token_selected_ns':100+k*1000+rep,'generation_complete_ns':200+k*2000+rep,'first_criteria_length':1,'criteria_calls':1,'clock_implementation':'clock_gettime(CLOCK_MONOTONIC)','explicit_sync_boundaries':['before_t0','first_token','generate_complete']}};raw={z:r[z] for z in ('task','qid','group_id','budget','protocol','output_ids','generation_scalars','min_new_tokens','eos_ids')};raw['generation_complete_ns']=r['boundaries']['generation_complete_ns'];jobs.append(job);events.extend([{'event':'job_started','job':job},{'event':'raw_generation','job':job,'raw':raw},{'event':'job_completed','job':job,'record':r}])
  for t in TASKS:
   for i in range(10):
    for k in KS:
     for prot in PROTOCOLS:add(t,'w'+str(i),str(i),k,prot,0,'warmup')
  for t,n in TASKS.items():
   for i in range(n):
    for k in KS:
     for prot in PROTOCOLS:
      for rep in range(5):add(t,'g'+str(i),str(i),k,prot,rep,'measure')
  return jobs,events
 def test_full_pipeline_and_paired_difference(self):
  jobs,events=self.fixture();r=summarize(jobs,events);self.assertEqual((r['images'],r['warmups'],r['measured']),(32,240,1280));x=r['tasks']['DocVQA_VAL']['natural']['128'];self.assertEqual(x['counts']['images'],11);self.assertAlmostEqual(x['ttft_ms']['paired_difference_vs_k256_ms']['median'],-.128);self.assertAlmostEqual(x['ttft_ms']['paired_ratio_vs_k256']['median'],128002/256002)
 def test_missing_duplicate_wrong_raw_or_time_rejected(self):
  for case in ('missing','duplicate','raw','clock','phase'):
   jobs,events=self.fixture()
   if case=='missing':events.pop()
   if case=='duplicate':jobs[-1]['replicate']=jobs[-2]['replicate']
   if case=='raw':events[-2]['raw']['output_ids']=[7]
   if case=='clock':events[-1]['record']['boundaries']['first_token_selected_ns']=100
   if case=='phase':jobs[-1]['phase']='unknown'
   with self.subTest(case=case),self.assertRaises((ValueError,KeyError)):summarize(jobs,events)
if __name__=='__main__':unittest.main()
