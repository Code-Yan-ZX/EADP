import unittest
import analysis
class AnalysisTest(unittest.TestCase):
 def fixtures(self):
  jobs=[{'order':i,'arm':'fake'} for i in range(9)];summaries=[{'job':j,'status':'passed','stage_summary':{'unknown':None},'allocator_model_load':{},'allocator_request':{}} for j in jobs];records=[{'job':j,'identity':{'selected_count':i+1,'prompt_length':3,'generated_length':2},'sample_count':4,'stage_summary':{'unknown':None},'allocator_model_load':{},'allocator_request':{},'container_id':str(i),'host_pid':i+1,'gpu_uuid':'g'} for i,j in enumerate(jobs)];return jobs,summaries,{'results':records}
 def test_all_nine_and_unknowns_preserved(self):
  j,s,f=self.fixtures();r=analysis.rows(j,f,lambda i:s[i]);self.assertEqual(len(r),9);self.assertIsNone(r[0]['stage_summary']['unknown']);self.assertEqual(r[-1]['selected_count'],9)
 def test_incomplete_or_swapped_job_rejected(self):
  j,s,f=self.fixtures()
  with self.assertRaises(ValueError):analysis.rows(j,f|{'results':f['results'][:-1]},lambda i:s[i])
  s[1]['job']=j[2]
  with self.assertRaises(ValueError):analysis.rows(j,f,lambda i:s[i])
if __name__=='__main__':unittest.main()
