import unittest
from unittest.mock import patch
import report
class Tests(unittest.TestCase):
 def sample(self):
  jobs=[];records=[];annotations=[]
  for i in range(3):
   for arm in ('eadp_k256','pixel_t0','pixel_t045'):
    j={'task':str(i),'qid':str(i),'group_id':'g'+str(i),'arm':arm};jobs.append(j)
    def ev(name,**kw):return {'event':name,'job':j,**kw}
    b={'t0_before_image_read_ns':10,'first_token_selected_ns':20,'generation_complete_ns':30}
    v={'boundaries':b,'dynamic_k':256,'tokens':[9],'eos':True,'truncated':False}
    m={'max_allocated_bytes':2**30,'max_reserved_bytes':2**30}
    records.append([ev('model_loading'),ev('model_ready',setup_wall_ns=10**9),ev('started'),ev('raw_generation'),ev('completed'),ev('started'),ev('raw_generation'),ev('completed',result=v,memory=m),ev('job_complete')])
    for phase in ('warmup','measured'):annotations.append({'task':j['task'],'qid':j['qid'],'arm':arm,'phase':phase,'t0_ns':10,'first_token_ns':20,'generation_end_ns':30,'dynamic_k':256 if phase=='measured' else None,'health_overlap_status':'unknown'})
  return jobs,records,annotations
 def test_complete_paired_and_scope(self):
  rows=report.summarize(*self.sample());self.assertEqual(len(rows),9);self.assertTrue(all(x['ttft_relative_to_eadp']==1 for x in rows));text=report.render(rows,'a','b','c');self.assertIn('不是 NVML',text);self.assertIn('直接重叠未知',text)
 def test_missing_or_damaged_fails(self):
  jobs,records,ann=self.sample();records.pop()
  with self.assertRaises(ValueError):report.summarize(jobs,records,ann)
  jobs,records,ann=self.sample();ann[-1]['health_overlap_status']='no overlap'
  with self.assertRaises(ValueError):report.summarize(jobs,records,ann)
 def test_bad_source_refused_before_data(self):
  with patch.object(report,'sha',return_value='bad'):
   with self.assertRaisesRegex(ValueError,'source/proof'):report.admission({'sha256':{'/test':'good'}})
if __name__=='__main__':unittest.main(verbosity=2)
