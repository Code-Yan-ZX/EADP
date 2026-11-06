import unittest,tempfile,json
from pathlib import Path
import pipeline,window_overlap
class Tests(unittest.TestCase):
 def test_24_results_keep_background_unknown_and_exact_jobs(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);jobs=[];refs={}
   for i in range(24):
    j={'order':i,'task':'t','qid':str(i),'arm':'a'};jobs.append(j);refs[('t',str(i),'a')]={};d=root/('job_%03d'%i);d.mkdir()
    (d/'launch.json').write_text('{}');(d/'exit.json').write_text(json.dumps({'container_identity':{'container_id':str(i)}}));(d/'record.json').write_text(json.dumps({'record':{'start_monotonic_ns':10,'first_token_monotonic_ns':20,'boundaries':{'time_namespace':'ns'}}}));(d/'raw.json').write_text('{}')
   kw=dict(plan={'jobs':jobs},jobs=jobs,design_jobs=jobs,references=refs,run_dir=root,core=lambda *a:{'status':'passed','jobs':24},job_check=lambda *a:None,raw_check=lambda *a:'sha',docker_check=lambda *a:{'checked':True},inspect=lambda cid:{},author={},builder=lambda *a:None,worker_path='worker',monitoring={'supervisor_health':None,'manual_health':None,'nvml':None,'external_watcher':None},window_classify=window_overlap.classify)
   r=pipeline.verify(**kw);self.assertEqual(len(r['docker_inspections']),24);self.assertEqual(r['background_annotations'][0]['background']['supervisor_health']['status'],'unknown');self.assertTrue(r['all_retained'])
   kw['design_jobs']=jobs[:-1]
   with self.assertRaises(ValueError):pipeline.verify(**kw)
if __name__=='__main__':unittest.main()
