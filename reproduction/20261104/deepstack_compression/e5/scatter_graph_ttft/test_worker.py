import importlib.util,json,os,sys,unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ttft_worker',HERE/'worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
class Schedule(unittest.TestCase):
 def test_schedule_is_balanced_paired_and_repeatable(self):
  anchors=[{'task':t,'qid':str(i)} for i,t in enumerate(('Doc','Text','AI'))]
  jobs=w.make_schedule(anchors);self.assertEqual(jobs,w.make_schedule(anchors));self.assertEqual(len(jobs),36)
  counts=Counter((x['phase'],x['task'],x['method']) for x in jobs)
  self.assertTrue(all(counts[(phase,t,m)]==count for phase,count in [('warmup',1),('measured',5)] for t in ('Doc','Text','AI') for m in ('original','graph')))
  for i in range(0,len(jobs),2):
   a,b=jobs[i:i+2];self.assertEqual((a['phase'],a['repeat'],a['task'],a['qid']),(b['phase'],b['repeat'],b['task'],b['qid']))
   self.assertEqual({a['method'],b['method']},{'original','graph'})
 def test_pair_mismatch_and_wrong_identity(self):
  job={'phase':'measured','repeat':0,'task':'Doc','qid':'1','protocol':'fixed32','method':'original'}
  a={'job':job,'ordered_indices':[1,2],'quota':[2],'result':{'output_ids':[3,4]}}
  b={'job':dict(job,method='graph'),'ordered_indices':[1,2],'quota':[2],'result':{'output_ids':[3,4]}}
  self.assertTrue(w.compare_pair(a,b)['eligible_for_equal_output_timing'])
  b['ordered_indices']=[2,1];self.assertFalse(w.compare_pair(a,b)['eligible_for_equal_output_timing'])
  b['ordered_indices']=[1,2];b['result']['output_ids']=[9];self.assertFalse(w.compare_pair(a,b)['eligible_for_equal_output_timing'])
  b['result']['output_ids']=[3,4];b['quota']=[1];self.assertFalse(w.compare_pair(a,b)['eligible_for_equal_output_timing'])
  b['job']['qid']='2'
  with self.assertRaises(ValueError):w.compare_pair(a,b)
 def test_wrong_import_path_is_rejected(self):
  class M:__file__='/wrong/runner.py'
  with self.assertRaises(ValueError):w.validate_import_paths({M:'/expected/runner.py'})
 def test_raw_is_emitted_before_return(self):
  class Result:
   def detach(self):return self
   def cpu(self):return self
   def tolist(self):return [[1,2]]
  calls=[];result=Result()
  call=w.durable_generation(lambda:result,lambda ids:calls.append(ids))
  self.assertIs(call(),result);self.assertEqual(calls,[[[1,2]]])
 def test_cache_probe_writes_and_removes_file(self):
  import tempfile
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'cache';self.assertTrue(w.probe_cache(path)['writable']);self.assertEqual(list(path.iterdir()),[])
 def test_actual_launcher_import(self):
  import subprocess
  out=subprocess.run([sys.executable,str(HERE/'launch.py'),'--help'],capture_output=True,text=True,timeout=120)
  self.assertEqual(out.returncode,0,out.stderr)
  self.assertIn('--stage',out.stdout)
 def test_actual_launcher_rejects_component_before_gpu_query(self):
  spec=importlib.util.spec_from_file_location('scatter_ttft_launcher_test',HERE/'launch.py');launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
  old=Path.read_text
  def fake_read(path,*args,**kwargs):
   if path==HERE/'plan001.json':return json.dumps({'budget_source_sha256':'x'})
   if path==w.BASE/'initial_plan.json':return json.dumps({'budget_start_unix':0,'budget_end_unix':10000000000})
   if path==HERE/'monitor_smoke001/started.json':return json.dumps({'worker_pid':os.getpid(),'timeout_seconds':3720,'grace_seconds':90,
       'stale_seconds':1200,'progress_paths':[str(HERE/'run_smoke001/observations/progress.jsonl')]})
   return old(path,*args,**kwargs)
  with patch.object(sys,'argv',['launch.py','--stage','smoke']),patch.object(Path,'read_text',fake_read),\
       patch.object(launch,'preflight'),patch.object(launch,'sha',return_value='x'),patch.object(launch,'validate_budget_window'),\
       patch.object(launch.component_gate,'require',side_effect=ValueError('no component proof')),patch.object(launch,'load') as load:
   with self.assertRaisesRegex(ValueError,'no component proof'):launch.main()
   load.assert_not_called()
if __name__=='__main__':unittest.main(verbosity=2)
