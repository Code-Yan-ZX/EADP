import importlib.util,json,sys,unittest
from collections import Counter
from pathlib import Path
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
 def test_actual_launcher_import(self):
  import subprocess
  out=subprocess.run([sys.executable,str(HERE/'launch.py'),'--help'],capture_output=True,text=True,timeout=120)
  self.assertEqual(out.returncode,0,out.stderr)
  self.assertIn('--stage',out.stdout)
if __name__=='__main__':unittest.main(verbosity=2)
