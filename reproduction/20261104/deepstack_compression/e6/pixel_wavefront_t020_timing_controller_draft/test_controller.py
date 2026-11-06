import json,tempfile,unittest
from pathlib import Path
import controller
class Tests(unittest.TestCase):
 def test_all_and_failed_job_remains_visible(self):
  jobs=[{'order':i,'task':'t','qid':'1','arm':'author','phase':'warmup' if i<6 else 'measurement'} for i in range(24)];refs={('t','1','author'):{}}
  def runner(j):return {'container_id':format(j['order']+1,'064x'),'returncode':0,'container_exit_code':0,'container_stopped_verified':True,'timed_out':False,'interrupted':None,'record':{'ok':True}}
  with tempfile.TemporaryDirectory() as d:
   journal=Path(d)/'all';result=controller.execute(jobs,refs,journal,runner,lambda j,r,ref:{'ns':1});self.assertEqual(len(result),24);self.assertEqual(len(journal.read_text().splitlines()),49)
   def fail(j):
    x=runner(j)
    if j['order']==5:x['returncode']=1
    return x
   journal=Path(d)/'failed'
   with self.assertRaises(ValueError):controller.execute(jobs,refs,journal,fail,lambda *a:{})
   lines=[json.loads(x) for x in journal.read_text().splitlines()];self.assertEqual(lines[-1]['event'],'failed');self.assertEqual(lines[-1]['job']['order'],5);self.assertNotIn('all_completed',[x['event'] for x in lines])
if __name__=='__main__':unittest.main()
