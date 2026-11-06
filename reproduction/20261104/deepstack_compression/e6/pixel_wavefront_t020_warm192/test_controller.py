import tempfile,unittest
from pathlib import Path
import controller
class ControllerTest(unittest.TestCase):
 def jobs(self):return [{'order':i,'block':i//2,'arm':controller.ARMS[i//2][i%2],'measured':list(range(48))} for i in range(8)]
 def test_all_eight_complete(self):
  with tempfile.TemporaryDirectory() as td:
   def run(j,d):d.mkdir();return {'container_identity':{'container_id':('%064x'%(j['order']+1)),'host_init':{'pid':j['order']+1,'start_ticks':1,'boot_id':'b'}}}
   result=controller.run(self.jobs(),Path(td)/'run',run,lambda *x:None,lambda:None,10000,10,10,now=lambda:0);self.assertEqual(result['measured'],384)
 def test_failure_stops_and_is_recorded(self):
  with tempfile.TemporaryDirectory() as td:
   seen=[]
   def run(j,d):seen.append(j['order']);raise RuntimeError('fail')
   out=Path(td)/'run'
   with self.assertRaises(RuntimeError):controller.run(self.jobs(),out,run,lambda *x:None,lambda:None,10000,10,10,now=lambda:0)
   self.assertEqual(seen,[0]);self.assertIn('controller_failed',(out/'controller.jsonl').read_text())
if __name__=='__main__':unittest.main()
