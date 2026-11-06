import tempfile,unittest,hashlib
from pathlib import Path
import completion_contract as c
class Tests(unittest.TestCase):
 def test_supervisor_and_weight_identity_fail_closed(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'file';p.write_bytes(b'x');sources={str(p):hashlib.sha256(b'x').hexdigest()};cmd=['python','launcher'];start={'command':cmd,'cwd':t,'worker_pid':3};end={'worker_pid':3,'reason':'completed','worker_returncode':0,'remaining_process_group':[]};weights=[{'path':str(i),'stat':[1,2,3,4,5]} for i in range(4)]
   self.assertTrue(c.check(start,end,cmd,t,sources,weights,weights)['supervisor_completed'])
   with self.assertRaises(ValueError):c.check(start,dict(end,remaining_process_group=[9]),cmd,t,sources,weights,weights)
   with self.assertRaises(ValueError):c.check(start,end,cmd,t,sources,weights,weights[:-1])
   p.write_bytes(b'y')
   with self.assertRaises(ValueError):c.check(start,end,cmd,t,sources,weights,weights)
if __name__=='__main__':unittest.main()
