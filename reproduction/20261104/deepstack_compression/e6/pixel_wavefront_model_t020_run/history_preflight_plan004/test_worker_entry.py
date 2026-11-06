import json,tempfile,unittest
from pathlib import Path
from worker_entry import await_identity,check_identity
class Tests(unittest.TestCase):
 def test_identity_rejects_wrong_pid_or_cgroup(self):
  cid='a'*64;x={'container_id':cid,'host_cgroup':'/docker/'+cid,'host_init':{'pid':123,'start_ticks':3,'boot_id':'boot'}}
  self.assertEqual(check_identity(x,'/docker/'+cid,1),cid)
  for group,pid in (('/other',1),('/docker/'+cid,2)):
   with self.assertRaises(ValueError):check_identity(x,group,pid)
  x['host_init']['start_ticks']=0
  with self.assertRaises(ValueError):check_identity(x,'/docker/'+cid,1)
 def test_partial_or_missing_identity_times_out(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'identity.json'
   with self.assertRaises(TimeoutError):await_identity(p,timeout=0)
   p.write_text('{')
   with self.assertRaises(TimeoutError):await_identity(p,timeout=0)
   p.write_text(json.dumps({'x':1}));self.assertEqual(await_identity(p,timeout=0),{'x':1})
if __name__=='__main__':unittest.main(verbosity=2)
