import unittest
from container_mount import add,MOUNT,verify_actual
class MountTests(unittest.TestCase):
 def test_exact_readonly_audit_mount(self):
  out=add(['docker','run','--workdir','/x','image'])
  self.assertEqual(out.count(MOUNT),1);self.assertTrue(MOUNT.endswith(',readonly'))
  self.assertLess(out.index(MOUNT),out.index('--workdir'))
 def test_existing_destination_is_refused(self):
  with self.assertRaises(ValueError):add(['docker','run','--mount','type=bind,src=/tmp,dst=/data/models/huggingface/hub','--workdir','/x','image'])
 def test_inspection_rejects_rw_or_wrong_source(self):
  command=add(['docker','run','--workdir','/x','image'])
  info={'Mounts':[{'Type':'bind','Source':'/data/models/huggingface/hub','Destination':'/data/models/huggingface/hub','RW':False}]}
  self.assertEqual(verify_actual(command,info),1)
  info['Mounts'][0]['RW']=True
  with self.assertRaises(ValueError):verify_actual(command,info)
if __name__=='__main__':unittest.main()
