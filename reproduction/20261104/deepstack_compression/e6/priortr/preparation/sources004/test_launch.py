import unittest
from smoke_launch import exit_code
class LauncherTests(unittest.TestCase):
 def test_rejects_every_unclean_exit_even_if_client_zero(self):
  good={'returncode':0,'container_exit_code':0,'timed_out':False,'interrupted':None,'client_exited_while_container_running':False,'gpu_pids_after':[],'container_stopped_verified':True,'container_identity':{'container_id':'x'}}
  self.assertEqual(exit_code(good),0)
  for key,value in [('returncode',1),('container_exit_code',1),('timed_out',True),('interrupted',15),('client_exited_while_container_running',True),('gpu_pids_after',['123']),('gpu_pids_after',None),('container_stopped_verified',False),('container_identity',None)]:
   bad=dict(good);bad[key]=value;self.assertNotEqual(exit_code(bad),0,key)
if __name__=='__main__':unittest.main()
