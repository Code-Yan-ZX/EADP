import unittest
from lifecycle import run
class Fake:
 def __init__(self,fail=None):self.fail=fail;self.calls=[];self.stopped=False;self.saved=None
 def mark(self,name):
  self.calls.append(name)
  if self.fail==name:raise RuntimeError(name)
 def create_container(self):self.mark('create');return 'owned'
 def recover_owned_container(self):self.calls.append('recover');return None
 def start_container(self,c):self.mark('start')
 def observe_running(self,c):self.mark('identity');return {'container_id':c}
 def wait_worker_ready(self,i):self.mark('ready');return {}
 def start_sampler(self,i):self.mark('start_sampler');return 'sampler'
 def wait_sampler_ready(self,s):self.mark('sampler_ready');return {'started':{},'first_sample':{}}
 def release_worker(self,i,r):self.mark('release')
 def wait_worker(self,c):self.mark('wait_worker');self.stopped=True;return {'exit_code':0}
 def inspect(self,c):self.mark('inspect');return {'container_id':c,'running':not self.stopped,'pid':0 if self.stopped else 42,'exit_code':0}
 def publish_stopped(self,i,s):self.mark('stopped');return 100
 def wait_sampler(self,s):self.mark('wait_sampler');return [],{}
 def stop_container(self,c):self.mark('stop');self.stopped=True
 def kill_container(self,c):self.mark('kill');self.stopped=True
 def stop_sampler(self,s):self.mark('stop_sampler')
 def persist_result(self,r):self.saved=r.copy();self.calls.append('persist')
class LifecycleTest(unittest.TestCase):
 def test_success_order_and_no_stop(self):
  x=Fake();r=run(x,lambda *a:{'status':'release'},lambda *a:{'status':'passed'})
  self.assertEqual(r['status'],'passed');self.assertLess(x.calls.index('sampler_ready'),x.calls.index('release'));self.assertLess(x.calls.index('inspect'),x.calls.index('stopped'));self.assertNotIn('stop',x.calls)
 def test_failed_create_recovers_owned_container_for_cleanup(self):
  x=Fake('create');x.recover_owned_container=lambda:'owned';r=run(x,lambda *a:{},lambda *a:{})
  self.assertEqual(r['status'],'failed');self.assertTrue(r['cleanup']['stopped_verified']);self.assertEqual(r['container_id'],'owned')
 def test_unknown_create_recovery_stays_failed(self):
  x=Fake('create')
  def unknown():raise RuntimeError('docker unavailable')
  x.recover_owned_container=unknown;r=run(x,lambda *a:{},lambda *a:{})
  self.assertEqual(r['status'],'failed');self.assertIn('docker unavailable',r['create_recovery_error'])
 def test_failed_start_still_stops_owned_container(self):
  x=Fake('start');r=run(x,lambda *a:{},lambda *a:{})
  self.assertEqual(r['status'],'failed');self.assertTrue(r['cleanup']['stopped_verified']);self.assertNotIn('release',x.calls)
 def test_sampler_failure_stops_without_release(self):
  x=Fake('sampler_ready');r=run(x,lambda *a:{},lambda *a:{})
  self.assertEqual(r['status'],'failed');self.assertNotIn('release',x.calls);self.assertTrue(r['cleanup']['stopped_verified']);self.assertIn('stop_sampler',x.calls);self.assertIsNotNone(x.saved)
 def test_stop_failure_falls_back_to_kill(self):
  x=Fake('stop');r=run(x,lambda *a:(_ for _ in ()).throw(ValueError('bad ready')),lambda *a:{})
  self.assertEqual(r['status'],'failed');self.assertTrue(r['cleanup']['kill_attempted']);self.assertTrue(r['cleanup']['stopped_verified']);self.assertIn('stop_sampler',x.calls)
 def test_invalid_exit_or_sample_audit_never_passes(self):
  x=Fake();r=run(x,lambda *a:{},lambda *a:{'status':'failed'});self.assertEqual(r['status'],'failed');self.assertIsNotNone(x.saved)
if __name__=='__main__':unittest.main()
