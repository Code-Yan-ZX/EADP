import signal,unittest
from interruptible import run_control
from test_lifecycle import Fake
class InterruptTest(unittest.TestCase):
 def test_term_fails_and_cleans_only_owned_container(self):
  x=Fake();previous=signal.getsignal(signal.SIGTERM)
  def interrupted(_):signal.raise_signal(signal.SIGTERM)
  x.observe_running=interrupted;r=run_control(x)
  self.assertEqual(r['status'],'failed');self.assertEqual(r['error']['type'],'InterruptedError');self.assertTrue(r['cleanup']['stopped_verified']);self.assertNotIn('release',x.calls);self.assertEqual(signal.getsignal(signal.SIGTERM),previous)
if __name__=='__main__':unittest.main()
