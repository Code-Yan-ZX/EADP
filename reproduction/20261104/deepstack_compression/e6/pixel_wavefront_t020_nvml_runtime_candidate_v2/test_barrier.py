import json,tempfile,unittest
from pathlib import Path
from barrier import write_ready,write_release,wait_release
import test_contract as fixtures

class BarrierTests(unittest.TestCase):
 def setUp(self):
  self.fixture=fixtures.MemoryHandshakeTests();self.fixture.setUp();self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.d=Path(self.tmp.name)
 def test_one_shot_handshake_and_release(self):
  x=self.fixture;write_ready(self.d,x.e,1,False);release=write_release(self.d,x.e,x.s,x.first)
  self.assertEqual(wait_release(self.d,x.e,1),release)
  with self.assertRaises(FileExistsError):write_release(self.d,x.e,x.s,x.first)
 def test_never_release_unready_or_changed_identity(self):
  x=self.fixture
  with self.assertRaises(ValueError):write_ready(self.d,x.e,1,True)
  write_ready(self.d,x.e,1,False);bad=dict(x.s,host_pid=999)
  with self.assertRaises(ValueError):write_release(self.d,x.e,bad,x.first)
  self.assertFalse((self.d/'release.json').exists())
 def test_wrong_release_is_rejected(self):
  x=self.fixture;(self.d/'release.json').write_text(json.dumps({'status':'release','nonce':'a'*32,'container_id':'b'*64,'mode':'C-DROP','first_sample_ns':1}))
  with self.assertRaises(ValueError):wait_release(self.d,x.e,1)
if __name__=='__main__':unittest.main()
