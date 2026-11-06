"""Destination must remain invisible until full fsynced JSON is ready."""
import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import controller_io,worker_barrier
class AtomicPublishTest(unittest.TestCase):
 def test_controller_link_exposes_complete_immutable_json(self):
  with tempfile.TemporaryDirectory() as td:
   dest=Path(td)/'value.json';real=os.link;seen=[]
   def observe(src,dst):
    self.assertFalse(Path(dst).exists());self.assertEqual(json.loads(Path(src).read_text()),{'hello':'world'});seen.append(True);return real(src,dst)
   with patch.object(controller_io.os,'link',side_effect=observe):controller_io.publish(dest,{'hello':'world'})
   self.assertEqual(len(seen),1);self.assertEqual(json.loads(dest.read_text()),{'hello':'world'})
   with self.assertRaises(FileExistsError):controller_io.publish(dest,{'overwrite':True})
   self.assertEqual(json.loads(dest.read_text()),{'hello':'world'})
 def test_worker_ready_is_complete_before_becoming_visible(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);e={'mode':'t0_author','nonce':'a'*32,'container_id':'b'*64,'gpu_uuid':'GPU-abcd','host_pid':42,'start_ticks':1,'boot_id':'b','max_sample_gap_ns':10};(d/'expected.json').write_text(json.dumps(e));(d/'release.json').write_text(json.dumps({'status':'release','mode':e['mode'],'nonce':e['nonce'],'container_id':e['container_id'],'first_sample_ns':1}));real=os.link;seen=[]
   def observe(src,dst):
    self.assertFalse(Path(dst).exists());self.assertEqual(json.loads(Path(src).read_text())['namespace_pid'],1);seen.append(True);return real(src,dst)
   with patch.object(worker_barrier.os,'link',side_effect=observe):worker_barrier.handshake(d,e['nonce'],lambda:False,namespace_pid=lambda:1)
   self.assertEqual(len(seen),1)
if __name__=='__main__':unittest.main()
