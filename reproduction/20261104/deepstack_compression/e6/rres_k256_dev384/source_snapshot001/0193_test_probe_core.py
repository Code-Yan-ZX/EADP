import json,tempfile,unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import torch
import probe_core
from test_panel_core import inputs
from test_processor_contract import TestProcessor
class Tests(unittest.TestCase):
 def run_probe(self,d,fail_at=None):
  rows,anchors,_,_=inputs();refs={(a['task'],a['qid']):{'processor':a['processor'],'image_start':a['image_start']} for a in anchors};fixture=TestProcessor();fixture.setUp();module=SimpleNamespace(QWEN3_FIXED_RESOLUTION=1024)
  class Wrapper:
   def __init__(self):self.calls=0
   def _build_messages(self,msg,dataset):return msg
   def _processor_inputs(self,_):
    self.calls+=1
    if self.calls==fail_at:raise RuntimeError('processor failure')
    return fixture.original if module.QWEN3_FIXED_RESOLUTION==1024 else fixture.low
  @contextmanager
  def fixed(m,n):
   m.QWEN3_FIXED_RESOLUTION=n
   try:yield
   finally:m.QWEN3_FIXED_RESOLUTION=1024
  for r in rows:r['message']='synthetic'
  return probe_core.run(rows,anchors,refs,torch,Wrapper(),module,fixture.cfg,fixed,lambda *a:None,fixture.rope,d,verify_image=lambda _:'b'*64)
 def test_complete(self):
  with tempfile.TemporaryDirectory() as d:
   self.assertEqual(len(self.run_probe(d)),384);self.assertEqual(json.loads((Path(d)/'processor_probe.json').read_text())['images'],384)
 def test_mid_failure_preserves_partial_events(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(RuntimeError):self.run_probe(d,fail_at=6)
   f=json.loads((Path(d)/'processor_failure.json').read_text());self.assertEqual((f['expected'],f['started'],f['verified']),(384,3,2));self.assertFalse((Path(d)/'processor_probe.json').exists())
if __name__=='__main__':unittest.main()
