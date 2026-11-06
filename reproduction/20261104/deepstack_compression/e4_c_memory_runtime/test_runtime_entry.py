import json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import runtime_preflight_entry as e
class RuntimeEntryTest(unittest.TestCase):
 def test_preflight_only_writes_no_gpu_and_binds_sources(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);approval={'gpu_authorized':False,'source_sha256':{'x':'y'}};(p/'launch.json').write_text(json.dumps({'approval_bundle':approval}))
   fake=types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda:False));previous={'status':'passed','cuda_initialized':False,'model_loaded':False}
   with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(e,'source_hashes',return_value=approval['source_sha256']),patch.dict(sys.modules,{'torch':fake}):
    r=e.preflight(p,approval,lambda stage:previous,lambda:{'bridge':'/fixed'},lambda:{'status':'passed'})
   self.assertFalse(r['cuda_initialized']);self.assertFalse(r['model_loaded']);self.assertFalse(r['predictions']);self.assertTrue((p/'worker_preflight.json').exists())
 def test_gpu_environment_or_source_tamper_fails(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);approval={'gpu_authorized':False,'source_sha256':{'x':'y'}};(p/'launch.json').write_text(json.dumps({'approval_bundle':approval}))
   with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
    with self.assertRaisesRegex(ValueError,'no-GPU'):e.preflight(p,approval,None,None,None)
   with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(e,'source_hashes',return_value={'x':'changed'}):
    with self.assertRaisesRegex(ValueError,'sources changed'):e.preflight(p,approval,None,None,None)
if __name__=='__main__':unittest.main()
