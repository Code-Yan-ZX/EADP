import json,tempfile,types,unittest
from pathlib import Path
import model_core
class FakeCuda:
 def get_device_name(self,n):return 'NVIDIA A40'
 def set_per_process_memory_fraction(self,*args):pass
 def max_memory_allocated(self):return 0
 def max_memory_reserved(self):return 0
class Tests(unittest.TestCase):
 def test_bound_inputs_then_cpu_fake_diagnostic(self):
  rows=[{'task':t,'qid':str(i),'group_id':str(i),'image_sha256':'i','message_sha256':'m'} for i,t in enumerate(('D','T','A'))];prep={'rows':rows,'refs':{(r['task'],r['qid']):{} for r in rows},'plan':{'recipe':{}},'weights_before':{}};plan={'anchors':rows};torch=types.SimpleNamespace(cuda=FakeCuda(),bfloat16='bf16');wrapper=types.SimpleNamespace(model=types.SimpleNamespace(dtype='bf16',config=types.SimpleNamespace(_attn_implementation='flash_attention_2'),model=types.SimpleNamespace(language_model=types.SimpleNamespace(register_forward_hook=lambda *a,**k:types.SimpleNamespace(remove=lambda:None)))));inventory=types.SimpleNamespace(validate_inventory=lambda *a:True);diag=types.SimpleNamespace(run=lambda *a:{'status':'completed','arms':18});seen=[]
  with tempfile.TemporaryDirectory() as d:
   result=model_core.run(prep,torch,None,lambda **kw:wrapper,lambda receipt,r:{},None,plan,{},d,None,None,None,None,None,None,None,inventory,diag,lambda t,w,row,ref:seen.append(row['qid']),lambda:None)
   self.assertEqual(result['arms'],18);self.assertEqual(seen,['0','1','2']);self.assertTrue(Path(d,'model_ready.json').exists())
 def test_weight_changed_before_completion_is_rejected(self):
  rows=[{'task':t,'qid':str(i),'group_id':str(i),'image_sha256':'i','message_sha256':'m'} for i,t in enumerate(('D','T','A'))]
  torch=types.SimpleNamespace(cuda=FakeCuda(),bfloat16='bf16');wrapper=types.SimpleNamespace(model=types.SimpleNamespace(dtype='bf16',config=types.SimpleNamespace(_attn_implementation='flash_attention_2'),model=types.SimpleNamespace(language_model=types.SimpleNamespace(register_forward_hook=lambda *a,**k:types.SimpleNamespace(remove=lambda:None)))));inventory=types.SimpleNamespace(validate_inventory=lambda *a:True)
  with tempfile.TemporaryDirectory() as d:
   weight=Path(d)/'weight';weight.write_text('a');stat=weight.stat();before=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns];prep={'rows':rows,'refs':{(r['task'],r['qid']):{} for r in rows},'plan':{'recipe':{}},'weights_before':{str(weight):before}}
   def run(*args):weight.write_text('changed');args[-1]();return {'arms':18}
   diag=types.SimpleNamespace(run=run)
   with self.assertRaisesRegex(ValueError,'weight identity changed'):model_core.run(prep,torch,None,lambda **kw:wrapper,lambda receipt,r:{},None,{'anchors':rows},{},d,None,None,None,None,None,None,None,inventory,diag,lambda *args:None,lambda:None)
   self.assertTrue(Path(d,'failure.json').exists());self.assertFalse(Path(d,'worker_complete.json').exists())
 def test_failure_written_without_completion(self):
  rows=[{'task':t,'qid':str(i),'group_id':str(i),'image_sha256':'i','message_sha256':'m'} for i,t in enumerate(('D','T','A'))];prep={'rows':rows,'refs':{},'plan':{'recipe':{}},'weights_before':{}};torch=types.SimpleNamespace(cuda=FakeCuda(),bfloat16='bf16');inventory=types.SimpleNamespace(validate_inventory=lambda *a:True)
  def fail(**kwargs):raise MemoryError('out of memory')
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(MemoryError):model_core.run(prep,torch,None,fail,lambda receipt,r:{},None,{'anchors':rows},{},d,None,None,None,None,None,None,None,inventory,None,None,lambda:None)
   self.assertEqual(json.loads(Path(d,'failure.json').read_text())['error_type'],'MemoryError');self.assertFalse(Path(d,'worker_complete.json').exists())
if __name__=='__main__':unittest.main()
