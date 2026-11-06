import tempfile,unittest
from types import SimpleNamespace
from worker_candidate import run_after_preflight
class WorkerTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.initialized=False;self.torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:self.initialized,get_device_name=lambda _:self.fail('GPU must not be accessed')))
  self.params=dict(stage='smoke',mode='C-ON',nonce='a'*32,directory=self.tmp.name,prepared={'schedule':{'schedule':[]},'source_sha256':{},'weights_before':{}},expected_schedule={'schedule':[]},identity={},torch=self.torch,native=None,wrapper_class=lambda **k:self.fail('model load'),preflight_one=None,execute_one=None,memory_measure=None,run_memory_jobs=None,durable_journal=None,sha=lambda p:'')
 def test_reject_cuda_before_handshake(self):
  self.initialized=True
  with self.assertRaisesRegex(ValueError,'CUDA initialized'):run_after_preflight(**self.params)
 def test_wrong_barrier_before_gpu(self):
  self.params['handshake_fn']=lambda *a:({'mode':'C-SKIP','nonce':'a'*32},{'mode':'C-ON','nonce':'a'*32})
  with self.assertRaisesRegex(ValueError,'barrier identity'):run_after_preflight(**self.params)
 def test_wrong_schedule_before_gpu(self):
  self.params['handshake_fn']=lambda *a:({'mode':'C-ON','nonce':'a'*32,'container_id':'b'*64},{'status':'release','mode':'C-ON','nonce':'a'*32,'container_id':'b'*64})
  self.params['expected_schedule']={'schedule':[1]}
  with self.assertRaisesRegex(ValueError,'schedule differs'):run_after_preflight(**self.params)
if __name__=='__main__':unittest.main()

class WorkerEndToEndFakeTest(unittest.TestCase):
 def test_24_job_single_mode_with_real_schedule_executor(self):
  import contextlib,importlib.util,json,sys
  from pathlib import Path
  memory=Path(__file__).resolve().parent.parent/'e4_c_memory';sys.path.insert(0,str(memory))
  try:
   spec=importlib.util.spec_from_file_location('eadp_memory_worker_cpu',memory/'worker.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
  finally:sys.path.remove(str(memory))
  modes=('C-ON','C-DROP','C-SKIP');rows=[{'task':'T','qid':str(i),'group_id':'g'+str(i),'split':'DEV','image_start':4} for i in range(3)];jobs=[]
  for i in range(72):
   j=i%3;jobs.append({'order':i,'mode':modes[j],'task':'T','qid':str(j),'group_id':'g'+str(j),'phase':'measure' if (i//3)%2 else 'warmup','protocol':'fixed32' if (i//3)%2 else 'natural'})
  schedule={'schedule':jobs};refs={('T',str(i)):{} for i in range(3)}
  prepared={'schedule':schedule,'source_sha256':{},'weights_before':{},'e1_acceptance_sha256':'e','e1_started_sha256':'s','e1_terminal_sha256':'t','anchor_sha256':'a','rows':rows,'refs':refs,'plan':{'recipe':{}}}
  class Token:
   def detach(self):return self
   def cpu(self):return self
   def tolist(self):return [[1]*32]
  class Wrapper:
   def __init__(self,**kw):self.model=SimpleNamespace(dtype='bf16',config=SimpleNamespace(_attn_implementation='flash_attention_2'),generate=lambda:Token())
  cuda=SimpleNamespace(is_initialized=lambda:False,get_device_name=lambda _:'NVIDIA A40',synchronize=lambda:None)
  torch=SimpleNamespace(cuda=cuda,bfloat16='bf16',inference_mode=contextlib.nullcontext)
  def execute(w,n,t,row,mode,protocol,sync):w.model.generate();return {'mode':mode,'protocol':protocol,'output_ids':[1]*32}
  def memory_measure(t,m,run):return run(),{'allocated_peak_bytes':1,'reserved_peak_bytes':2,'kv_cache_bytes_final':3,'kv_cache_bytes_peak_logical':4,'kv_observed_forwards':1}
  def journal(path):
   f=Path(path).open('x')
   def append(row):f.write(json.dumps(row)+'\n');f.flush()
   return f,append
  e={'mode':'C-ON','nonce':'a'*32,'container_id':'b'*64};release={'status':'release','mode':'C-ON','nonce':'a'*32,'container_id':'b'*64}
  with tempfile.TemporaryDirectory() as d:
   out=run_after_preflight(stage='smoke',mode='C-ON',nonce='a'*32,directory=d,prepared=prepared,expected_schedule=schedule,identity={},torch=torch,native=None,wrapper_class=Wrapper,preflight_one=lambda *a:None,execute_one=execute,memory_measure=memory_measure,run_memory_jobs=worker.run_memory_jobs,durable_journal=journal,sha=lambda p:'',handshake_fn=lambda *a:(e,release))
   self.assertEqual(out['jobs'],24);events=[json.loads(x) for x in (Path(d)/'journal.jsonl').read_text().splitlines()];self.assertEqual(len(events),72);self.assertEqual(sum(x['status']=='completed' for x in events),24)
