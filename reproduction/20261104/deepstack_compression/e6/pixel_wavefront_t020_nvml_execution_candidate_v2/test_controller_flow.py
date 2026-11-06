"""Single-job controller contract with fake Docker/NVML; never uses GPU."""
import json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import controller_entry as c
class ControllerFlow(unittest.TestCase):
 def test_real_controller_continuation_preserves_mode_and_evidence(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);d=root/'exec';d.mkdir();author=root/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';author.parent.mkdir(parents=True);author.write_text('{"container_image":"sha256:test"}')
   job={'order':0,'arm':'t020_wavefront'};approval={'jobs':[job]+[{}]*8,'plan_sha256':'a'*64};calls=[]
   def publish(path,value):Path(path).write_text(json.dumps(value))
   def construct(**kw):calls.append(kw);return types.SimpleNamespace(directory=kw['directory'])
   def lifecycle(ops,query):
    o=ops.directory;publish(o/'expected.json',{'container_id':'b'*64,'host_pid':42});(o/'samples.jsonl').write_text('{"type":"sample"}\n');(o/'device_metadata.jsonl').write_text('{"status":"completed"}\n')
    names=('barrier_released','model_load_start','model_load_end','request_start','request_end','teardown_end');events=[]
    for i,name in enumerate(names):
     x={'event':name,'monotonic_ns':i+1}
     if name in ('model_load_end','request_end'):x.update(allocated_peak_bytes=1,reserved_peak_bytes=2)
     events.append(x)
    (o/'worker_events.jsonl').write_text('\n'.join(json.dumps(x) for x in events));publish(o/'worker_complete.json',{'status':'completed'});publish(o/'identity.json',{'status':'passed'});return {'status':'passed'}
   ctrl=types.SimpleNamespace(query_gpu=lambda:('gpu',''),publish=publish,construct=construct,guarded_lifecycle=lifecycle);guard=types.SimpleNamespace(create_command=lambda argv,name,nonce:['docker','create'],parse_inventory=lambda *a:'GPU-abcd');runtime=types.SimpleNamespace(container_command=lambda *a:['docker','run']);stages=types.SimpleNamespace(summarize=lambda *a:{'status':'summarized'});verify=types.SimpleNamespace(verify=lambda *a:None)
   with patch.object(c,'D',d),patch.object(c,'ROOT',root):value=c.run_one(job,approval,ctrl,guard,runtime,stages,verify)
   self.assertEqual(calls[0]['mode'],'t020_wavefront');self.assertEqual(value['job'],job);self.assertTrue((d/'run_001/job_000/memory_summary.json').exists())
 def test_job_identity_mismatch_refuses_before_creation(self):
  with tempfile.TemporaryDirectory() as t,patch.object(c,'D',Path(t)):
   with self.assertRaises(ValueError):c.run_one({'order':0,'arm':'t0_author'},{'jobs':[{'order':0,'arm':'other'}]},None,None,None,None,None)
   self.assertFalse((Path(t)/'run_001').exists())
