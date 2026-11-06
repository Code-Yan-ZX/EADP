"""Fake-only controller checks: all subprocess creation is forbidden."""
import json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import controller
class Runtime:
 def __init__(self,busy=False):self.busy=busy;self.calls=[]
 def gpu_pids(self):return [123] if self.busy else []
 def identity(self,pid):return {'pid':pid}
 def container_command(self,author,worker,gpu,name):
  self.calls.append((worker,gpu,name));return ['docker','run','--device','/dev/nvidia0','--device','/dev/nvidiactl','--device','/dev/nvidia-uvm','--device','/dev/nvidia-uvm-tools','--env','CUDA_VISIBLE_DEVICES=0','--rm']+worker
class Timing:
 def __init__(self,statuses):self.statuses=iter(statuses);self.calls=0
 def run_bounded(self,command,name,timeout,gpu_pids,identity,artifacts,require_gpu_idle):
  self.calls+=1
  if timeout!=1600 or not require_gpu_idle or gpu_pids()!=[] or command.count('--device')!=4 or '--rm' in command:raise AssertionError('controller boundary')
  (artifacts/'observations/worker_complete.json').write_text(json.dumps({'status':next(self.statuses)}));return 0
class Tests(unittest.TestCase):
 def test_two_arms_and_mismatch_are_not_discarded(self):
  with tempfile.TemporaryDirectory() as d,patch.object(subprocess,'Popen',side_effect=AssertionError('forbidden subprocess')):
   runtime=Runtime();timing=Timing(['failed','passed']);out=Path(d)/'run';rc=controller.dispatch(runtime,timing,{},out,{'fake':True},lambda p:json.loads(p.read_text()))
   result=json.loads((out/'controller_complete.json').read_text());self.assertEqual(rc,1);self.assertEqual(timing.calls,2);self.assertEqual([x['status'] for x in result['arms']],['failed','passed'])
 def test_busy_gpu_never_launches(self):
  with tempfile.TemporaryDirectory() as d,patch.object(subprocess,'Popen',side_effect=AssertionError('forbidden subprocess')):
   runtime=Runtime(busy=True);timing=Timing([])
   with self.assertRaisesRegex(ValueError,'GPU busy'):controller.dispatch(runtime,timing,{},Path(d)/'run',{},lambda p:json.loads(p.read_text()))
   self.assertEqual(runtime.calls,[]);self.assertEqual(timing.calls,0)
