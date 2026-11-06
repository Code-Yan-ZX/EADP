import json,signal,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pixel_natural_timing_pilot_v4'))
import gpu_launch
timing_launch=gpu_launch.gpu_bounded

class LifecycleTests(unittest.TestCase):
 def test_timeout_or_signal_with_failed_stop_kills_and_verifies_container(self):
  for mode in ('timeout','signal'):
   for failure in ('timeout','nonzero'):
    state={'running':True};handlers={};calls=[];clock=[0]
    def fake_signal(sig,handler):
     old=handlers.get(sig,signal.SIG_DFL);handlers[sig]=handler;return old
    class Process:
     def poll(self):
      if mode=='signal' and state['running'] and callable(handlers.get(signal.SIGTERM)):handlers[signal.SIGTERM](signal.SIGTERM,None)
      return None if state['running'] else 137
     def wait(self,timeout=None):return 137
     def kill(self):state['running']=False
    def docker(args,**kwargs):
     action=args[1];calls.append(action)
     if action=='stop':
      if failure=='timeout':raise subprocess.TimeoutExpired(args,30)
      return SimpleNamespace(returncode=1,stdout='',stderr='stop failed')
     if action=='kill':state['running']=False;return SimpleNamespace(returncode=0,stdout='ok',stderr='')
     if action=='inspect':
      data={'Id':'container-id','Image':'image','State':{'Running':state['running'],'Pid':99 if state['running'] else 0,'ExitCode':137},
            'HostConfig':{'Devices':[],'DeviceRequests':None,'PidMode':''},'Config':{'Entrypoint':['python'],'Cmd':['worker']}}
      return SimpleNamespace(returncode=0,stdout=json.dumps([data]),stderr='')
     raise AssertionError(action)
    def monotonic():clock[0]+=100 if mode=='timeout' else 0;return clock[0]
    identity=lambda pid:{'pid':pid,'start_ticks':1,'boot_id':'boot'}
    with tempfile.TemporaryDirectory() as d,patch.object(timing_launch.signal,'signal',side_effect=fake_signal),patch.object(timing_launch.subprocess,'Popen',return_value=Process()),patch.object(timing_launch.subprocess,'run',side_effect=docker),patch.object(timing_launch.time,'monotonic',side_effect=monotonic),patch.object(timing_launch,'process_attestation',return_value={'host_argv':['worker'],'host_cgroup':'container-id'}):
     result=timing_launch.run_bounded(['fake'],'test',60,None,identity,Path(d),require_gpu_idle=False);proof=json.loads((Path(d)/'exit.json').read_text())
    self.assertEqual(result,124 if mode=='timeout' else 128+signal.SIGTERM);self.assertIn('stop',calls);self.assertIn('kill',calls)
    self.assertTrue(proof['container_stopped_verified']);self.assertFalse(proof['last_inspect']['running']);self.assertEqual(proof['last_inspect']['pid'],0)
if __name__=='__main__':unittest.main()
