import json,subprocess,tempfile,unittest,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pixel_natural_timing_pilot_v4'))
import gpu_launch
b=gpu_launch.gpu_bounded
class Callback(unittest.TestCase):
 def test_release_exception_stops_then_kills_and_records(self):
  state={'running':True};calls=[]
  class Process:
   def poll(self):return None if state['running'] else 137
   def wait(self,timeout=None):return 137
   def kill(self):state['running']=False
  def docker(args,**kwargs):
   action=args[1];calls.append(action)
   if action=='stop':return SimpleNamespace(returncode=1,stdout='',stderr='stop failed')
   if action=='kill':state['running']=False;return SimpleNamespace(returncode=0,stdout='',stderr='')
   data={'Id':'container-id','Image':'image','State':{'Running':state['running'],'Pid':99 if state['running'] else 0,'ExitCode':137},'HostConfig':{'Devices':[],'DeviceRequests':None,'PidMode':''},'Config':{'Entrypoint':['python'],'Cmd':['worker']}}
   return SimpleNamespace(returncode=0,stdout=json.dumps([data]),stderr='')
  identity=lambda pid:{'pid':pid,'start_ticks':1,'boot_id':'boot'}
  with tempfile.TemporaryDirectory() as d,patch.object(b.subprocess,'Popen',return_value=Process()),patch.object(b.subprocess,'run',side_effect=docker),patch.object(b,'process_attestation',return_value={'host_argv':['worker'],'host_cgroup':'container-id'}):
   code=b.run_bounded(['fake'],'test',60,None,identity,Path(d),require_gpu_idle=False,on_identity=lambda x:(_ for _ in ()).throw(ValueError('release refused')))
   proof=json.loads((Path(d)/'exit.json').read_text())
  self.assertNotEqual(code,0);self.assertEqual(proof['failure_type'],'ValueError');self.assertIn('stop',calls);self.assertIn('kill',calls);self.assertTrue(proof['container_stopped_verified'])
if __name__=='__main__':unittest.main(verbosity=2)
