import copy,unittest
from pathlib import Path
import container_profile as c
class Profile(unittest.TestCase):
 def sample(self):
  pilot=Path('/tmp/pilot');run=pilot/'run_pilot001/job_000';job={'order':0};controller={'pid':10,'start_ticks':20,'boot_id':'b'};name='eadp-pixel-ttft-000-10';image='sha256:image'
  def command(plan,args,gpu,name):return ['docker','run','--rm','--network','none','--user','1000:1000','--device','/dev/nvidia0','--mount','type=bind,src=/tmp,dst=/tmp,readonly','--env','A=B','--workdir','/work','--entrypoint','/python',image,*args]
  cmd=command({},[str(pilot/'gpu_worker_candidate.py'),'--job-index','0'],True,name);cmd.remove('--rm')
  launch={'job':job,'controller':controller,'command':cmd,'container_name':name};ex={'controller':controller,'container_identity':{'container_id':'id','image':image}}
  obj={'Name':'/'+name,'Id':'id','Image':image,'State':{'Running':False,'Pid':0,'ExitCode':0,'OOMKilled':False},'Config':{'Image':image,'Entrypoint':['/python'],'User':'1000:1000','WorkingDir':'/work','Cmd':[str(pilot/'gpu_worker_candidate.py'),'--job-index','0'],'Env':['A=B']},'HostConfig':{'NetworkMode':'none','PidMode':'','DeviceRequests':None,'Devices':[{'PathOnHost':'/dev/nvidia0','PathInContainer':'/dev/nvidia0','CgroupPermissions':'rwm'}]},'Mounts':[{'Type':'bind','Source':'/tmp','Destination':'/tmp','RW':False}]}
  return run,launch,ex,obj,{'container_image':image},command,job,pilot
 def test_valid_and_profile_tampering(self):
  args=list(self.sample());self.assertEqual(c.validate(*args)['container_id'],'id')
  for target,value in [('User','root'),('Env',['A=C']),('Cmd',['bad'])]:
   args=list(self.sample());args[3]['Config'][target]=value
   with self.assertRaises(ValueError):c.validate(*args)
  args=list(self.sample());args[3]['Mounts'][0]['RW']=True
  with self.assertRaises(ValueError):c.validate(*args)
 def test_running_or_device_tampering(self):
  args=list(self.sample());args[3]['State']['Running']=True
  with self.assertRaises(ValueError):c.validate(*args)
  args=list(self.sample());args[3]['HostConfig']['Devices']=[]
  with self.assertRaises(ValueError):c.validate(*args)
if __name__=='__main__':unittest.main(verbosity=2)
