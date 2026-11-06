"""Real fixed-plan command plus synthetic Docker inspect counterexamples."""
import copy, unittest
from pathlib import Path
from full_imports import HERE,load
c=load('eadp_selected_full_container_identity',HERE/'container_identity.py')

def example():
 run=c.HERE/'panel_TEST';pid=999;plan,cmd,name=c.expected_command(run,pid);image=plan['container_image']
 env=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env']
 devices=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--device']
 mounts=[]
 for i,x in enumerate(cmd[:-1]):
  if x!='--mount':continue
  parts=cmd[i+1].split(',');attrs=dict(t.split('=',1) for t in parts if '=' in t)
  mounts.append({'Type':'bind','Source':attrs['src'],'Destination':attrs['dst'],'RW':'readonly' not in parts})
 ctrl={'pid':pid,'start_ticks':1,'boot_id':'test'};launch={'controller':ctrl,'command':cmd}
 exit={'controller':ctrl,'name':name,'container_identity':{'container_id':'container','image':image}}
 inspect={'Id':'container','Image':image,'State':{'Running':False,'Pid':0,'ExitCode':0,'OOMKilled':False},
          'Config':{'Image':image,'Entrypoint':[cmd[cmd.index('--entrypoint')+1]],'User':cmd[cmd.index('--user')+1],
                    'WorkingDir':cmd[cmd.index('--workdir')+1],'Cmd':cmd[cmd.index(image)+1:],'Env':env},
          'HostConfig':{'NetworkMode':cmd[cmd.index('--network')+1],'Devices':[{'PathOnHost':d,'PathInContainer':d,'CgroupPermissions':'rwm'} for d in devices],'DeviceRequests':None},
          'Mounts':mounts}
 return run,launch,exit,inspect

class ContainerIdentityTests(unittest.TestCase):
 def test_real_plan_and_docker_identity(self):
  run,launch,exit,inspect=example();out=c.validate(run,launch,exit,inspect)
  self.assertEqual(out['container_id'],'container');self.assertEqual(len(out['devices']),4);self.assertGreater(out['bind_mounts'],0)
 def test_tampered_actual_state_is_rejected(self):
  run,launch,exit,inspect=example()
  for mode in ('image','device','entrypoint','user','cwd','network','argv','env','mount','running','oom','othergpu'):
   with self.subTest(mode=mode):
    bad=copy.deepcopy(inspect)
    if mode=='image':bad['Image']='sha256:wrong'
    if mode=='device':bad['HostConfig']['Devices'].pop()
    if mode=='entrypoint':bad['Config']['Entrypoint']=['/tmp/python']
    if mode=='user':bad['Config']['User']='0:0'
    if mode=='cwd':bad['Config']['WorkingDir']='/tmp'
    if mode=='network':bad['HostConfig']['NetworkMode']='host'
    if mode=='argv':bad['Config']['Cmd'][-1]='/tmp'
    if mode=='env':bad['Config']['Env'].append('CUDA_VISIBLE_DEVICES=99')
    if mode=='mount':bad['Mounts'][0]['RW']=not bad['Mounts'][0]['RW']
    if mode=='running':bad['State']['Running']=True
    if mode=='oom':bad['State']['OOMKilled']=True
    if mode=='othergpu':bad['HostConfig']['DeviceRequests']=[{'Count':-1}]
    with self.assertRaises(ValueError):c.validate(run,launch,exit,bad)
 def test_recorded_command_and_controller_are_bound(self):
  run,launch,exit,inspect=example()
  altered=copy.deepcopy(launch);altered['command'][-1]='/tmp'
  with self.assertRaises(ValueError):c.validate(run,altered,exit,inspect)
  altered=copy.deepcopy(exit);altered['controller']['pid']=1000
  with self.assertRaises(ValueError):c.validate(run,launch,altered,inspect)
if __name__=='__main__':unittest.main()
