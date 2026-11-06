import copy,json,sys,types,unittest
from unittest.mock import patch
import gpu_gate as g
class Tests(unittest.TestCase):
 def fixture(self):
  image='sha256:img';name='pilot';run=str(g.V/'run001');cmd=['docker','run','--device','/dev/nvidia0','--env','X=y','--mount','type=bind,src=/tmp,dst=/tmp,readonly','--network','none','--user','1000:1000','--workdir','/tmp','--entrypoint','python','--rm',image,str(g.V/'worker.py'),'--out',run]
  la={'name':name,'gpu_requested':True,'command':[x for x in cmd if x!='--rm']};ex={'container_stopped_verified':True,'container_exit_code':0,'failure_type':None,'gpu_pids_after':[],'container_identity':{'container_id':'id'}}
  obj={'Id':'id','Image':image,'State':{'Running':False,'Pid':0,'ExitCode':0,'OOMKilled':False},'Config':{'Image':image,'Entrypoint':['python'],'User':'1000:1000','WorkingDir':'/tmp','Cmd':cmd[cmd.index(image)+1:],'Env':['X=y']},'HostConfig':{'NetworkMode':'none','Devices':[{'PathOnHost':'/dev/nvidia0','PathInContainer':'/dev/nvidia0','CgroupPermissions':'rwm'}],'DeviceRequests':None},'Mounts':[{'Type':'bind','Source':'/tmp','Destination':'/tmp','RW':False}]}
  author=types.SimpleNamespace(container_command=lambda *args:cmd.copy());return la,ex,obj,author,image
 def check(self,la,ex,obj,author,image):
  with patch.dict(sys.modules,{'_multistream_v3_preflight_author':author}),patch.object(g,'read',return_value={'container_image':image}),patch.object(g.subprocess,'run',return_value=types.SimpleNamespace(stdout=json.dumps([obj]))):return g.inspect_gpu(la,ex)
 def test_accept(self):
  la,ex,obj,a,image=self.fixture();self.assertEqual(self.check(la,ex,obj,a,image)['container_id'],'id')
 def test_tamper_image_devices_env_mount(self):
  for field in ('image','devices','env','mount','command'):
   la,ex,obj,a,image=self.fixture()
   if field=='image':obj['Image']='other'
   elif field=='devices':obj['HostConfig']['Devices']=[]
   elif field=='env':obj['Config']['Env']=['X=z']
   elif field=='mount':obj['Mounts'][0]['RW']=True
   else:obj['Config']['Cmd']=['other']
   with self.assertRaises(ValueError):self.check(la,ex,obj,a,image)
if __name__=='__main__':unittest.main()
