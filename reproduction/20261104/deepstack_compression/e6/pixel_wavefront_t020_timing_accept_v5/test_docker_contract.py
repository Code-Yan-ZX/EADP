import unittest,copy
import docker_contract as d
class Tests(unittest.TestCase):
 def test_actual_docker_profile_and_forgery(self):
  job={'order':0};author={'container_image':'sha256:image'};worker='/tmp/worker.py';cid='a'*64;name='eadp-pixel-t020-ttft-000-8'
  def command(a,argv,gpu,n):return ['docker','run','--rm','--name',n,'--entrypoint','python','--user','1000','--workdir','/tmp','--network','none','--env','A=B','--device','/dev/nvidia0','--mount','type=bind,src=/tmp,dst=/mnt,readonly',a['container_image']]+argv
  cmd=command(author,[worker,'--job-index','0'],True,name);cmd.remove('--rm');ctrl={'pid':8};launch={'job':job,'controller':ctrl,'container_name':name,'command':cmd};ex={'controller':ctrl,'container_identity':{'container_id':cid,'image':'sha256:image'}}
  obj={'Name':'/'+name,'Id':cid,'Image':'sha256:image','Config':{'Image':'sha256:image','Entrypoint':['python'],'User':'1000','WorkingDir':'/tmp','Cmd':[worker,'--job-index','0'],'Env':['A=B']},'HostConfig':{'NetworkMode':'none','PidMode':'','DeviceRequests':None,'Devices':[{'PathOnHost':'/dev/nvidia0','PathInContainer':'/dev/nvidia0','CgroupPermissions':'rwm'}]},'State':{'Running':False,'Pid':0,'ExitCode':0,'OOMKilled':False},'Mounts':[{'Type':'bind','Source':'/tmp','Destination':'/mnt','RW':False}]}
  self.assertEqual(d.validate(job,launch,ex,obj,author,command,worker)['container_id'],cid)
  for key in ('image','oom','devices','mounts'):
   x=copy.deepcopy(obj)
   if key=='image':x['Image']='other'
   if key=='oom':x['State']['OOMKilled']=True
   if key=='devices':x['HostConfig']['Devices']=[]
   if key=='mounts':x['Mounts'][0]['RW']=True
   with self.assertRaises(ValueError,msg=key):d.validate(job,launch,ex,x,author,command,worker)
if __name__=='__main__':unittest.main()
