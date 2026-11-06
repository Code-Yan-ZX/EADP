import copy,importlib.util,unittest,tempfile,sys
from unittest.mock import patch
from pathlib import Path
p=Path(__file__).resolve().parent/'verify.py';spec=importlib.util.spec_from_file_location('random_verify_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def fixture(self):
  worker=[str(m.D/'model_worker.py'),'--out',str(m.O)];cmd=['docker','run','--name','test','--user','1000:1000','--workdir','/work','--env','X=Y','--mount','type=bind,src=/host,dst=/guest,readonly','--device','/dev/nvidia0:/dev/nvidia0','--entrypoint','/python','sha256:img',*worker]
  doc={'Image':'sha256:img','Name':'/test','Config':{'Entrypoint':['/python'],'Cmd':worker,'User':'1000:1000','WorkingDir':'/work','Env':['X=Y']},'Mounts':[{'Type':'bind','Destination':'/guest','Source':'/host','RW':False}],'HostConfig':{'Devices':[{'PathOnHost':'/dev/nvidia0'}]}}
  return doc,cmd
 def test_realistic_container_recipe(self):
  doc,cmd=self.fixture();m.validate_docker(doc,cmd)
 def test_tampered_image_mount_device_or_command_rejected(self):
  for field in ('image','mount','device','command','env'):
   doc,cmd=self.fixture()
   if field=='image':doc['Image']='other'
   if field=='mount':doc['Mounts'][0]['RW']=True
   if field=='device':doc['HostConfig']['Devices']=[]
   if field=='command':doc['Config']['Cmd'][0]='wrong'
   if field=='env':doc['Config']['Env']=['X=other']
   with self.subTest(field=field),self.assertRaises(ValueError):m.validate_docker(doc,cmd)
 def test_missing_acceptance_plan_prevents_output_access(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp)/'verification001.json'
   with patch.object(m,'A',Path(temp)),patch.object(sys,'argv',['verify.py','--out',str(out)]),patch.object(m,'load') as module_loader:
    with self.assertRaises(FileNotFoundError):m.main()
    module_loader.assert_not_called();self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
