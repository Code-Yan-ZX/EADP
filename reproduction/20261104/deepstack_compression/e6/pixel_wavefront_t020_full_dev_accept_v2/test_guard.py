import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import finish_guard as guard

class GuardTests(unittest.TestCase):
    def run_case(self,mutate=None):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)/'a/b/c/d/e';controller={'pid':123};name='fake';cmd=['docker','run','--rm']
            expected=['docker','run'];device=['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']
            launch={'stage':'run','plan_sha256':'sha','controller':controller,'container_name':name,'command':expected}
            ex={'controller':controller,'returncode':0,'container_exit_code':0,'timed_out':False,'container_stopped_verified':True,
                'gpu_pids_after':[],'container_identity':{'container_id':'id','image':'image'},'last_inspect':{'id':'id'}}
            obj={'Id':'id','Image':'image','State':{'Running':False,'Pid':0,'ExitCode':0,'OOMKilled':False},
                 'Config':{'Cmd':[str(d/'worker.py'),'--out',str(d/'run_run001/observations')],'Env':[]},
                 'HostConfig':{'NetworkMode':'none','Devices':[{'PathOnHost':x,'PathInContainer':x,'CgroupPermissions':'rwm'} for x in device]},'Mounts':[]}
            if mutate:mutate(launch,ex,obj)
            def read(path):
                if path.name=='launch.json':return launch
                if path.name=='exit.json':return ex
                return {}
            runtime=types.SimpleNamespace(container_command=lambda *a:list(cmd))
            module=types.SimpleNamespace(runtime_module=lambda:runtime)
            process=types.SimpleNamespace(stdout=json.dumps([obj]))
            with patch.object(guard,'D',d),patch.object(guard.baseline,'supervised',return_value=({'worker_pid':123,'command':['--stage','run']},{})),patch.object(guard.baseline,'read',side_effect=read),patch.object(guard.baseline,'sha',return_value='sha'),patch.object(guard.subprocess,'run',return_value=process),patch.dict(sys.modules,{'launch':module}):
                return guard.validate()

    def test_real_entry_structure(self):self.assertEqual(self.run_case()['container_id'],'id')
    def test_reject_wrong_running_container(self):
        for change in (lambda l,e,o:o['State'].__setitem__('Pid',1),lambda l,e,o:o['State'].__setitem__('OOMKilled',True),lambda l,e,o:o['Config'].__setitem__('Cmd',['wrong']),lambda l,e,o:l.__setitem__('command',['wrong'])):
            with self.subTest(change=change),self.assertRaises(ValueError):self.run_case(change)

if __name__=='__main__':unittest.main()
