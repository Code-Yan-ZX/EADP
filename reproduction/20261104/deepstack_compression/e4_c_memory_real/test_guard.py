import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import guard
import controller
import preflight_launch

IMAGE='sha256:'+'a'*64
NAME='eadp-memory-test-1'; NONCE='b'*32

def fixture():
    cmd=['docker','run','--rm','--network','none','--cap-drop','ALL','--security-opt','no-new-privileges',
         '--user','1000:1000','--name',NAME,
         '--mount','type=bind,src=/tmp,dst=/data,readonly']
    for x in guard.DEVICES: cmd+=['--device',x]
    cmd+=['--env','CUDA_VISIBLE_DEVICES=0','--env','HF_HUB_OFFLINE=1',
          '--workdir','/work','--entrypoint','/python',IMAGE,'/worker','--stage','smoke']
    actual={'Id':'c'*64,'Name':'/'+NAME,'Image':IMAGE,'State':{'Running':True,'Pid':999},
            'Config':{'Image':IMAGE,'Entrypoint':['/python'],'Cmd':['/worker','--stage','smoke'],
                      'User':'1000:1000','WorkingDir':'/work','Env':['CUDA_VISIBLE_DEVICES=0','HF_HUB_OFFLINE=1']},
            'HostConfig':{'NetworkMode':'none','PidMode':'','DeviceRequests':None,'CapDrop':['ALL'],
                          'SecurityOpt':['no-new-privileges'],
                          'Devices':[{'PathOnHost':x,'PathInContainer':x,'CgroupPermissions':'rwm'} for x in guard.DEVICES]},
            'Mounts':[{'Type':'bind','Source':'/tmp','Destination':'/data','RW':False}]}
    return cmd,actual

class DeviceTest(unittest.TestCase):
    def test_cpu_preflight_rejects_unsupervised_and_wrong_budget(self):
        plan=json.loads((preflight_launch.gate.HERE/'preflight_plan.json').read_text())
        cmd=plan['command'];worker=cmd[cmd.index('--')+1:];base=str(preflight_launch.gate.BASE)
        s={'worker_pid':42,'process_group':42,'cwd':base,'timeout_seconds':2100.0,
           'grace_seconds':120.0,'snapshot_seconds':3600.0,'poll_seconds':5.0,
           'stale_seconds':None,'progress_paths':[],'command':worker}
        def check(x,**kw):return preflight_launch.validate_supervisor(plan,x,42,kw.get('pgid',42),42,base,
                                                                      kw.get('parent',cmd),worker,kw.get('terminal',False))
        check(s)
        for key,value in [('worker_pid',41),('timeout_seconds',999),('grace_seconds',1),('command',['wrong'])]:
            bad=dict(s);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):check(bad)
        for kw in ({'parent':['wrong']},{'pgid':1},{'terminal':True}):
            with self.assertRaises(ValueError):check(s,**kw)

    def test_real_query_is_bounded_and_has_no_cuda_initialization(self):
        calls=[]
        def fake_run(args,**kwargs):
            calls.append((args,kwargs));return SimpleNamespace(stdout='answer')
        self.assertEqual(controller.query_gpu(fake_run),('answer','answer'))
        self.assertEqual(len(calls),2)
        self.assertTrue(all(c[1]['timeout']==10 and c[1]['check'] for c in calls))
        self.assertEqual(calls[0][0][1],'--query-gpu=index,name,uuid')
        self.assertEqual(calls[1][0][1],'--query-compute-apps=gpu_uuid,pid,used_gpu_memory')

    def test_guarded_lifecycle_blocks_before_create_and_after_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            ops=SimpleNamespace(directory=Path(td),gpu_uuid='GPU-abcd')
            with patch.object(controller,'lifecycle') as life:
                with self.assertRaises(ValueError): controller.guarded_lifecycle(ops,lambda:('0, NVIDIA A40, GPU-abcd','123'))
                life.assert_not_called()
            self.assertTrue((Path(td)/'before_gpu_query.json').exists())
        with tempfile.TemporaryDirectory() as td:
            ops=SimpleNamespace(directory=Path(td),gpu_uuid='GPU-abcd')
            calls=iter([('0, NVIDIA A40, GPU-abcd',''),('0, NVIDIA A40, GPU-abcd','123')])
            with patch.object(controller,'lifecycle',return_value={'status':'passed'}) as life:
                with self.assertRaises(ValueError): controller.guarded_lifecycle(ops,lambda:next(calls))
                life.assert_called_once()
            self.assertTrue((Path(td)/'after_gpu_query.json').exists())
        with tempfile.TemporaryDirectory() as td:
            ops=SimpleNamespace(directory=Path(td),gpu_uuid='GPU-abcd')
            def unavailable():raise TimeoutError('unavailable')
            with patch.object(controller,'lifecycle') as life:
                with self.assertRaises(TimeoutError):controller.guarded_lifecycle(ops,unavailable)
                life.assert_not_called()
            self.assertIn('TimeoutError',(Path(td)/'before_gpu_query.json').read_text())

    def test_host_process_argv_and_cgroup_are_bound(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);cid='c'*64
            (d/'cmdline').write_bytes(b'/python\0/worker\0--stage\0smoke\0')
            (d/'cgroup').write_text('0::/docker/'+cid+'\n')
            controller.validate_worker_proc(d,cid,['/python','/worker','--stage','smoke'])
            with self.assertRaises(ValueError):controller.validate_worker_proc(d,cid,['/python','/wrong'])
            (d/'cgroup').write_text('0::/docker/other\n')
            with self.assertRaises(ValueError):controller.validate_worker_proc(d,cid,['/python','/worker','--stage','smoke'])

    def test_exclusive_inventory(self):
        self.assertEqual(guard.parse_inventory('0, NVIDIA A40, GPU-abcd-1234\n',''), 'GPU-abcd-1234')
        for inv,compute in [('0, NVIDIA A40, GPU-abcd\n','123\n'),('1, NVIDIA A40, GPU-abcd',''),
                            ('0, NVIDIA A40, GPU-abcd\n1, NVIDIA A40, GPU-aaaa',''),
                            ('0, Unknown, GPU-abcd',''),('0, NVIDIA A40, bad','')]:
            with self.subTest(inv=inv, compute=compute), self.assertRaises(ValueError): guard.parse_inventory(inv,compute)

    def test_retained_owned_command(self):
        original,_=fixture(); created=guard.create_command(original,NAME,NONCE)
        self.assertEqual(created[:2],['docker','create']);self.assertNotIn('--rm',created)
        self.assertIn('eadp.memory.nonce='+NONCE,created)
        for mutate in (lambda c:c.remove('--rm'),lambda c:c.extend(['--gpus','all']),
                       lambda c:c.extend(['--label','x=y']),lambda c:c.remove('/dev/nvidia0')):
            x=original.copy();mutate(x)
            with self.assertRaises(ValueError):guard.create_command(x,NAME,NONCE)

    def test_inspect_rejects_image_device_mount_process_and_env_tampering(self):
        original,obj=fixture();cmd=guard.create_command(original,NAME,NONCE);worker=['/worker','--stage','smoke']
        self.assertEqual(guard.validate_running(obj,cmd,IMAGE,NAME,worker)['host_pid'],999)
        cases=[]
        for kind in ('image','pid','state','cmd','env','device','request','mount','network','security'):
            x=copy.deepcopy(obj)
            if kind=='image':x['Image']='wrong'
            elif kind=='pid':x['State']['Pid']=0
            elif kind=='state':x['State']['Running']=False
            elif kind=='cmd':x['Config']['Cmd']=['wrong']
            elif kind=='env':x['Config']['Env']=['CUDA_VISIBLE_DEVICES=1','HF_HUB_OFFLINE=1']
            elif kind=='device':x['HostConfig']['Devices'].pop()
            elif kind=='request':x['HostConfig']['DeviceRequests']=[{'Count':1}]
            elif kind=='mount':x['Mounts'][0]['RW']=True
            elif kind=='network':x['HostConfig']['NetworkMode']='host'
            else:x['HostConfig']['CapDrop']=[]
            cases.append((kind,x))
        for kind,x in cases:
            with self.subTest(kind=kind), self.assertRaises(ValueError):guard.validate_running(x,cmd,IMAGE,NAME,worker)

if __name__=='__main__':unittest.main()
