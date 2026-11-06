import importlib.util,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import full_admission,full_launch,full_worker
_spec=importlib.util.spec_from_file_location('eadp_full_preflight_worker',Path(__file__).resolve().parent/'preflight_worker.py')
preflight_worker=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(preflight_worker)
if Path(preflight_worker.__file__).resolve()!=(Path(__file__).resolve().parent/'preflight_worker.py'):raise ValueError('wrong full preflight worker import')
from schedule import build

def rows():return [{'task':t,'qid':str(i),'group_id':t+'-'+str(i),'split':'DEV'} for t in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST') for i in range(128)]

class FullTests(unittest.TestCase):
 def test_full_fixed_5940_jobs_and_protocol_counts(self):
  from collections import Counter
  s=build(rows(),[],'full');self.assertEqual(len(s['schedule']),5940);self.assertEqual(Counter(x['phase'] for x in s['schedule']),{'warmup':180,'measure':5760});self.assertEqual(Counter(x['protocol'] for x in s['schedule']),{'natural':2970,'fixed32':2970})
 def test_missing_review_denies_before_pilot_or_gpu(self):
  with patch.object(full_admission,'REVIEW',Path('/definitely-absent-full-review')),patch.object(full_admission,'pilot_gate',side_effect=AssertionError('must not inspect')):
   with self.assertRaisesRegex(ValueError,'independent review missing'):full_admission.admit(preflight_only=True)
 def test_actual_worker_rejects_before_prepare(self):
  with patch('sys.argv',['full_worker.py','--out','unused']),patch.object(full_worker,'admit',side_effect=ValueError('missing review')),patch.object(full_worker,'prepare',side_effect=AssertionError('must not load')):
   with self.assertRaisesRegex(ValueError,'missing review'):full_worker.main()
 def test_actual_launchers_reject_before_container(self):
  with patch('sys.argv',['full_launch.py','--run-id','x']),patch.object(full_launch,'admit',side_effect=ValueError('review missing')):
   with self.assertRaisesRegex(ValueError,'review missing'):full_launch.main()
  with patch('sys.argv',['preflight_worker.py','--run-id','x']),patch.object(preflight_worker,'admit',side_effect=ValueError('review missing')):
   with self.assertRaisesRegex(ValueError,'review missing'):preflight_worker.main()
 def test_full_launcher_reaches_mocked_boundary_and_denies_busy_gpu(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);worker=root/'full_worker.py';worker.write_text('worker');budget=root/'budget_plan.json';v={'budget_source':'../unused','inner_timeout_seconds':28800,'outer_timeout_seconds':28980,'journal_stale_seconds':1800,'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':120,'minimum_free_bytes':0};budget.write_text(json.dumps(v));source=[worker,budget];approval={'source_sha256':{str(x):full_launch.sha(x) for x in source}}
   def monitor(name):
    m=root/('monitor_full'+name);m.mkdir();(m/'started.json').write_text(json.dumps({'worker_pid':os.getpid(),'timeout_seconds':28980,'stale_seconds':1800,'progress_paths':[str(root/('full_'+name)/'journal.jsonl')],'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':120}))
   monitor('x');monitor('busy');calls=[];mod=types.ModuleType('run_panel');mod.container_command=lambda plan,args,gpu,name:(calls.append((args,gpu)) or ['docker','run','--rm','image',*args]);mod.gpu_pids=lambda:[];mod.identity=lambda pid:{'pid':pid};old=list(sys.path)
   try:
    with patch.object(full_launch,'HERE',root),patch.object(full_launch,'sources',return_value=source),patch.object(full_launch,'admit',return_value=approval),patch.object(full_launch,'validate_budget_window'),patch.object(full_launch,'run_bounded',return_value=0),patch.dict(sys.modules,{'run_panel':mod}),patch('sys.argv',['full_launch.py','--run-id','x']):
     self.assertEqual(full_launch.main(),0);self.assertIs(calls[0][1],True);mod.gpu_pids=lambda:[22]
     with patch('sys.argv',['full_launch.py','--run-id','busy']):
      with self.assertRaisesRegex(ValueError,'GPU occupied'):full_launch.main()
     self.assertFalse((root/'full_busy').exists());self.assertEqual(len(calls),1)
   finally:sys.path[:]=old
 def test_full_cpu_launcher_uses_no_gpu(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);worker=root/'full_worker.py';worker.write_text('worker');budget=root/'budget_plan.json';budget.write_text(json.dumps({'budget_source':'../unused'}));source=[worker,budget];approval={'source_sha256':{str(x):preflight_worker.sha(x) for x in source}}
   m=root/'monitor_worker_preflightx';m.mkdir();(m/'started.json').write_text(json.dumps({'worker_pid':os.getpid(),'timeout_seconds':1500,'grace_seconds':90}));calls=[];mod=types.ModuleType('run_panel');mod.container_command=lambda plan,args,gpu,name:(calls.append((args,gpu)) or ['docker','run','--rm','image',*args]);mod.identity=lambda pid:{'pid':pid};old=list(sys.path)
   try:
    with patch.object(preflight_worker,'HERE',root),patch.object(preflight_worker,'sources',return_value=source),patch.object(preflight_worker,'admit',return_value=approval),patch.object(preflight_worker,'validate_budget_window'),patch.object(preflight_worker,'run_bounded',return_value=0) as bounded,patch.dict(sys.modules,{'run_panel':mod}),patch('sys.argv',['preflight_worker.py','--run-id','x']):
     self.assertEqual(preflight_worker.main(),0);self.assertIs(calls[0][1],False);self.assertIsNone(bounded.call_args.args[3]);self.assertIs(bounded.call_args.kwargs['require_gpu_idle'],False)
   finally:sys.path[:]=old
if __name__=='__main__':unittest.main()

class ContainerIdentityTests(unittest.TestCase):
 def make(self):
  from container_identity import HERE
  run=HERE/'full_001';name='eadp-latency-full-001-123';image='sha256:image';controller={'pid':123};cid='a'*64
  cmd=['docker','run','--rm','--network','none','--user','1000:1000','--workdir','/work','--entrypoint','/python','--env','A=1','--device','/dev/nvidia0','--mount','type=bind,src=/tmp,dst=/input,readonly',image,str(HERE/'full_worker.py'),'--out',str(run)]
  recorded=[x for x in cmd if x!='--rm'];launch={'name':name,'command':recorded,'controller':controller};ex={'controller':controller,'container_identity':{'container_id':cid,'image':image}}
  obj={'Id':cid,'Name':'/'+name,'Image':image,'State':{'Running':False,'Pid':0,'ExitCode':0,'OOMKilled':False},'Config':{'Image':image,'Entrypoint':['/python'],'User':'1000:1000','WorkingDir':'/work','Cmd':recorded[recorded.index(image)+1:],'Env':['A=1']},'HostConfig':{'NetworkMode':'none','PidMode':'','DeviceRequests':None,'Devices':[{'PathOnHost':'/dev/nvidia0','PathInContainer':'/dev/nvidia0','CgroupPermissions':'rwm'}]},'Mounts':[{'Type':'bind','Source':'/tmp','Destination':'/input','RW':False}]}
  return run,launch,ex,obj,{'container_image':image},lambda *_:list(cmd)
 def test_accepts_exact_actual_identity(self):
  from container_identity import validate
  self.assertEqual(validate(*self.make())['bind_mounts'],1)
 def test_tampered_command_image_environment_device_mount_process_rejected(self):
  import copy
  from container_identity import validate
  for kind in ('command','image','env','device','readonly','mount','entrypoint','user','network','oom'):
   run,launch,ex,obj,plan,builder=self.make();obj=copy.deepcopy(obj)
   if kind=='command':launch['command'].append('extra')
   if kind=='image':obj['Image']='wrong'
   if kind=='env':obj['Config']['Env']=['A=2']
   if kind=='device':obj['HostConfig']['Devices']=[]
   if kind=='readonly':obj['Mounts'][0]['RW']=True
   if kind=='mount':obj['Mounts'][0]['Source']='/different'
   if kind=='entrypoint':obj['Config']['Entrypoint']=['/other']
   if kind=='user':obj['Config']['User']='root'
   if kind=='network':obj['HostConfig']['NetworkMode']='host'
   if kind=='oom':obj['State']['OOMKilled']=True
   with self.subTest(kind=kind),self.assertRaises(ValueError):validate(run,launch,ex,obj,plan,builder)
