import copy,unittest
from contracts import COMPONENTS
from test_contracts import schedule
from pilot_verify import check_events
from pilot_admission import admit
class Tests(unittest.TestCase):
 def setUp(self):
  self.s=schedule();self.fp={c:c+'-hash' for c in COMPONENTS};self.events=[{'status':'capture_generated'},{'status':'capture_cleanup','passed':True,'capture_completed':True},{'status':'archive_written'},{'status':'capture_identity_passed','identity':{'status':'passed'}},{'status':'reference_fingerprints','sha256':self.fp}];seen=set()
  for j in self.s['jobs']:
   c=j['component']
   if c not in seen:
    self.events.extend({'status':'warmup_validated','component':c,'index':i,'full_output_equal':True,'output_sha256':self.fp[c]} for i in range(3));seen.add(c)
   self.events.append({'status':'measured','job':j,'full_output_equal':True,'output_sha256':self.fp[c],'gpu_event_ms':0.1,'host_wall_ms':0.2,'scope':'excludes decoder DS addition'})
 def test_complete_and_bad_output(self):
  check_events(self.events,self.s,self.fp)
  bad=copy.deepcopy(self.events);next(e for e in bad if e['status']=='measured')['output_sha256']='fake'
  with self.assertRaises(ValueError):check_events(bad,self.s,self.fp)
 def test_missing_jobs_and_unclean_hooks(self):
  with self.assertRaises(ValueError):check_events(self.events[:-1],self.s,self.fp)
  bad=copy.deepcopy(self.events);bad[1]['passed']=False
  with self.assertRaises(ValueError):check_events(bad,self.s,self.fp)
 def test_missing_review_locks_gpu_and_preflight(self):
  for mode in (True,False):
   with self.assertRaises(ValueError):admit(preflight_only=mode,host=False)
if __name__=='__main__':unittest.main()

class ContainerTests(unittest.TestCase):
 def test_full_inspect_and_tampering(self):
  import json,sys
  from pathlib import Path
  from pilot_admission import HERE,validate_container
  from real_worker import PLAN
  sys.path.insert(0,str(PLAN.parent));from run_panel import container_command
  p=json.loads(PLAN.read_text());out=HERE/'worker_preflight001';name='test-pilot';cmd=container_command(p,[str(HERE/'pilot_worker.py'),'--out',str(out),'--preflight-only'],False,name);cmd.remove('--rm')
  image=p['container_image'];mounts=[]
  for i,x in enumerate(cmd[:-1]):
   if x=='--mount':
    parts=cmd[i+1].split(',');a=dict(y.split('=',1) for y in parts if '=' in y);mounts.append({'Type':'bind','Source':a['src'],'Destination':a['dst'],'RW':'readonly' not in parts})
  env=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env'];ident={'container_id':'abc'};launch={'command':cmd,'name':name,'out':str(out),'controller':{'pid':5}};ex={'controller':{'pid':5},'container_identity':ident}
  obj={'Name':'/'+name,'Id':'abc','Image':image,'State':{'Running':False,'Pid':0,'ExitCode':0,'OOMKilled':False},'Config':{'Image':image,'Entrypoint':[cmd[cmd.index('--entrypoint')+1]],'User':cmd[cmd.index('--user')+1],'WorkingDir':cmd[cmd.index('--workdir')+1],'Cmd':cmd[cmd.index(image)+1:],'Env':env},'HostConfig':{'NetworkMode':cmd[cmd.index('--network')+1],'Devices':[],'DeviceRequests':None,'PidMode':''},'Mounts':mounts}
  validate_container(launch,ex,obj,gpu=False)
  for key,value in [('Running',True),('OOMKilled',True),('ExitCode',1)]:
   bad=copy.deepcopy(obj);bad['State'][key]=value
   with self.assertRaises(ValueError):validate_container(launch,ex,bad,gpu=False)
  bad=copy.deepcopy(obj);bad['HostConfig']['Devices']=[{'PathOnHost':'/dev/nvidia0','PathInContainer':'/dev/nvidia0','CgroupPermissions':'rwm'}]
  with self.assertRaises(ValueError):validate_container(launch,ex,bad,gpu=False)
  bad=copy.deepcopy(obj);bad['Mounts'][0]['RW']=not bad['Mounts'][0]['RW']
  with self.assertRaises(ValueError):validate_container(launch,ex,bad,gpu=False)

class SupervisionTests(unittest.TestCase):
 def test_cpu_supervisor_binds_directory_argument(self):
  import json,tempfile
  from pathlib import Path
  from identity_gate import supervised
  with tempfile.TemporaryDirectory() as temp:
   d=Path(temp);script=d/'cpu_validation.py';out=d/'result';start=d/'started.json';end=d/'terminal.json'
   start.write_text(json.dumps({'cwd':str(d),'command':['python',str(script),'--out',str(out)],'worker_pid':123}));end.write_text(json.dumps({'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':123}))
   supervised(start,end,script,out)
   with self.assertRaises(ValueError):supervised(start,end,script,out/'verification.json')
