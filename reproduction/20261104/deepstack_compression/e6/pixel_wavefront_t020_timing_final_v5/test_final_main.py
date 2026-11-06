"""Synthetic complete controller main; never launches Docker or queries a GPU."""
import importlib.util,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import launch

def import_file(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class MainTest(unittest.TestCase):
 def test_main_processes_all_24_and_binds_weights(self):
  jobs=[]
  for phase,reps in [('warmup',[0]),('measurement',[0,1,2])]:
   for i in range(3):
    for arm in ('author','wavefront'):
     for rep in reps:jobs.append({'task':str(i),'qid':str(i),'arm':arm,'phase':phase,'repeat':rep,'order':len(jobs)})
  refs={(j['task'],j['qid'],j['arm']):{} for j in jobs};controller=import_file('synthetic_controller_main',launch.R/'controller.py');calls=[]
  class Host:
   @staticmethod
   def bounded_one(job,jobdir,command,name,ctrl,ps,wp,ws,bounded,pids,identity,supervisor,timeout):
    jobdir.mkdir();calls.append(job['order']);(jobdir/'raw.json').write_text('{}');(jobdir/'record.json').write_text(json.dumps({'job':job,'record':{},'metrics':{'ok':True}}))
    return {'container_identity':{'container_id':format(job['order']+1,'064x'),'host_init':{'pid':100+job['order'],'start_ticks':1,'boot_id':'boot'}}}
  runtime=types.SimpleNamespace(container_command=lambda author,args,gpu,name:['docker','--rm']+args,identity=lambda pid:{'pid':pid},gpu_pids=lambda:[])
  dummy={'_pixel_t020_final_controller':controller,'_pixel_t020_final_host':Host,'_pixel_t020_final_check':types.SimpleNamespace(check=lambda *a:{'ok':True}),'_pixel_t020_final_refs':types.SimpleNamespace(load=lambda:(refs,{})),'_pixel_t020_final_rawprefix':types.SimpleNamespace(check=lambda *a:'sha'),'_pixel_t020_final_bounded':types.SimpleNamespace(run_bounded=None),'_pixel_t020_final_author_runtime':runtime}
  with tempfile.TemporaryDirectory() as t:
   launch_dir=Path(t);srcpaths=[launch.R/'controller.py',launch.R/'host_runner.py',launch.B/'e6/pixel_wavefront_t020_timing_controller_draft/job_contract.py',launch.C/'reference.py',launch_dir/'raw_prefix.py',launch_dir/'gpu_bounded.py',launch.ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_panel.py']
   plan={'source_sha256':{str(p.resolve()):'sha' for p in srcpaths},'outer_timeout_seconds':100,'grace_seconds':1,'snapshot_seconds':3600,'poll_seconds':5,'budget_source_sha256':'budget','minimum_disk_bytes':0,'weight_files':['a','b','c','d'],'inner_timeout_seconds':1,'cleanup_seconds':1}
   approval={'plan_sha256':'p','jobs':jobs,'worker_sha256':'w'}
   class Gate:
    @staticmethod
    def admit():return approval
    @staticmethod
    def sha(p):return 'budget'
    @staticmethod
    def read(p):
     p=Path(p)
     if p.name=='gpu_plan001.json':return plan
     if p.name=='initial_plan.json':return {'budget_start_unix':0,'budget_end_unix':1000}
     if p.name=='started.json':return {}
     if p.name=='run_plan.json':return {}
     return json.loads(p.read_text())
   dummy['_pixel_t020_final_gpu_gate']=Gate
   def load(name,path,expected):return dummy[name]
   weights=[{'path':str(i),'stat':[1,2,3,4,5]} for i in range(4)]
   with patch.object(launch,'D',launch_dir),patch.object(launch,'load',side_effect=load),patch.object(launch,'validate_supervisor'),patch.object(launch,'weight_stats',return_value=weights),patch.object(launch.time,'time',return_value=1),patch.object(launch.shutil,'disk_usage',return_value=types.SimpleNamespace(free=100)),patch.object(sys,'argv',['launch.py','--run-id','001']):
    self.assertEqual(launch.main(),0)
   self.assertEqual(calls,list(range(24)));self.assertTrue(json.loads((launch_dir/'run_001/final.json').read_text())['post_run_source_and_approval_checked']);self.assertEqual(json.loads((launch_dir/'run_001/weight_stats_before.json').read_text()),weights)
 def test_failed_gate_never_queries_gpu(self):
  class Gate:
   @staticmethod
   def admit():raise ValueError('no admission')
  with patch.object(launch,'load',return_value=Gate),patch.object(sys,'argv',['launch.py','--run-id','001']):
   with self.assertRaises(ValueError):launch.main()
if __name__=='__main__':unittest.main()

class ParentArgvTest(unittest.TestCase):
 def test_parent_command_binds_supervisor_budget_and_child(self):
  cwd=Path('/');plan={'outer_timeout_seconds':100,'grace_seconds':10,'snapshot_seconds':3600,'poll_seconds':5};child=['python','worker.py'];base=['python',str(launch.B/'e4_c_latency/timing_supervise.py'),'--run-dir',str(launch.D/'monitor_run001'),'--timeout','100','--grace-seconds','10','--snapshot-seconds','3600','--poll-seconds','5','--']+child
  self.assertTrue(launch.validate_parent_argv(base,cwd,child,plan))
  for bad in (base[:-1]+['evil.py'],[('99' if x=='100' else x) for x in base],base+['--timeout','100']):
   with self.assertRaises(ValueError):launch.validate_parent_argv(bad,cwd,child,plan)
