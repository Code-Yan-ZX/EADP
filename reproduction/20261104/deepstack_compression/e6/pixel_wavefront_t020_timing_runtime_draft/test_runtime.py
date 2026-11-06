import hashlib, json, tempfile, unittest
from pathlib import Path
import host_runner, worker
class RuntimeTest(unittest.TestCase):
 def test_release_requires_real_host_identity_and_is_atomic(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t); wd=d/'worker.py';wd.write_text('worker'); wh=hashlib.sha256(wd.read_bytes()).hexdigest();job={'order':3};ctrl={'pid':123};cid='a'*64
   launch=host_runner.create_launch(job,d/'job',['docker'],'name',ctrl,'plan')
   ident={'container_id':cid,'host_init':{'pid':8,'start_ticks':9,'boot_id':'b'},'host_cgroup':'/docker/'+cid,'host_argv':[str(wd),'--job-index','3']}
   bad=dict(ident,host_argv=[str(wd),'--job-index','4'])
   with self.assertRaises(ValueError):host_runner.publish_release(job,d/'job',launch,bad,ctrl,wd,'plan',wh)
   self.assertFalse((d/'job/release.json').exists())
   value=host_runner.publish_release(job,d/'job',launch,ident,ctrl,wd,'plan',wh)
   self.assertTrue(worker.validate_release(job,launch,value,cid[:12],d/'job/launch.json','plan'))
   with self.assertRaises(ValueError):worker.validate_release(job,launch,value,'wrong',d/'job/launch.json','plan')
   self.assertFalse((d/'job/release.pending').exists())
 def test_raw_output_remains_when_generation_followup_fails(self):
  job={'task':'t','qid':'1','arm':'author'}; ref={'processor':{'input_ids':{'shape':[1,2]}}};p={'references':{('t','1','author'):ref}}
  def continuation(prepared,job,row,e1,emit):
   emit([[1,2,3]])
   raise RuntimeError('after generation')
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaises(RuntimeError):worker.run_one(job,t,p,{}, {},continuation,lambda *a:None)
   self.assertEqual(json.loads((Path(t)/'raw.json').read_text())['full_ids'],[1,2,3])
   self.assertTrue(json.loads((Path(t)/'worker_failure.json').read_text())['raw_exists'])
   self.assertFalse((Path(t)/'record.json').exists())
 def test_duplicate_raw_rejected_and_preserved(self):
  job={'task':'t','qid':'1','arm':'author'};ref={'processor':{'input_ids':{'shape':[1,1]}}};p={'references':{('t','1','author'):ref}}
  def continuation(a,b,c,d,emit):emit([[1,2]]);emit([[1,3]])
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaises(ValueError):worker.run_one(job,t,p,{}, {},continuation,lambda *a:None)
   self.assertEqual(json.loads((Path(t)/'raw.json').read_text())['full_ids'],[1,2])
 def test_gpu_idle_and_supervisor_before_directory(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t)/'job';calls=[]
   with self.assertRaises(ValueError):host_runner.bounded_one({},d,[], 'n',{},'p','x','s',lambda *a,**k: calls.append('launch'),lambda:[1],lambda x:x,lambda:None,1)
   self.assertFalse(d.exists());self.assertEqual(calls,[])
   with self.assertRaises(RuntimeError):host_runner.bounded_one({},d,[], 'n',{},'p','x','s',lambda *a,**k: calls.append('launch'),lambda:[],lambda x:x,lambda:(_ for _ in ()).throw(RuntimeError('bad supervisor')),1)
   self.assertFalse(d.exists());self.assertEqual(calls,[])
 def test_real_bounded_interface_keeps_failure(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t)/'job'; wd=Path(t)/'worker';wd.write_text('x');sha=hashlib.sha256(wd.read_bytes()).hexdigest();cid='b'*64;job={'order':0};ctrl={'pid':1}
   def bounded(cmd,name,timeout,pids,ident,jobdir,require_gpu_idle,on_identity):
    on_identity({'container_id':cid,'host_init':{'pid':10,'start_ticks':11,'boot_id':'boot'},'host_cgroup':'/docker/'+cid,'host_argv':[str(wd),'--job-index','0']})
    (jobdir/'exit.json').write_text(json.dumps({'returncode':1,'container_exit_code':1,'container_stopped_verified':True,'gpu_pids_after':[],'timed_out':False,'interrupted':None,'client_exited_while_container_running':False,'failure_type':None}))
    return 1
   with self.assertRaises(ValueError):host_runner.bounded_one(job,d,['docker'],'n',ctrl,'p',wd,sha,bounded,lambda:[],lambda x:x,lambda:None,10)
   self.assertTrue((d/'release.json').exists());self.assertTrue((d/'exit.json').exists())
if __name__=='__main__':unittest.main()

class ControllerTest(unittest.TestCase):
 def jobs(self):
  import controller
  jobs=[]
  for p, reps in [('warmup',[0]),('measurement',[0,1,2])]:
   for image in range(3):
    for arm in ('author','wavefront'):
     for rep in reps: jobs.append({'task':str(image),'qid':str(image),'arm':arm,'phase':p,'repeat':rep,'order':len(jobs)})
  return jobs
 def test_all_24_and_duplicate_process_rejected(self):
  import controller
  jobs=self.jobs(); result=lambda j:{'container_identity':{'container_id':format(j['order']+1,'064x'),'host_init':{'pid':100+j['order'],'start_ticks':1,'boot_id':'b'}}}
  with tempfile.TemporaryDirectory() as t:
   out=Path(t)/'one'; report=controller.run_all(jobs,out,lambda j,p:result(j),lambda *a:None,lambda:None,1000,1,1,lambda:0)
   self.assertEqual(report['measured'],18);self.assertEqual(json.loads((out/'controller.jsonl').read_text().splitlines()[-1])['event'],'all_completed')
   def bad(j,p):
    x=result(j)
    if j['order']==8:x['container_identity']['host_init']['pid']=100
    return x
   out=Path(t)/'two'
   with self.assertRaises(ValueError):controller.run_all(jobs,out,bad,lambda *a:None,lambda:None,1000,1,1,lambda:0)
   self.assertEqual(json.loads((out/'controller.jsonl').read_text().splitlines()[-1])['event'],'controller_failed')
 def test_bad_budget_and_schedule_do_not_create_output(self):
  import controller
  with tempfile.TemporaryDirectory() as t:
   out=Path(t)/'x'
   with self.assertRaises(ValueError):controller.run_all(self.jobs(),out,lambda *a:None,lambda *a:None,lambda:None,24,1,1,lambda:0)
   self.assertFalse(out.exists())
   j=self.jobs();j[10]=dict(j[9],order=10)
   with self.assertRaises(ValueError):controller.validate_jobs(j)
