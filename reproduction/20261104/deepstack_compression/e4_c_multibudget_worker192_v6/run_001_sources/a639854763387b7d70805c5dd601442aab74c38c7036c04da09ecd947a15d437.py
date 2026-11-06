import json,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import gpu_launch as g
class Tests(unittest.TestCase):
 def setUp(self):
  self.rows=[{'task':str(i),'qid':str(i),'group_id':'g'+str(i),'split':'DEV'} for i in range(3)]
  self.jobs=g.boot.pin('pilot_contract',g.HERE/'pilot_contract.py').make_schedule(self.rows)
 def rows_for(self,job):
  def r(event,**kw):return {'event':event,'job':job,**kw}
  def value(observed):return {'processor_verified':True,'tokens':[9],'eos':True,'truncated':False,'dynamic_k':256 if observed else None,'count_observer_enabled':observed}
  return [r('model_loading'),r('model_ready'),r('started',phase='warmup'),r('raw_generation',phase='warmup',ids=[[9]]),r('completed',phase='warmup',result=value(False)),r('started',phase='measured'),r('raw_generation',phase='measured',ids=[[9]]),r('completed',phase='measured',result=value(True)),r('job_complete')]
 def test_supervisor_exact_limits(self):
  pid=os.getpid();cwd=os.getcwd();start={'worker_pid':pid,'process_group':pid,'timeout_seconds':11700,'grace_seconds':120,'snapshot_seconds':3600,'poll_seconds':5,'cwd':cwd,'command':['python',str(g.HERE/'gpu_launch.py')]}
  self.assertTrue(g.validate_supervisor(start,pid,pid,cwd));start['timeout_seconds']=11701
  with self.assertRaises(ValueError):g.validate_supervisor(start,pid,pid,cwd)
 def test_worker_phase_loss_and_k_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'journal';rows=self.rows_for(self.jobs[0]);save=lambda: p.write_text(''.join(json.dumps(x)+'\n' for x in rows));save()
   self.assertTrue(g.validate_worker_journal(p,self.jobs[0]));rows[7]['result']['dynamic_k']=None;save()
   with self.assertRaises(ValueError):g.validate_worker_journal(p,self.jobs[0])
   rows[7]['result']['dynamic_k']=256;rows.pop(3);save()
   with self.assertRaises(ValueError):g.validate_worker_journal(p,self.jobs[0])
 def test_release_binds_exact_container_and_controller(self):
  import gpu_worker_candidate as child
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);(folder/'launch.json').write_text('{}');job=self.jobs[0];controller={'pid':100,'start_ticks':4,'boot_id':'b'}
   launch={'controller':controller,'plan_sha256':'p'};cid='a'*64
   identity={'container_id':cid,'host_init':{'pid':9,'start_ticks':8,'boot_id':'b'},'host_cgroup':'/docker/'+cid,'host_argv':['python',str(g.HERE/'gpu_worker_candidate.py'),'--job-index',str(job['order'])]}
   with patch.object(g,'validate_supervisor',return_value=True):
    rel=g.release_worker(job,folder,launch,identity,controller,{})
   self.assertTrue(child.validate_release(job,launch,rel,cid[:12],g.pre.sha(folder/'launch.json'),'p'))
   with self.assertRaises(ValueError):child.validate_release(job,launch,rel,'b'*12,g.pre.sha(folder/'launch.json'),'p')
 def test_nine_jobs_through_bounded_callback(self):
  import json
  class Runtime:
   def gpu_pids(self):return []
   def identity(self,pid):return {'pid':pid,'start_ticks':4,'boot_id':'b'}
   def container_command(self,author,args,gpu,name):return ['docker','run','--rm',*args]
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);log=out/'controller.jsonl';count=[]
   def bounded(argv,name,timeout,gpu_pids,identity,jobdir,require_gpu_idle,on_identity):
    index=int(jobdir.name.split('_')[1]);job=self.jobs[index];cid=('%064x'%(index+1));ctl=Runtime().identity(os.getpid())
    observed={'container_id':cid,'host_init':{'pid':index+100,'start_ticks':index+1,'boot_id':'b'},'host_cgroup':'/docker/'+cid,'host_argv':['python',str(g.HERE/'gpu_worker_candidate.py'),'--job-index',str(index)]}
    on_identity(observed)
    observations=jobdir/'observations';observations.mkdir();(observations/'journal.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in self.rows_for(job)))
    result={'returncode':0,'container_exit_code':0,'container_stopped_verified':True,'gpu_pids_after':[],'timed_out':False,'interrupted':None,'client_exited_while_container_running':False,'failure_type':None,'container_identity':observed}
    (jobdir/'exit.json').write_text(json.dumps(result));count.append(index);return 0
   real_read=g.pre.read
   def read(path):
    if path==g.HERE/'monitor_pilot001/started.json':return {}
    return real_read(path)
   with patch.object(g,'run_bounded',side_effect=bounded),patch.object(g,'validate_supervisor',return_value=True),patch.object(g.pre,'read',side_effect=read),patch.object(g.pre,'sha',return_value='p'):
    self.assertEqual(g.run_jobs({'jobs':self.jobs},Runtime(),out,time.time()+50000,log),9)
   self.assertEqual(count,list(range(9)));self.assertTrue(all((out/('job_%03d'%i)/'release.json').exists() for i in range(9)))
 def test_first_busy_gpu_before_any_directory(self):
  class Runtime:
   def gpu_pids(self):return [1]
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);log=[]
   with self.assertRaises(ValueError):g.run_jobs({'jobs':self.jobs},Runtime(),p,time.time()+50000,p/'journal')
   self.assertEqual(list(p.iterdir()),[])
 def test_main_rejects_before_gpu_or_output_when_no_review(self):
  with patch.object(g.adm,'require',side_effect=ValueError('review missing')),patch.object(g,'run_jobs') as run:
   with self.assertRaisesRegex(ValueError,'review missing'):g.main()
   run.assert_not_called()
if __name__=='__main__':unittest.main(verbosity=2)
