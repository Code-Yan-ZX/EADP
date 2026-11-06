import copy,os,unittest
from pathlib import Path
import admission
class Tests(unittest.TestCase):
 def setUp(self):
  self.paths={k:Path('/tmp')/(k+'.json') for k in ('cpu','cs','ct','pref','ps','pt','launch','exit','review','reviewer','handshake','hl','he','release','hs','ht')};p=self.paths
  self.cs={'worker_pid':1,'cwd':'/','command':['python',str(admission.HERE/'cpu_check.py')]};self.ct={'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':1}
  self.cpu={'status':'passed','plan_sha256':'plan','GT_opened':False,'gpu_used':False}
  self.pref={'status':'passed','source_plan_sha256':'plan','cuda_initialized':False,'model_loaded':False,'weight_sha256_verified':4,'same_arm_references':9,'GT_opened':False}
  self.launch={'plan_sha256':'plan','controller':{'pid':2}};self.exit={'controller':{'pid':2},'returncode':0,'container_exit_code':0,'timed_out':False,'interrupted':None,'failure_type':None,'client_exited_while_container_running':False,'container_stopped_verified':True,'last_inspect':{'device_requests':None,'host_devices':[]}}
  self.ps={'worker_pid':2,'cwd':'/','command':['python',str(admission.HERE/'preflight_launch.py')]};self.pt={'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':2}
  self.review={'status':'passed','independent_review':True,'GT_opened':False,'scoring_performed':False,'plan_sha256':'plan','cpu_sha256':'cpu','cpu_started_sha256':'cs','cpu_terminal_sha256':'ct','preflight_launch_sha256':'launch','preflight_sha256':'pref','preflight_started_sha256':'ps','preflight_terminal_sha256':'pt','preflight_exit_sha256':'exit','reviewer_sha256':'reviewer'}
  self.hs={'worker_pid':4,'cwd':'/','command':['python',str(admission.HERE/'handshake_launch.py')]};self.ht={'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':4}
  self.hl={'plan_sha256':'plan','controller':{'pid':4}};self.he={'controller':{'pid':4},'returncode':0,'container_exit_code':0,'timed_out':False,'interrupted':None,'failure_type':None,'client_exited_while_container_running':False,'container_stopped_verified':True,'last_inspect':{'device_requests':None,'host_devices':[]},'container_identity':{'container_id':'c'}}
  self.handshake={'status':'passed','cuda_initialized':False,'model_loaded':False,'launch_sha256':'hl','release_sha256':'release','container_id':'c'}
  self.review.update(handshake_sha256='handshake',handshake_launch_sha256='hl',handshake_exit_sha256='he',handshake_started_sha256='hs',handshake_terminal_sha256='ht',handshake_release_sha256='release')
  self.rs={'worker_pid':3,'cwd':'/tmp','command':['python',str(p['reviewer']),'--out',str(p['review'])]};self.rt={'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':3}
 def run_validate(self):
  return admission.validate('plan',self.cpu,self.cs,self.ct,self.pref,self.launch,self.exit,self.ps,self.pt,self.handshake,self.hl,self.he,self.hs,self.ht,self.review,self.rs,self.rt,self.paths,lambda p:Path(p).stem)
 def test_success_and_preflight_tampering(self):
  self.assertTrue(self.run_validate())
  for key,value in [('model_loaded',True),('cuda_initialized',True),('weight_sha256_verified',3),('same_arm_references',8)]:
   old=self.pref[key];self.pref[key]=value
   with self.assertRaises(ValueError):self.run_validate()
   self.pref[key]=old
 def test_bad_containers_and_supervisors_refused(self):
  for key,value in [('container_stopped_verified',False),('timed_out',True),('client_exited_while_container_running',True)]:
   old=self.exit[key];self.exit[key]=value
   with self.assertRaises(ValueError):self.run_validate()
   self.exit[key]=old
  self.ct['worker_returncode']=1
  with self.assertRaises(ValueError):self.run_validate()
 def test_handshake_requires_device_free_container_and_clean_exit(self):
  self.assertTrue(self.run_validate());self.he['last_inspect']['host_devices']=['gpu']
  with self.assertRaises(ValueError):self.run_validate()
  self.he['last_inspect']['host_devices']=[];self.handshake['container_id']='wrong'
  with self.assertRaises(ValueError):self.run_validate()
 def test_review_provenance_and_exit_required(self):
  self.review['cpu_sha256']='wrong'
  with self.assertRaises(ValueError):self.run_validate()
  self.review['cpu_sha256']='cpu';self.rs['command'][-1]='/tmp/other.json'
  with self.assertRaises(ValueError):self.run_validate()
 def test_missing_real_proof_fails(self):
  with self.assertRaises((FileNotFoundError,ValueError)):admission.require()
if __name__=='__main__':unittest.main(verbosity=2)
