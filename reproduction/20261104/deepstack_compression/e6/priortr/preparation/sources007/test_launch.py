import unittest
from smoke_launch import exit_code,configure_container
from smoke_worker import validate_preflight_records,HERE,INSTALL
from unittest.mock import patch
from pathlib import Path
class LauncherTests(unittest.TestCase):
 def test_preflight_rejects_bad_container_or_supervisor(self):
  script=str(HERE/'container_preflight003.py');python=str(INSTALL/'env/bin/python');image='sha256:image';ident={'initial':{'id':'id','image_id':image,'path':python,'args':[script],'state':{'Running':True,'Pid':123}},'host_init':{'pid':123,'start_ticks':456,'boot_id':'b'}}
  launch={'command':['docker','USER=eadp','LOGNAME=eadp','CUDA_VISIBLE_DEVICES=','TORCHINDUCTOR_CACHE_DIR='+str(HERE/'container_preflight003_cache/inductor'),'--entrypoint',python,image,script]}
  exited={'status':'passed','returncode':0,'failure':None,'identity':ident,'final_inspect':{'id':'id','image_id':image,'path':python,'args':[script],'state':{'Running':False,'Pid':0,'ExitCode':0}}}
  output={'status':'passed','cuda_initialized':False,'username':'eadp','torch':'2.10.0+cu128','transformers':'5.2.0.dev0','n_frozen_samples':8,'fingerprint_sha256':'h','launcher_sha256':'h'}
  started={'worker_pid':10,'cwd':str(HERE),'command':['python','container_preflight003_host.py']};terminal={'worker_pid':10,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
  with patch('smoke_worker.sha',return_value='h'):
   validate_preflight_records(launch,ident,exited,output,started,terminal)
   for bad in (dict(output,cuda_initialized=True),dict(output,launcher_sha256='wrong')):
    with self.assertRaises(ValueError):validate_preflight_records(launch,ident,exited,bad,started,terminal)
   with self.assertRaises(ValueError):validate_preflight_records(dict(launch,command=launch['command']+['--gpus']),ident,exited,output,started,terminal)
   bad=dict(launch,command=launch['command'][:-2]+['sha256:wrong',script])
   with self.assertRaises(ValueError):validate_preflight_records(bad,ident,exited,output,started,terminal)
   with self.assertRaises(ValueError):validate_preflight_records(launch,ident,exited,output,started,dict(terminal,worker_returncode=1))
 def test_container_user_and_data_disk_cache(self):
  cache=Path('/media/disk2/example-cache');cmd=['docker','run','--env','PYTHONPATH=old','--env','HF_HOME=old','--workdir','/work','image']
  out=configure_container(cmd,cache)
  for item in ('USER=eadp','LOGNAME=eadp','PYTHONPATH='+str(Path(__file__).parent.resolve()),'TORCHINDUCTOR_CACHE_DIR='+str(cache/'inductor'),'TMPDIR='+str(cache)):
   self.assertEqual(out.count(item),1)
  self.assertNotIn('PYTHONPATH=old',out)
  with self.assertRaises(ValueError):configure_container(['docker','run','--env','USER=x','--env','USER=y','--workdir','/work'],cache)
 def test_rejects_every_unclean_exit_even_if_client_zero(self):
  good={'returncode':0,'container_exit_code':0,'timed_out':False,'interrupted':None,'client_exited_while_container_running':False,'gpu_pids_after':[],'container_stopped_verified':True,'container_identity':{'container_id':'x'}}
  self.assertEqual(exit_code(good),0)
  for key,value in [('returncode',1),('container_exit_code',1),('timed_out',True),('interrupted',15),('client_exited_while_container_running',True),('gpu_pids_after',['123']),('gpu_pids_after',None),('container_stopped_verified',False),('container_identity',None)]:
   bad=dict(good);bad[key]=value;self.assertNotEqual(exit_code(bad),0,key)
if __name__=='__main__':unittest.main()
