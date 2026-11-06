import json,tempfile,unittest
from pathlib import Path
import recovery
class FakeGuard:
 def __init__(self,name,nonce,command):self.cid='b'*64;self.running=True;self.ops=[]
 def recover_owned_container(self):return self.cid
 def inspect(self,cid):return {'container_id':cid,'running':self.running,'pid':42 if self.running else 0}
 def stop_container(self,cid):self.ops.append('stop');raise RuntimeError('stop failed')
 def kill_container(self,cid):self.ops.append('kill');self.running=False
class RecoveryTest(unittest.TestCase):
 def test_stop_failure_falls_back_to_owned_kill(self):
  job={'order':0};nonce='a'*32
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'launch.json').write_text(json.dumps({'job':job,'plan_sha256':'p','nonce':nonce,'docker_create':['docker','create']}));v=recovery.recover(job,p,'p',FakeGuard)
   self.assertEqual(v['status'],'stopped');self.assertTrue(v['errors']);self.assertIn('sampler_status',v)
 def test_wrong_job_never_touches_container(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'launch.json').write_text(json.dumps({'job':{'order':1},'plan_sha256':'p','nonce':'a'*32,'docker_create':[]}))
   with self.assertRaises(ValueError):recovery.recover({'order':0},p,'p',FakeGuard)
 def test_sampler_signals_only_matching_owned_supervisor(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t);mon=d/'sampler_monitor';mon.mkdir();worker=['python','sampler','--directory',str(d)];supervisor=['python','supervise','--','python','sampler','--directory',str(d)]
   (d/'launch.json').write_text(json.dumps({'sampler_worker':worker,'sampler_supervisor':supervisor}));(mon/'started.json').write_text(json.dumps({'worker_pid':42,'process_group':42,'command':worker}))
   proc=d/'proc';(proc/'42').mkdir(parents=True);(proc/'17').mkdir();(proc/'42/cmdline').write_bytes(b'\0'.join(x.encode() for x in worker)+b'\0');(proc/'17/cmdline').write_bytes(b'\0'.join(x.encode() for x in supervisor)+b'\0');(proc/'42/stat').write_text('42 (python) S 17 42')
   calls=[]
   def kill(pid,sig):calls.append(pid);(mon/'terminal.json').write_text(json.dumps({'worker_pid':42,'remaining_process_group':[]}))
   result=recovery.stop_sampler(d,proc=proc,kill=kill,getsid=lambda pid:pid,clock=lambda:0,sleep=lambda _:None)
   self.assertEqual(result['status'],'stopped');self.assertEqual(calls,[17])
 def test_sampler_missing_or_wrong_identity_is_unknown(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t);(d/'launch.json').write_text(json.dumps({'sampler_worker':['w'],'sampler_supervisor':['s']}));(d/'sampler_monitor').mkdir();calls=[]
   result=recovery.stop_sampler(d,proc=d,kill=lambda *a:calls.append(a))
   self.assertEqual(result['status'],'unknown');self.assertEqual(calls,[])
