import subprocess,sys,os,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_gpu_is_hard_locked(self):
  import preflight_gate
  with self.assertRaises(RuntimeError):preflight_gate.gpu()
 def test_no_review_rejects_before_prior_reads(self):
  import preflight_gate
  if (D/'preflight_plan001.json').exists():self.skipTest('plan now exists')
  old=preflight_gate.prior
  try:
   preflight_gate.prior=lambda:(_ for _ in ()).throw(AssertionError('unexpected read'))
   with self.assertRaises(ValueError):preflight_gate.preflight()
  finally:preflight_gate.prior=old
 def test_same_name_fake_gate_rejected(self):
  code="import sys,types; m=types.ModuleType('_pixel_t020_runtime_preflight_gate');m.__file__='/tmp/fake.py';sys.modules[m.__name__]=m;import preflight_entry\ntry:preflight_entry.load(m.__name__,preflight_entry.D/'preflight_gate.py')\nexcept ValueError:pass\nelse:raise AssertionError('accepted foreign gate')"
  p=subprocess.run([sys.executable,'-c',code],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=30)
  self.assertEqual(p.returncode,0,p.stderr)
if __name__=='__main__':unittest.main()

class ContainerIdentityTests(unittest.TestCase):
 def sample(self):
  import preflight_entry as e
  cid='b'*64; out=e.D/'run_preflight001/worker_preflight.json';approval={'plan_sha256':'p'}
  launch={'approval':approval,'command':['docker','run',str(e.D/'preflight_entry.py'),'--out',str(out)]}
  ident={'container_id':cid,'image':'sha256:image','host_init':{'pid':9,'start_ticks':10,'boot_id':'boot'},'host_argv':['python',str(e.D/'preflight_entry.py'),'--out',str(out)],'host_cgroup':'/docker/'+cid}
  return e,approval,launch,ident,out
 def test_accepts_matching_container_rejects_host_and_spoof(self):
  e,a,l,i,o=self.sample();self.assertEqual(e.verify_identity(a,l,i,'b'*12,'/docker/'+('b'*64),'sha256:image',o),'b'*64)
  for mutate in ('hostname','cgroup','self_cgroup','argv','image','approval'):
   import copy
   aa=copy.deepcopy(a);ll=copy.deepcopy(l);ii=copy.deepcopy(i);hostname='b'*12;self_cgroup='/docker/'+('b'*64);image='sha256:image'
   if mutate=='hostname':hostname='host'
   elif mutate=='cgroup':ii['host_cgroup']='/not-container'
   elif mutate=='self_cgroup':self_cgroup='/host/no-container'
   elif mutate=='argv':ii['host_argv']=['python','/tmp/evil']
   elif mutate=='image':image='sha256:other'
   else:ll['approval']={'wrong':1}
   with self.assertRaises(ValueError,msg=mutate):e.verify_identity(aa,ll,ii,hostname,self_cgroup,image,o)
 def test_both_real_entries_hash_gate_before_loading(self):
  import preflight_entry,preflight_launch,tempfile
  from pathlib import Path
  for m in (preflight_entry,preflight_launch):
   self.assertEqual(m.check_gate(m.D/'preflight_gate.py'),m.D/'preflight_gate.py')
   with tempfile.TemporaryDirectory() as t:
    p=Path(t)/'fake.py';marker=Path(t)/'executed';p.write_text('open('+repr(str(marker))+',"w").write("bad")')
    with self.assertRaises(ValueError):m.check_gate(p)
    self.assertFalse(marker.exists())
 def test_incomplete_identity_is_bounded(self):
  import preflight_entry,tempfile
  from pathlib import Path
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'identity.json';p.write_text('{')
   with self.assertRaises(TimeoutError):preflight_entry.wait_identity(p,timeout=0)

class UpstreamGateTests(unittest.TestCase):
 def test_upstream_gate_hash_before_exec(self):
  import preflight_gate,tempfile
  from pathlib import Path
  self.assertIsNotNone(preflight_gate.load('_test_prior_gate',preflight_gate.P/'preflight_gate.py',preflight_gate.UPSTREAM_GATE_SHA))
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'gate.py';marker=Path(t)/'executed';p.write_text('open('+repr(str(marker))+',"w").write("bad")')
   with self.assertRaises(ValueError):preflight_gate.load('_test_fake_prior_gate',p,preflight_gate.UPSTREAM_GATE_SHA)
   self.assertFalse(marker.exists())

class RealPriorChainTests(unittest.TestCase):
 def test_actual_prior_chain_imports_without_alias_collision(self):
  import preflight_gate
  _,q=preflight_gate.prior()
  self.assertEqual(q['status'],'passed')
