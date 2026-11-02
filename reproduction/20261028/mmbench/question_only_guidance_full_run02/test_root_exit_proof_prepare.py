"""Synthetic proof checks; no actual session, device, predictions or GT."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import root_exit_proof_prepare as p

class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.out=self.root/'execution';self.out.mkdir()
  self.boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
  self.ids=[dict(pid=i,start_ticks=100+i,boot_id=self.boot) for i in (101,102,103)]
  self.put('plan.json',dict(run_id='MMB_full_source_vs_question_only_run02'));self.plan=p.sha(self.root/'plan.json');self.patch=patch.object(p,'EXPECTED_PLAN',self.plan);self.patch.start();self.addCleanup(self.patch.stop)
  self.put('root_gpu_allocation.json',dict(run_id='MMB_full_source_vs_question_only_run02',authorization='explicit_root_gpu_allocation',single_attempt=True,automatic_retry=False,plan_sha256=self.plan,boot_id=self.boot));self.alloc=p.sha(self.root/'root_gpu_allocation.json')
  self.put('launch.claim.json',dict(plan_sha256=self.plan,allocation_sha256=self.alloc,controller=self.ids[0]))
  self.put('execution/worker_identity.json',dict(plan_sha256=self.plan,allocation_sha256=self.alloc,identity=self.ids[1],parent=self.ids[0]))
  (self.out/'worker.log').write_text('worker');(self.out/'verifier.log').write_text('verifier');(self.out/'verifier.cid').write_text('a'*64)
  self.put('execution/native_release.json',dict(status='native_exited_validation_pending',native_exit_code=0,original_worker_exited=True,GPU_pid_absent=True,worker_log_sha256=p.sha(self.out/'worker.log'),plan_sha256=self.plan,allocation_sha256=self.alloc,controller=self.ids[0],worker=self.ids[1]))
  self.put('execution/state.json',dict(status='complete',native_exit_code=0,verifier_exit_code=0,verifier_container_absent=True,verifier_log_sha256=p.sha(self.out/'verifier.log'),plan_sha256=self.plan,allocation_sha256=self.alloc,controller=self.ids[0],worker=self.ids[1],verifier_client=self.ids[2]))
  self.put('execution/anchor_gate.json',dict(passed=True,n_rows=60,n_generations=120,all_anchors_before_remaining=True));self.put('execution/runtime.json',dict(dummy=1))
  ordered={arm:{kind:dict(sha256='sha-'+arm+'-'+kind) for kind in ('predictions','observations')} for arm in ('source','question_only')}
  self.put('execution/native_complete.json',dict(status='complete',n_rows=4377,n_generations=8754,scoring=False,plan_sha256=self.plan,anchor_gate=dict(sha256=p.sha(self.out/'anchor_gate.json')),runtime=dict(sha256=p.sha(self.out/'runtime.json')),ordered=ordered))
  self.put('execution/generation_verification.json',dict(success=True,n_rows=4377,n_generations=8754,n_anchor_rows=60,n_groups=1176,CPU_only=True,model_loaded=False,CUDA_initialized=False,GT_read=False,scoring_started=False,plan_sha256=self.plan,native_complete_sha256=p.sha(self.out/'native_complete.json'),native_release_sha256=p.sha(self.out/'native_release.json'),anchor_gate_sha256=p.sha(self.out/'anchor_gate.json'),runtime_sha256=p.sha(self.out/'runtime.json'),prediction_sha256={arm:ordered[arm]['predictions']['sha256'] for arm in ordered},observation_sha256={arm:ordered[arm]['observations']['sha256'] for arm in ordered}))
 def put(self,path,value):(self.root/path).write_text(json.dumps(value))
 def call(self,**kwargs):
  options=dict(root=self.root,session=86075,exit_code=0,via='functions.write_stdin',process_fn=lambda pid:None,container_fn=lambda cid:dict(running=False,absent=True),gpu_fn=lambda:[]);options.update(kwargs);return p.prepare(**options)
 def test_full_valid_and_required_score_schema(self):
  r=self.call();self.assertEqual(r['root_pty_session'],86075);self.assertEqual(r['root_pty_exit_observed_by'],'functions.write_stdin');self.assertEqual([x['role'] for x in r['process_checks']],['controller','worker','verifier_client']);self.assertFalse((self.root/'root_exit_proof.json').exists())
 def test_explicit_actual_exit_required(self):
  for kw in (dict(session=1),dict(exit_code=1),dict(via='other')):
   with self.assertRaises(ValueError):self.call(**kw)
 def test_each_original_identity_and_reused_pid(self):
  for target in self.ids:
   with self.assertRaises(ValueError):self.call(process_fn=lambda pid,target=target:target if pid==target['pid'] else None)
  self.assertTrue(self.call(process_fn=lambda pid:dict(pid=pid,start_ticks=0,boot_id=self.boot))['verifier_container_not_running'])
 def test_gpu_and_container_running(self):
  with self.assertRaises(ValueError):self.call(gpu_fn=lambda:[self.ids[1]['pid']])
  with self.assertRaises(ValueError):self.call(container_fn=lambda cid:dict(running=True))
 def test_failed_state_and_failure_file(self):
  s=p.read(self.out/'state.json');s['status']='native_running';self.put('execution/state.json',s)
  with self.assertRaises(ValueError):self.call()
  s['status']='complete';self.put('execution/state.json',s);self.put('execution/worker_failure.json',{})
  with self.assertRaises(ValueError):self.call()
 def test_frozen_chain_tamper(self):
  for name in ('native_complete.json','native_release.json','anchor_gate.json','runtime.json'):
   path=self.out/name;old=path.read_bytes();path.write_bytes(old+b' ')
   try:
    with self.subTest(name=name),self.assertRaises(ValueError):self.call()
   finally:path.write_bytes(old)
 def test_docker_daemon_failure_not_container_absence(self):
  class R: returncode=1;stdout='';stderr='Cannot connect to Docker daemon'
  with patch.object(p.subprocess,'run',return_value=R()):
   with self.assertRaises(ValueError):p.container_check('a'*64)
 def test_never_reads_prediction_or_ground_truth(self):
  with patch.object(p,'read',wraps=p.read) as r:self.call();self.assertFalse(any('predictions.jsonl' in str(c.args[0]) or 'mmbench_dev_' in str(c.args[0]) for c in r.call_args_list))
if __name__=='__main__':unittest.main(verbosity=2)
