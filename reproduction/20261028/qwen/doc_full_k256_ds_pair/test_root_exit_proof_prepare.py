"""Synthetic-only checks. No model, GPU, Docker, labels or real run outputs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import root_exit_proof_prepare as p

class ProofTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name); (self.root/'execution').mkdir()
  self.boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
  self.roles=[dict(pid=i,start_ticks=100+i,session_id=i,boot_id=self.boot) for i in (101,102,103)]
  self.put('plan.json',dict(run_id='Qwen_Doc5349_K256_native_selection_prefill_DS_pair')); self.plan=p.sha(self.root/'plan.json')
  self.put('root_gpu_allocation.json',dict(run_id='Qwen_Doc5349_K256_native_selection_prefill_DS_pair',authorization='explicit_root_gpu_allocation',single_attempt=True,automatic_retry=False,boot_id=self.boot,plan_sha256=self.plan)); self.alloc=p.sha(self.root/'root_gpu_allocation.json')
  self.cid='a'*64; identity=dict(container_id=self.cid,host_init=self.roles[2],host_cgroup='synthetic')
  self.put('launch.claim.json',dict(controller=self.roles[0],plan_sha256=self.plan,allocation_sha256=self.alloc))
  self.put('root_launch_process_snapshot.json',dict(root_exec_handle=86476,controller=self.roles[0],docker_client=self.roles[1],container_identity=identity,plan_sha256=self.plan,allocation_sha256=self.alloc))
  self.put('execution/runtime.json',dict(example='metadata'))
  keys=['prediction_sha256','observation_sha256','ordered_prediction_sha256','ordered_observation_sha256','native_anchors_sha256']; digests={k:'x' for k in keys}
  complete=dict(success=True,n_questions=5349,n_generations=10698,native_anchors=32,GT_read=False,scoring_started=False,plan_sha256=self.plan,**digests)
  self.put('execution/native_complete.json',complete)
  v=dict(success=True,n_questions=5349,n_generations=10698,native_anchors=32,CPU_only=True,model_loaded=False,CUDA_initialized=False,GT_read=False,scoring_started=False,plan_sha256=self.plan,native_complete_sha256=p.sha(self.root/'execution/native_complete.json'),runtime_sha256=p.sha(self.root/'execution/runtime.json'),**digests)
  for k in ('processor_identity_checks','selection_checks','prefill_intervention_checks','independent_token_redecodes'): v[k]=10698
  self.put('execution/independent_verification.json',v)
  self.put('execution/release.json',dict(success=True,native_container_exited=True,plan_sha256=self.plan,allocation_sha256=self.alloc,independent_verification_sha256=p.sha(self.root/'execution/independent_verification.json')))
  self.put('state.json',dict(status='complete',native_exit_code=0,CPU_verifier_exit_code=0,original_container_exited=True,GPU_pid_absent=True,plan_sha256=self.plan,allocation_sha256=self.alloc,release_sha256=p.sha(self.root/'execution/release.json'),controller=self.roles[0],docker_client=self.roles[1],container_identity=identity))
 def put(self,name,obj): (self.root/name).write_text(json.dumps(obj))
 def call(self,**kw):
  opts=dict(root=self.root,handle=86476,observed_exit=0,expected_plan_sha=self.plan,get_process=lambda pid:None,inspect=lambda cid:dict(running=False,container_absent=True),get_gpu_pids=lambda:[]); opts.update(kw);return p.prepare(**opts)
 def test_complete_evidence_and_expected_schema(self):
  out=self.call(); self.assertTrue(out['all_original_processes_exited']); self.assertEqual(out['root_owned_exec_handle'],86476);self.assertEqual([x['role'] for x in out['process_checks']],['controller','docker_client','host_init']); self.assertFalse((self.root/'root_terminal_process_proof.json').exists())
 def test_root_exit_and_handle_rejected(self):
  for kw in ({'handle':1},{'observed_exit':1},{'expected_plan_sha':'0'*64}):
   with self.subTest(kw=kw),self.assertRaises(ValueError):self.call(**kw)
 def test_original_process_live_rejected_and_reused_pid_allowed(self):
  with self.assertRaises(ValueError):self.call(get_process=lambda pid:dict(pid=pid,start_ticks=101+100,boot_id=self.boot))
  self.assertTrue(self.call(get_process=lambda pid:dict(pid=pid,start_ticks=1,boot_id=self.boot))['all_original_processes_exited'])
 def test_gpu_pid_and_running_container_rejected(self):
  for kw in ({'get_gpu_pids':lambda:[103]},{'inspect':lambda cid:dict(running=True)}):
   with self.assertRaises(ValueError):self.call(**kw)
 def test_failed_controller_and_failure_record_rejected(self):
  state=p.read(self.root/'state.json');state['status']='running';self.put('state.json',state)
  with self.assertRaises(ValueError):self.call()
  state['status']='complete';self.put('state.json',state);self.put('failure.json',{})
  with self.assertRaises(ValueError):self.call()
 def test_release_and_verifier_tamper_rejected(self):
  v=p.read(self.root/'execution/independent_verification.json');v['n_generations']=1;self.put('execution/independent_verification.json',v)
  with self.assertRaises(ValueError):self.call()
 def test_docker_errors_are_not_absence(self):
  class R: returncode=1;stdout='';stderr='Cannot connect to Docker daemon'
  with patch.object(p.subprocess,'run',return_value=R()):
   with self.assertRaises(ValueError):p.docker_not_running(self.cid)
 def test_never_opens_predictions_or_ground_truth(self):
  with patch.object(p,'read',wraps=p.read) as wrapped:
   self.call(); names=[str(c.args[0]) for c in wrapped.call_args_list]
  self.assertFalse(any('prediction' in s or 'canonical' in s for s in names))

if __name__=='__main__': unittest.main(verbosity=2)
