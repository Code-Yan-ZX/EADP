"""Candidate host adapter for memory lifecycle. NO production CLI/admission.

A future review must bind exact Docker configuration, source snapshots,
expected pre-CUDA worker implementation and real no-GPU integration before use.
All subprocesses are bounded; retained raw observations remain mandatory.
"""
import json,os,subprocess,time
from pathlib import Path
from controller_io import identity_from_inspect,publish,publish_stopped,host_alive
from sampler_supervisor import SamplerSupervisor

class ControllerAdapter:
 def __init__(self,guard,directory,mode,nonce,gpu_uuid,max_gap_ns,worker_timeout,sampler_command,sampler_worker_command,sampler_monitor,ready_timeout=120):
  self.guard=guard;self.directory=Path(directory).resolve();self.mode=mode;self.nonce=nonce;self.gpu_uuid=gpu_uuid;self.max_gap_ns=max_gap_ns
  if not self.directory.is_dir() or not 0<worker_timeout<=28800 or not 0<ready_timeout<=1800:raise ValueError('invalid controller directory or timeout')
  self.worker_timeout=worker_timeout;self.ready_timeout=ready_timeout;self.identity=None;self.ready=None
  self.sampler=SamplerSupervisor(sampler_command,sampler_worker_command,sampler_monitor,self.directory/'sampler_ready.json')
 def create_container(self):return self.guard.create_container()
 def recover_owned_container(self):return self.guard.recover_owned_container()
 def start_container(self,cid):self.guard.start_container(cid)
 def observe_running(self,cid):
  identity=identity_from_inspect(self.guard.inspect(cid),self.mode,self.nonce,self.gpu_uuid,self.max_gap_ns)
  if not host_alive(identity):raise ValueError('worker identity not alive')
  publish(self.directory/'expected.json',identity);self.identity=identity;return identity
 def wait_worker_ready(self,identity):
  path=self.directory/'worker_ready.json';deadline=time.monotonic()+self.ready_timeout
  while time.monotonic()<deadline:
   if not host_alive(identity):raise ValueError('worker exited before ready')
   try:return json.loads(path.read_text())
   except FileNotFoundError:time.sleep(.1)
  raise TimeoutError('worker readiness timeout')
 def start_sampler(self,identity):self.sampler.start();return self.sampler
 def wait_sampler_ready(self,sampler):self.ready=sampler.wait_ready(min(self.ready_timeout,300));return self.ready
 def release_worker(self,identity,release):publish(self.directory/'release.json',release)
 def wait_worker(self,cid):
  p=subprocess.Popen(['docker','wait',cid],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
  try:
   out,err=p.communicate(timeout=self.worker_timeout)
   if p.returncode!=0 or not out.strip().isdigit():raise RuntimeError('docker wait failed: '+err.strip()[:500])
   return {'exit_code':int(out.strip())}
  finally:
   if p.poll() is None:
    p.terminate()
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
 def inspect(self,cid):return self.guard.inspect(cid)
 def publish_stopped(self,identity,inspection):return publish_stopped(self.directory,identity,inspection,host_alive)
 def wait_sampler(self,sampler):
  terminal=sampler.wait_finished(60);events=[json.loads(line) for line in (self.directory/'samples.jsonl').read_text().splitlines() if line]
  samples=[x for x in events if x.get('type')=='sample'];summary=[x for x in events if x.get('type')=='summary'];started=[x for x in events if x.get('status')=='sampling']
  if len(started)!=1 or len(summary)!=1 or events[0]!=started[0] or events[-1]!=summary[0] or summary[0].get('samples')!=len(samples) or not samples or self.ready!={'started':started[0],'first_sample':samples[0]} or any(x.get('type')=='failure' for x in events):raise ValueError('sampler journal inconsistent')
  return samples,terminal
 def stop_container(self,cid):self.guard.stop_container(cid)
 def kill_container(self,cid):self.guard.kill_container(cid)
 def stop_sampler(self,sampler):sampler.stop()
 def persist_result(self,result):publish(self.directory/'controller_result.json',result)
