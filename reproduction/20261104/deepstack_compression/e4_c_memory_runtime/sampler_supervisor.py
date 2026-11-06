"""Host sampler supervision primitive; no model/GPU launch authority.

External admission must bind sampler source, expected PID, GPU UUID, command,
timeout and resulting terminal to the Docker controller before release.
"""
import json,os,subprocess,time
from pathlib import Path

def read(path):return json.loads(Path(path).read_text())
class SamplerSupervisor:
 def __init__(self,command,worker_command,monitor_dir,ready_path,popen=subprocess.Popen,clock=time.monotonic,sleep=time.sleep):
  if not command or not all(isinstance(x,str) and x for x in command):raise ValueError('invalid supervisor command')
  if not worker_command or command.count('--')!=1 or command[command.index('--')+1:]!=worker_command:raise ValueError('sampler worker command mismatch')
  self.worker_command=list(worker_command);self.command=command;self.monitor_dir=Path(monitor_dir);self.ready_path=Path(ready_path);self.popen=popen;self.clock=clock;self.sleep=sleep;self.process=None
 def start(self):
  if self.process is not None or self.monitor_dir.exists() or self.ready_path.exists():raise ValueError('sampler paths reused')
  self.process=self.popen(self.command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
 def wait_ready(self,timeout):
  if self.process is None or not 0<timeout<=300:raise ValueError('sampler not started or invalid deadline')
  deadline=self.clock()+timeout
  while self.clock()<deadline:
   if self.process.poll() is not None:raise ValueError('sampler supervisor exited before ready')
   try:
    s=read(self.monitor_dir/'started.json');r=read(self.ready_path)
   except FileNotFoundError:self.sleep(.1);continue
   if type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s.get('process_group')!=s['worker_pid'] or s.get('command')!=self.worker_command:raise ValueError('sampler supervisor identity')
   if not isinstance(r.get('started'),dict) or not isinstance(r.get('first_sample'),dict):raise ValueError('sampler ready evidence')
   return r
  raise TimeoutError('sampler readiness timeout')
 def wait_finished(self,timeout):
  if self.process is None or not 0<timeout<=360:raise ValueError('invalid sampler wait')
  try:rc=self.process.wait(timeout=timeout)
  except subprocess.TimeoutExpired:raise TimeoutError('sampler supervisor did not stop')
  s=read(self.monitor_dir/'started.json');t=read(self.monitor_dir/'terminal.json')
  if rc!=0 or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or t.get('worker_pid')!=s.get('worker_pid') or s.get('command')!=self.worker_command or s.get('process_group')!=s.get('worker_pid'):raise ValueError('sampler did not exit cleanly')
  return t
 def stop(self,timeout=30):
  if self.process is None:return
  if self.process.poll() is None:
   self.process.terminate()
   try:self.process.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    self.process.kill();self.process.wait(timeout=5)
    raise RuntimeError('sampler supervisor required kill; child cleanup unverified')
  try:s=read(self.monitor_dir/'started.json');t=read(self.monitor_dir/'terminal.json')
  except (FileNotFoundError,ValueError) as exc:raise RuntimeError('sampler cleanup terminal unavailable') from exc
  if t.get('worker_pid')!=s.get('worker_pid') or t.get('remaining_process_group')!=[]:raise RuntimeError('sampler child cleanup not verified')
