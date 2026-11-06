"""Filesystem barrier primitives; the Docker controller is not implemented.

The eventual controller must establish real host PID/start/cgroup, preserve the
sampler's source and supervisor identity, and authenticate the stopped event.
These helpers only implement the per-run, one-shot message exchange.
"""
import json,os,tempfile,time
from pathlib import Path
from contract import identity,admit_release

def publish(path,payload):
 path=Path(path);raw=(json.dumps(payload,sort_keys=True)+'\n').encode();fd,tmp=tempfile.mkstemp(prefix='.barrier-',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  os.link(tmp,path) # exclusive: an earlier result can never be replaced
  d=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
  try:os.fsync(d)
  finally:os.close(d)
 finally:os.unlink(tmp)

def write_ready(directory,expected,namespace_pid,cuda_initialized):
 identity(expected)
 if namespace_pid!=1 or cuda_initialized is not False:raise ValueError('worker not at pre-CUDA container barrier')
 ready={'mode':expected['mode'],'nonce':expected['nonce'],'container_id':expected['container_id'],'namespace_pid':1,'cuda_initialized':False}
 publish(Path(directory)/'worker_ready.json',ready);return ready

def write_release(directory,expected,sampler_started,first_sample):
 directory=Path(directory);ready=json.loads((directory/'worker_ready.json').read_text());release=admit_release(expected,ready,sampler_started,first_sample)
 publish(directory/'release.json',release);return release

def wait_release(directory,expected,timeout_seconds,clock=time.monotonic,sleep=time.sleep):
 identity(expected)
 if not 0<timeout_seconds<=300:raise ValueError('invalid barrier timeout')
 path=Path(directory)/'release.json';deadline=clock()+timeout_seconds
 while clock()<deadline:
  try:release=json.loads(path.read_text())
  except FileNotFoundError:sleep(.1);continue
  if release.get('status')!='release' or any(release.get(k)!=expected[k] for k in ('mode','nonce','container_id')) or type(release.get('first_sample_ns')) is not int:raise ValueError('invalid barrier release')
  return release
 raise TimeoutError('memory barrier release missing')
