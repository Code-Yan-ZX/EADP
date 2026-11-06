"""No-CUDA worker-side barrier prototype, no GPU model entry point.

The private shared directory and all identity fields must be bound by a future
independently reviewed Docker controller. A release file alone is not an
authorization proof and this module provides no GPU launch authority.
"""
import importlib.util,json,os,tempfile,time
from pathlib import Path
MEMORY=Path(__file__).resolve().parent

def contract():
 spec=importlib.util.spec_from_file_location('eadp_memory_contract',MEMORY/'contract.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def wait_json(path,deadline,clock=time.monotonic,sleep=time.sleep):
 while clock()<deadline:
  try:return json.loads(Path(path).read_text())
  except FileNotFoundError:sleep(.1)
 raise TimeoutError('worker barrier timed out: '+Path(path).name)

def handshake(directory,nonce,is_cuda_initialized,namespace_pid=os.getpid,clock=time.monotonic,sleep=time.sleep,timeout=300):
 if not 0<timeout<=300:raise ValueError('invalid worker barrier timeout')
 d=Path(directory);deadline=clock()+timeout;expected=wait_json(d/'expected.json',deadline,clock,sleep);contract().identity(expected)
 if expected['nonce']!=nonce or namespace_pid()!=1 or is_cuda_initialized() is not False:raise ValueError('worker not bound to pre-CUDA PID 1')
 ready={'mode':expected['mode'],'nonce':nonce,'container_id':expected['container_id'],'namespace_pid':1,'cuda_initialized':False}
 # Link after fsync publishes the complete JSON once, never a partial file.
 fd,tmp=tempfile.mkstemp(prefix='.nvml-worker-',dir=d)
 try:
  with os.fdopen(fd,'w') as f:json.dump(ready,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.link(tmp,d/'worker_ready.json')
  parent_fd=os.open(d,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
  try:os.fsync(parent_fd)
  finally:os.close(parent_fd)
 finally:os.unlink(tmp)
 release=wait_json(d/'release.json',deadline,clock,sleep)
 if release.get('status')!='release' or any(release.get(k)!=expected[k] for k in ('mode','nonce','container_id')) or type(release.get('first_sample_ns')) is not int or release['first_sample_ns']<=0 or is_cuda_initialized() is not False:raise ValueError('invalid pre-CUDA release')
 return expected,release
