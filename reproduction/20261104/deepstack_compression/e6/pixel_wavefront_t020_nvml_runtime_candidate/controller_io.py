"""Host identity and immutable barrier evidence for a future memory controller.

No CLI; does not grant admission, instantiate Docker, poll GPUs or load models.
"""
import hashlib,importlib.util,json,os,time
from pathlib import Path
MEMORY=Path(__file__).resolve().parent
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
contract=module('eadp_controller_memory_contract',MEMORY/'contract.py')
def publish(path,data):
 path=Path(path);raw=(json.dumps(data,sort_keys=True,separators=(',',':'))+'\n').encode()
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
 try:os.fsync(fd)
 finally:os.close(fd)
 return hashlib.sha256(raw).hexdigest()
def identity_from_inspect(inspect,mode,nonce,gpu_uuid,max_gap_ns,proc=Path('/proc')):
 cid=inspect.get('container_id');pid=inspect.get('pid')
 if inspect.get('running') is not True or type(pid) is not int or pid<=0:raise ValueError('container has no running host PID')
 root=proc/str(pid);stat=(root/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
 if len(tail)<20 or tail[0]=='Z' or cid not in (root/'cgroup').read_text():raise ValueError('host worker identity/cgroup mismatch')
 out={'mode':mode,'nonce':nonce,'container_id':cid,'gpu_uuid':gpu_uuid,'host_pid':pid,'start_ticks':int(tail[19]),'boot_id':(proc/'sys/kernel/random/boot_id').read_text().strip(),'max_sample_gap_ns':max_gap_ns}
 return contract.identity(out)
def publish_stopped(directory,expected,inspection,alive,clock=time.monotonic_ns):
 contract.identity(expected)
 if inspection.get('container_id')!=expected['container_id'] or inspection.get('running') is not False or inspection.get('pid')!=0 or inspection.get('exit_code')!=0 or alive(expected):raise ValueError('worker not independently confirmed stopped')
 d=Path(directory);raw=inspection.get('raw')
 if not isinstance(raw,dict) or raw.get('Id')!=expected['container_id'] or (raw.get('State') or {}).get('Running') is not False or (raw.get('State') or {}).get('Pid')!=0 or (raw.get('State') or {}).get('ExitCode')!=0:raise ValueError('raw Docker inspect mismatch')
 h=publish(d/'stopped_inspect.json',[raw]);end=clock()
 event={k:expected[k] for k in ('nonce','container_id','host_pid','start_ticks','boot_id')};event.update(status='stopped',monotonic_ns=end,inspect_sha256=h)
 publish(d/'stopped.json',event);return end
def host_alive(expected,proc=Path('/proc')):
 contract.identity(expected)
 try:
  if (proc/'sys/kernel/random/boot_id').read_text().strip()!=expected['boot_id']:return False
  root=proc/str(expected['host_pid']);stat=(root/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
  return tail[0]!='Z' and int(tail[19])==expected['start_ticks'] and expected['container_id'] in (root/'cgroup').read_text()
 except (FileNotFoundError,ProcessLookupError,IndexError,ValueError):return False
