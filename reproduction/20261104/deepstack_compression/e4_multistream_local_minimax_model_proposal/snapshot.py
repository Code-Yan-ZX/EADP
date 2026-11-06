"""Candidate endpoint source snapshot; not a proof of no transient changes."""
import hashlib,importlib.util,json,os,shutil,sys
from pathlib import Path
D=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 p=D/'admission.py'
 if sha(p)!='eabc64818fefeefd46563bce12a547b093b7165ca3fa2819c1fb576a2804aef4':raise ValueError('gate source')
 spec=importlib.util.spec_from_file_location('_minimax32_snapshot_gate',p);gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate);gate.admit(False)
 plan=gate.read(D/'execution_plan001.json');start=gate.read(D/'monitor_run001/started.json');terminal=gate.read(D/'monitor_run001/terminal.json');exit=gate.read(D/'run_001/exit.json')
 if (start.get('worker_pid')!=terminal.get('worker_pid') or terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or exit.get('returncode')!=0 or exit.get('container_exit_code')!=0 or exit.get('container_stopped_verified') is not True or exit.get('gpu_pids_after')!=[]):raise ValueError('run must really finish/clean before snapshot')
 out=D/'run_001_sources';out.mkdir(exist_ok=False);items={}
 for i,(name,expected) in enumerate(sorted(plan['source_sha256'].items())):
  src=Path(name);before=src.stat();dst=out/('%04d'%i);shutil.copyfile(src,dst);after=src.stat()
  if sha(src)!=expected or sha(dst)!=expected or (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('source changed during snapshot')
  items[name]={'snapshot':str(dst),'sha256':expected}
  with (out/'journal.jsonl').open('a') as f:json.dump({'source':name,'sha256':expected},f);f.write('\n');f.flush();os.fsync(f.fileno())
 with (out/'manifest.json').open('x') as f:json.dump({'status':'passed','execution_plan_sha256':sha(D/'execution_plan001.json'),'sources':items,'scope':'post-run endpoint snapshot, not proof against transient changes'},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
