"""Retain the run's exact frozen source inventory after real termination."""
import hashlib,importlib.util,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;RUN=BASE/'e4_c_multibudget_references_v2'
spec=importlib.util.spec_from_file_location('_budget_snapshot_gate',RUN/'gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
def main():
 gate.terminal(RUN/'monitor_run001/started.json',RUN/'monitor_run001/terminal.json')
 ex=gate.read(RUN/'run_001/exit.json');state=ex.get('last_inspect') or {}
 if ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or state.get('running') is not False or state.get('pid')!=0 or ex.get('gpu_pids_after')!=[]:raise ValueError('GPU run has not exited cleanly')
 plan=gate.read(RUN/'execution_plan001.json');sources=plan['source_sha256']
 if len(sources)<220:raise ValueError('unexpected source inventory')
 out=RUN/'run_001_sources';out.mkdir(exist_ok=False);entries={}
 for original,expected in sorted(sources.items()):
  source=Path(original);before=source.stat();name=hashlib.sha256(original.encode()).hexdigest()+source.suffix;dest=out/name;h=hashlib.sha256()
  with source.open('rb') as inp,dest.open('xb') as dst:
   for block in iter(lambda:inp.read(1048576),b''):dst.write(block);h.update(block)
   dst.flush();os.fsync(dst.fileno())
  after=source.stat()
  if h.hexdigest()!=expected or [before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns]!=[after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns] or gate.sha(dest)!=expected:raise ValueError('source changed during snapshot '+original)
  item={'source':original,'snapshot':str(dest),'sha256':expected,'bytes':after.st_size};entries[original]=item
  with (out/'snapshot_journal.jsonl').open('a') as f:f.write(json.dumps(item)+'\n');f.flush();os.fsync(f.fileno())
 data={'status':'passed','scope':'post-run frozen endpoint snapshot; cannot exclude earlier transient change','execution_plan_sha256':gate.sha(RUN/'execution_plan001.json'),
       'snapshot_script_sha256':gate.sha(__file__),'sources':entries}
 with (out/'manifest.json').open('x') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','files':len(entries),'manifest_sha256':gate.sha(out/'manifest.json')}),flush=True)
if __name__=='__main__':main()
