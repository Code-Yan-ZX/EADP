"""Immediately retain all frozen timing sources after actual clean exit."""
import hashlib,importlib.util,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;RUN=BASE/'e4_c_multibudget_worker192_v5'
spec=importlib.util.spec_from_file_location('_budget_ttft_snap_identity',RUN/'gpu_gate.py');identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
def main():
 identity.terminal(RUN/'monitor_run001',RUN/'launch.py')
 ex=json.loads((RUN/'run_001/exit.json').read_text());state=ex.get('last_inspect') or {}
 if ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or state.get('running') is not False or state.get('pid')!=0 or ex.get('gpu_pids_after')!=[]:raise ValueError('timing run not cleanly exited')
 plan=json.loads((RUN/'gpu_plan001.json').read_text());sources=plan['source_sha256']
 if not sources:raise ValueError('empty timing source inventory')
 out=RUN/'run_001_sources';out.mkdir(exist_ok=False);items={}
 for source_name,expected in sorted(sources.items()):
  source=Path(source_name);before=source.stat();dst=out/(hashlib.sha256(source_name.encode()).hexdigest()+source.suffix);h=hashlib.sha256()
  with source.open('rb') as inp,dst.open('xb') as copy:
   for block in iter(lambda:inp.read(1048576),b''):copy.write(block);h.update(block)
   copy.flush();os.fsync(copy.fileno())
  after=source.stat()
  if h.hexdigest()!=expected or [before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns]!=[after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns] or identity.sha(dst)!=expected:raise ValueError('source snapshot changed '+source_name)
  item={'source':source_name,'snapshot':str(dst),'sha256':expected,'bytes':after.st_size};items[source_name]=item
  with (out/'snapshot_journal.jsonl').open('a') as f:f.write(json.dumps(item)+'\n');f.flush();os.fsync(f.fileno())
 data={'status':'passed','scope':'post-run endpoint source snapshot; transient changes cannot be excluded','runtime_plan_sha256':identity.sha(RUN/'gpu_plan001.json'),'snapshot_script_sha256':identity.sha(__file__),'sources':items}
 with (out/'manifest.json').open('x') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
