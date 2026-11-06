"""Save source identities after the run; endpoint snapshots cannot rule out transient changes."""
import hashlib,json,os,shutil
from pathlib import Path
import gate
D=Path(__file__).resolve().parent
def main():
 p=gate.plan();out=D/'run_001_sources';out.mkdir(exist_ok=False);items={}
 for i,(name,expected) in enumerate(sorted(p['source_sha256'].items())):
  src=Path(name);before=src.stat();dst=out/('%04d'%i);shutil.copyfile(src,dst);after=src.stat()
  if gate.sha(src)!=expected or gate.sha(dst)!=expected or (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('changed source '+name)
  items[name]={'snapshot':str(dst),'sha256':expected}
  with (out/'journal.jsonl').open('a') as f:json.dump({'source':name,'sha256':expected},f);f.write('\n');f.flush();os.fsync(f.fileno())
 with (out/'manifest.json').open('x') as f:json.dump({'status':'passed','runtime_plan_sha256':gate.sha(D/'runtime_plan001.json'),'sources':items,'scope':'post-run endpoint snapshot; transient changes cannot be excluded'},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
