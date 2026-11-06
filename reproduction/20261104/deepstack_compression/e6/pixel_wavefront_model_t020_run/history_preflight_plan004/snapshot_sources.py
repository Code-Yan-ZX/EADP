"""Copy and rehash the frozen source set before model/GPU arm execution."""
import hashlib,json,os,shutil
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def snapshot(sources,directory):
 directory=Path(directory);directory.mkdir(exist_ok=False);items={}
 for name,expected in sorted(sources.items()):
  if not Path(name).is_absolute() or sha(name)!=expected:raise ValueError('changed source '+name)
  dest=directory/(hashlib.sha256(name.encode()).hexdigest()+'.bin')
  with Path(name).open('rb') as src,dest.open('xb') as out:shutil.copyfileobj(src,out,length=1048576);out.flush();os.fsync(out.fileno())
  if sha(dest)!=expected or sha(name)!=expected:raise ValueError('source changed while copying '+name)
  items[name]={'path':str(dest.resolve()),'sha256':expected}
 with (directory/'manifest.json').open('x') as f:json.dump({'status':'passed','sources':items},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return items
