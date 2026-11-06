"""Save exact post-run source inventory without overwriting the frozen run."""
import hashlib,json,os,shutil
from pathlib import Path
D=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 plan=json.loads((D/'gpu_plan001.json').read_text());out=D/'source_snapshot001';out.mkdir(exist_ok=False);files={}
 for i,(path,digest) in enumerate(sorted(plan['source_sha256'].items())):
  if sha(path)!=digest:raise ValueError('source differs from frozen plan: '+path)
  dest=out/('%04d_%s'%(i,Path(path).name));shutil.copyfile(path,dest)
  if sha(dest)!=digest or sha(path)!=digest:raise ValueError('source changed during snapshot: '+path)
  files[path]={'path':str(dest),'sha256':digest}
 with (out/'manifest.json').open('x') as f:json.dump({'status':'passed','plan_sha256':sha(D/'gpu_plan001.json'),'files':files,'note':'post-run snapshot; endpoint hashes cannot exclude transient modification'},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
