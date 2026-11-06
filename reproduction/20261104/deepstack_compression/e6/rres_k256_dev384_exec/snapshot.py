"""After actual clean generation exit, snapshot each frozen source by value."""
import hashlib,json,os,shutil
from pathlib import Path
E=Path(__file__).resolve().parent;B=E.parents[1];F=B/'e6/rres_k256_dev384'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def main():
 start=json.loads((F/'monitor_run001/started.json').read_text());end=json.loads((F/'monitor_run001/terminal.json').read_text());cmd=start['command'];cwd=Path(start['cwd'])
 if len(cmd)<2 or (cwd/cmd[1]).resolve()!=E/'launch.py' or start['worker_pid']!=end['worker_pid'] or end.get('reason')!='completed' or end.get('worker_returncode')!=0 or end.get('remaining_process_group')!=[]:raise ValueError('generation not cleanly exited')
 plan=json.loads((F/'gpu_plan001.json').read_text());out=F/'source_snapshot001';out.mkdir(exist_ok=False);files={}
 for i,(path,digest) in enumerate(sorted(plan['source_sha256'].items())):
  if sha(path)!=digest:raise ValueError('source differs from frozen plan')
  dest=out/('%04d_%s'%(i,Path(path).name));shutil.copyfile(path,dest)
  if sha(dest)!=digest or sha(path)!=digest:raise ValueError('source changed during snapshot')
  files[path]={'path':str(dest),'sha256':digest}
 with (out/'manifest.json').open('x') as f:json.dump({'status':'passed','plan_sha256':sha(F/'gpu_plan001.json'),'files':files,'note':'post-run snapshot; endpoint hashes cannot exclude transient modification'},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
