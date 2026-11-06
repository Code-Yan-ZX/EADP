"""Post-run immutable source snapshot; no model, GPU or GT access."""
import hashlib,json,os,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 started=read(HERE/'monitor_run001/started.json');terminal=read(HERE/'monitor_run001/terminal.json')
 if terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or started.get('worker_pid')!=terminal.get('worker_pid'):raise ValueError('GPU run did not exit cleanly')
 plan=read(HERE/'gpu_plan001.json');dest=HERE/'run001_sources';dest.mkdir();copies={}
 for i,(source,digest) in enumerate(sorted(plan['source_sha256'].items())):
  src=Path(source);target=dest/('%04d'%i)
  if sha(src)!=digest:raise ValueError('source drift before snapshot')
  shutil.copyfile(src,target)
  with target.open('rb') as copied:os.fsync(copied.fileno())
  if sha(target)!=digest or sha(src)!=digest:raise ValueError('source drift while copying')
  copies[source]={'sha256':digest,'copy':str(target.resolve())}
 with (dest/'manifest.json').open('x') as f:json.dump({'status':'passed','post_run_snapshot':True,'source_files':copies,'GT_opened':False,'limits':['Endpoint hashes and post-run copies cannot exclude transient source replacement']},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
