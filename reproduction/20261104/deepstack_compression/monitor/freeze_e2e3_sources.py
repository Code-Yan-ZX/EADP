"""Copy and hash the full frozen E2/E3 run source inventory after clean exit."""
import hashlib,json,os,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent;RUN=BASE/'e2_e3_full/run_001';MON=BASE/'e2_e3_full/monitor_001';OUT=BASE/'e2_e3_full/run_001_sources'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 start,end=read(MON/'started.json'),read(MON/'terminal.json');exit=read(RUN/'exit.json')
 if not (end['reason']=='completed' and end['worker_returncode']==0 and end['remaining_process_group']==[] and start['worker_pid']==end['worker_pid'] and exit['returncode']==0 and exit['container_exit_code']==0 and exit['container_stopped_verified'] is True and exit['gpu_pids_after']==[]):raise ValueError('run did not complete cleanly')
 container=exit['container_identity']['container_id'];actual=json.loads(subprocess.check_output(['docker','inspect',container],text=True))[0]['State']
 if actual['Running'] or actual['Pid']!=0 or actual['ExitCode']!=0 or actual['OOMKilled']:raise ValueError('container still running or failed')
 frozen=read(RUN/'frozen.json');inventory=frozen['code_sha256'];OUT.mkdir(exist_ok=False);entries={}
 for src,digest in inventory.items():
  path=Path(src)
  if sha(path)!=digest:raise ValueError('source changed: '+src)
  dest=OUT/'files'/path.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True)
  with path.open('rb') as f,dest.open('xb') as o:
   for chunk in iter(lambda:f.read(1024*1024),b''):o.write(chunk)
   o.flush();os.fsync(o.fileno())
  if sha(dest)!=digest or sha(path)!=digest:raise ValueError('source changed while copying: '+src)
  entries[src]={'sha256':digest,'snapshot':str(dest)}
 manifest={'status':'passed','sources':entries,'count':len(entries),'frozen_sha256':sha(RUN/'frozen.json'),
           'run_started_sha256':sha(MON/'started.json'),'run_terminal_sha256':sha(MON/'terminal.json'),
           'exit_sha256':sha(RUN/'exit.json'),'container_id':container,'GT_opened':False,'scoring_performed':False}
 with (OUT/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','sources':len(entries)}))
if __name__=='__main__':main()
