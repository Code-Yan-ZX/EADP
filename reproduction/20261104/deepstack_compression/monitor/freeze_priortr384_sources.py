"""Snapshot PriorTR DEV384 frozen source inventory after clean exit."""
import hashlib,json,os,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent;D=BASE/'e6/priortr/dev384';RUN=D/'run_001';MON=D/'monitor_001';OUT=D/'run_001_sources'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 s,t=read(MON/'started.json'),read(MON/'terminal.json');e=read(RUN/'exit.json')
 if not(t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid'] and e['returncode']==0 and e['container_exit_code']==0 and e['container_stopped_verified'] is True and e['gpu_pids_after']==[]):raise ValueError('unclean run')
 cid=e['container_identity']['container_id'];obj=json.loads(subprocess.check_output(['docker','inspect',cid],timeout=20))[0]
 if obj['State']['Running'] or obj['State']['Pid']!=0 or obj['State']['ExitCode']!=0 or obj['State']['OOMKilled']:raise ValueError('container not cleanly stopped')
 done=read(RUN/'worker_complete.json')
 if done!={'questions':384,'natural_generations':1152,'GT_opened':False,'scoring_performed':False} or read(RUN/'data/worker_complete.json')!=done:raise ValueError('full generation incomplete')
 frozen=read(RUN/'frozen.json');OUT.mkdir(exist_ok=False);entries={}
 for path,digest in frozen['source_sha256'].items():
  p=Path(path)
  if sha(p)!=digest:raise ValueError('source changed')
  dest=OUT/'files'/p.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True)
  with p.open('rb') as f,dest.open('xb') as o:
   for chunk in iter(lambda:f.read(1024*1024),b''):o.write(chunk)
   o.flush();os.fsync(o.fileno())
  if sha(dest)!=digest or sha(p)!=digest:raise ValueError('source changed while copied')
  entries[path]={'sha256':digest,'snapshot':str(dest)}
 result={'status':'passed','sources':entries,'count':len(entries),'frozen_sha256':sha(RUN/'frozen.json'),'run_started_sha256':sha(MON/'started.json'),'run_terminal_sha256':sha(MON/'terminal.json'),'exit_sha256':sha(RUN/'exit.json'),'container_id':cid,'GT_opened':False,'scoring_performed':False}
 with (OUT/'manifest.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','sources':len(entries)}))
if __name__=='__main__':main()
