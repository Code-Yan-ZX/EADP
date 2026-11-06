"""After complete clean run, archive all frozen source bytes before verification."""
import argparse,hashlib,importlib.util,json,os,shutil,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
ADMISSION_SHA='1e6bb7d1ccee1e18340f7d25110509ae2637683097af4ae7c050e013bbd70ab6'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args();out=args.out.resolve()
 if out!=D/'run001_sources' or out.exists():raise ValueError('snapshot path')
 path=D/'admission.py'
 if sha(path)!=ADMISSION_SHA:raise ValueError('admission hash')
 spec=importlib.util.spec_from_file_location('_multistream352v2_snapshot_admission',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);approval=m.admit(False,host=True);m.terminal(D/'monitor_run001',D/'launch.py')
 run=D/'run001';x=m.read(run/'exit.json');w=m.read(run/'worker_complete.json')
 if x.get('container_stopped_verified') is not True or x.get('container_exit_code')!=0 or x.get('gpu_pids_after')!=[] or w.get('status')!='completed' or w.get('images')!=9:raise ValueError('run not complete')
 out.mkdir();inventory={}
 for i,(source,digest) in enumerate(sorted(approval['source_sha256'].items())):
  p=Path(source);target=out/('%04d'%i)
  if sha(p)!=digest:raise ValueError('source changed')
  shutil.copyfile(p,target)
  if sha(target)!=digest:raise ValueError('snapshot mismatch')
  inventory[source]={'sha256':digest,'snapshot':str(target)}
 obj={'status':'passed','sources':inventory,'gpu_review_sha256':approval['review_sha256'],'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'GT_opened':False,'scoring_performed':False}
 with (out/'manifest.json').open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
