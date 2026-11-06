"""After complete clean run, archive all frozen source bytes before verification."""
import argparse,hashlib,importlib.util,json,os,shutil,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;V=B/'e4_multistream_expansion343_gpu'
GATE_SHA='3b076731b71aeba5ae80e4cf02a34ee9849bd4508b6cae845fb62c61a0e59d96'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args();out=args.out.resolve()
 if out!=D/'run001_sources' or out.exists():raise ValueError('snapshot path')
 path=D/'gpu_gate.py'
 if sha(path)!=GATE_SHA:raise ValueError('admission hash')
 spec=importlib.util.spec_from_file_location('_multistream352v2_snapshot_admission',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);guard=m.admit();approval=guard['full_approval'];m.terminal(D/'monitor_run001',D/'launch.py')
 run=V/'run001';x=m.read(run/'exit.json');w=m.read(run/'worker_complete.json');m.inspect_gpu(m.read(run/'launch.json'),x)
 if x.get('container_stopped_verified') is not True or x.get('container_exit_code')!=0 or x.get('gpu_pids_after')!=[] or w.get('status')!='completed' or w.get('images')!=343:raise ValueError('run not complete')
 out.mkdir();inventory={}
 for i,(source,digest) in enumerate(sorted(dict(approval['source_sha256'],**guard['guard_source_sha256']).items())):
  p=Path(source);target=out/('%04d'%i)
  if sha(p)!=digest:raise ValueError('source changed')
  shutil.copyfile(p,target)
  if sha(target)!=digest:raise ValueError('snapshot mismatch')
  inventory[source]={'sha256':digest,'snapshot':str(target)}
 obj={'status':'passed','sources':inventory,'gpu_review_sha256':approval['review_sha256'],'guard_review_sha256':guard['guard_review_sha256'],'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'GT_opened':False,'scoring_performed':False}
 with (out/'manifest.json').open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
