"""After GPU exit only: snapshot and hash all frozen GPU source files."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];RUN=B/'e6/pixel_wavefront_t020_timing_final_v4'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args();out=args.out.resolve()
 if out!=(RUN/'run_001_sources').resolve():raise ValueError('snapshot path')
 start=json.loads((RUN/'monitor_run001/started.json').read_text());term=json.loads((RUN/'monitor_run001/terminal.json').read_text());plan=json.loads((RUN/'gpu_plan001.json').read_text())
 if start.get('command')!=plan['run_supervisor_command'] or Path(start.get('cwd','/')).resolve()!=Path(plan['run_supervisor_cwd']).resolve() or term.get('reason')!='completed' or term.get('worker_returncode')!=0 or term.get('remaining_process_group')!=[] or start.get('worker_pid')!=term.get('worker_pid'):raise ValueError('GPU run not cleanly exited')
 out.mkdir(exist_ok=False);items={}
 for i,(path,expected) in enumerate(sorted(plan['source_sha256'].items())):
  source=Path(path);dest=out/('%04d.bin'%i)
  if sha(source)!=expected:raise ValueError('source changed before snapshot')
  with source.open('rb') as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
  if sha(dest)!=expected:raise ValueError('source copy mismatch')
  items[path]={'sha256':expected,'copy':str(dest)}
 with (out/'manifest.json').open('x') as f:json.dump({'status':'passed','gpu_plan_sha256':sha(RUN/'gpu_plan001.json'),'sources':items},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
