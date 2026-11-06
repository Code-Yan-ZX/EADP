"""After clean M3 runtime exit, snapshot reviewed sources without reading results."""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;EXEC=ROOT/'m3_oracle_exec'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for data in iter(lambda:f.read(1048576),b''):h.update(data)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();run=EXEC/'runtime_001'
 started=read(EXEC/'monitor_runtime001/started.json');terminal=read(EXEC/'monitor_runtime001/terminal.json')
 launch=read(run/'launch.json');ex=read(run/'exit.json');complete=read(run/'panel/worker_complete.json')
 if (terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or
     started.get('worker_pid')!=terminal.get('worker_pid') or launch.get('controller',{}).get('pid')!=terminal.get('worker_pid') or
     ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or
     ex.get('container_stopped_verified') is not True or ex.get('client_exited_while_container_running') is not False or ex.get('errors')!=[] or
     complete.get('status')!='completed' or complete.get('questions')!=128 or complete.get('new_generations')!=768):
  raise ValueError('M3 actual completed execution not confirmed')
 cid=ex['container_identity']['container_id'];q=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=20,check=True)
 v=json.loads(q.stdout)[0];state=v['State']
 if v.get('Id')!=cid or state.get('Running') is not False or state.get('Pid')!=0 or state.get('ExitCode')!=0:raise ValueError('actual M3 container not confirmed stopped')
 approval=read(EXEC/'runtime_approval.json');cpu=read(approval['cpu_proof']);src=cpu['source_sha256_after']
 if cpu.get('status')!='passed' or cpu.get('source_sha256_before')!=src or any(sha(p)!=d for p,d in src.items()):raise ValueError('reviewed source changed')
 out=a.out.resolve();out.mkdir(exist_ok=False)
 for name,digest in src.items():
  path=Path(name);dest=out/'files'/path.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
  if sha(dest)!=digest or sha(path)!=digest:raise ValueError('source changed during snapshot: '+name)
 proof={'status':'passed','source_sha256':src,'cpu_proof_sha256':sha(approval['cpu_proof']),
        'run_terminal_sha256':sha(EXEC/'monitor_runtime001/terminal.json'),'container_exit_sha256':sha(run/'exit.json'),
        'worker_complete_sha256':sha(run/'panel/worker_complete.json'),'reference_sources_opened':False,'prediction_files_opened':False}
 with (out/'manifest.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
