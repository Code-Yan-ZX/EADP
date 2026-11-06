"""After clean runtime, archive source inventory and independent acceptance scripts."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;B=ROOT/'m3_oracle_batch2';V=ROOT/'m3_oracle_batch2_verify'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==B/'runtime_001_sources'
 end=json.loads((B/'monitor_runtime001/terminal.json').read_text());assert end['reason']=='completed' and end['worker_returncode']==0 and end['remaining_process_group']==[]
 cpu=Path(json.loads((B/'runtime_approval.json').read_text())['cpu_proof']);c=json.loads(cpu.read_text());src=c['source_sha256_after'];assert c['status']=='passed' and c['source_sha256_before']==src and all(sha(p)==d for p,d in src.items())
 ex=json.loads((B/'runtime_001/exit.json').read_text());assert ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['errors']==[] and ex['gpu_pids_after']==[]
 a.out.mkdir(exist_ok=False)
 for source,digest in src.items():
  dest=a.out/'files'/source.lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);assert sha(dest)==digest
 audit_cpu=V/'cpu_check_003/verification.json';audit_started=V/'monitor_cpu003/started.json';audit_terminal=V/'monitor_cpu003/terminal.json';audit=json.loads(audit_cpu.read_text());started=json.loads(audit_started.read_text());terminal=json.loads(audit_terminal.read_text());assert audit['status']=='passed' and audit['source_sha256_before']==audit['source_sha256_after'] and terminal['reason']=='completed' and terminal['worker_returncode']==0 and terminal['remaining_process_group']==[] and started['worker_pid']==started['process_group']==terminal['worker_pid']
 cmd=started['command'];cwd=Path(started['cwd']);assert [(cwd/x).resolve() for x in cmd if Path(x).name=='cpu_validation.py']==[V/'cpu_validation.py'] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==audit_cpu.parent
 extra=audit['source_sha256_after'];assert all(sha(p)==digest for p,digest in extra.items())
 for source,digest in extra.items():
  dest=a.out/'files'/source.lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);assert sha(dest)==digest
 m={'status':'passed','source_sha256':src,'acceptance_source_sha256':extra,'cpu_proof_sha256':sha(cpu),'acceptance_cpu_proof_sha256':sha(audit_cpu),'acceptance_cpu_started_sha256':sha(audit_started),'acceptance_cpu_terminal_sha256':sha(audit_terminal),'run_terminal_sha256':sha(B/'monitor_runtime001/terminal.json'),'container_exit_sha256':sha(B/'runtime_001/exit.json')}
 with (a.out/'manifest.json').open('x') as f:json.dump(m,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','source_count':len(src)}))
if __name__=='__main__':main()
