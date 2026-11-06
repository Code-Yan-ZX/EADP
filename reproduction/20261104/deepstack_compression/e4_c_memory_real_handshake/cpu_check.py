"""Freeze exact source hashes while executing only no-GPU tests."""
import argparse,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D));import gate
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();before=gate.verify_plan();run=subprocess.run([sys.executable,'-m','unittest','-v','test_handshake'],cwd=D,capture_output=True,text=True,timeout=180);print(run.stdout,end='');print(run.stderr,end='',file=sys.stderr)
if run.returncode:raise SystemExit(run.returncode)
if gate.verify_plan()!=before:raise ValueError('source changed during CPU checks')
with a.out.open('x') as f:json.dump({'status':'passed','plan_sha256':gate.sha(D/'plan001.json'),'tests':4,'GT_opened':False,'gpu_used':False,'model_loaded':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
