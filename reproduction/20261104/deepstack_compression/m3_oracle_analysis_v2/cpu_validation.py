"""Bounded isolated v2 gate synthetic tests; no prediction or reference reads."""
import argparse,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import prelaunch_gate
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
 before={str(x.resolve()):prelaunch_gate.sha(x) for x in prelaunch_gate.sources()}
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([sys.executable,'-m','unittest','-v','test_prelaunch'],cwd=HERE,env=env,capture_output=True,text=True,timeout=120)
 (a.out/'stdout.log').write_text(r.stdout);(a.out/'stderr.log').write_text(r.stderr);after={str(x.resolve()):prelaunch_gate.sha(x) for x in prelaunch_gate.sources()};ok=r.returncode==0 and before==after
 result={'status':'passed' if ok else 'failed','returncode':r.returncode,'source_sha256_before':before,'source_sha256_after':after,'stdout_sha256':prelaunch_gate.sha(a.out/'stdout.log'),'stderr_sha256':prelaunch_gate.sha(a.out/'stderr.log'),'GT_opened':False,'predictions_parsed':False,'scope':'synthetic CPU only'}
 with (a.out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
