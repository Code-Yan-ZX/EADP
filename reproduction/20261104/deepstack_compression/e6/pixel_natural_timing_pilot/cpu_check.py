"""Run frozen CPU contract and selector-wrapper tests; no GPU/labels."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import preflight

def main():
 plan=preflight.read(HERE/'plan001.json');preflight.verify_source_plan(plan);preflight.validate_upstream()
 run=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],capture_output=True,text=True,timeout=120)
 if run.returncode:raise ValueError('CPU tests failed: '+run.stderr[-4000:])
 preflight.verify_source_plan(plan)
 proof={'status':'passed','stage':'CPU only; no GPU authorization','plan_sha256':preflight.sha(HERE/'plan001.json'),
        'source_sha256':plan['sha256'],'test_stdout':run.stdout,'test_stderr':run.stderr,'GT_opened':False,
        'predictions_opened':False,'gpu_used':False,'notes':['no real model','selector K wrapper overhead included in measured candidate']}
 out=HERE/'cpu001.json'
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
