"""Run isolated synthetic tests including real CLI fail-closed access sentinel."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=json.loads((D/'score_plan001.json').read_text());before={k:sha(k) for k in p['source_sha256']}
 if before!=p['source_sha256']:raise ValueError('frozen source changed')
 logs={}
 for test in ('test_paired','test_score','test_gate_cli','test_main_synthetic'):
  v=subprocess.run([sys.executable,'-m','unittest','-v',test],cwd=D,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=120)
  path=D/(test+'_cpu001.log');path.write_text(v.stdout+v.stderr);logs[test]=sha(path)
  if v.returncode:raise ValueError('synthetic test failed: '+test)
 after={k:sha(k) for k in p['source_sha256']}
 if before!=after:raise ValueError('source drift')
 proof={'status':'passed','source_sha256_before':before,'source_sha256_after':after,'score_plan_sha256':sha(D/'score_plan001.json'),
        'test_logs':logs,'GPU_used':False,'target_predictions_opened':False,'GT_opened':False,'scoring_performed':False}
 with (D/'cpu_proof001.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
