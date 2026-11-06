"""Bind and re-run independent test helpers omitted from first proof."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;A=B/'e6/rres_eadp_paired_timing_assessment';old=I/'rres_eadp_timing_statistics_v2_prelaunch_review.json';out=I/'rres_eadp_timing_statistics_v2_supplement.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if out.exists():raise ValueError('existing output')
 paths=[A/'observation_oracle.py',A/'test_observation_windows.py',A/'test_analysis_v2_independent.py',I/'review_rres_eadp_timing_statistics_v2.py',old]
 before={str(p):sha(p) for p in paths};tests=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 for pat in ('test_observation_windows.py','test_analysis_v2_independent.py'):
  r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(A),'-p',pat],cwd=B,env=env,text=True,capture_output=True,timeout=120)
  if r.returncode:raise ValueError(r.stderr)
  tests.append(r.stderr)
 if before!={str(p):sha(p) for p in paths}:raise ValueError('source drift')
 with out.open('x') as f:json.dump({'status':'passed','scope':'bind independent synthetic test helpers to statistics prelaunch review','source_sha256':before,'reviewer_sha256':sha(__file__),'GT_opened':False,'timing_results_opened':False,'GPU_used':False,'errors':[],'tests':tests},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
