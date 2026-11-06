"""Freeze analysis source and synthetic CPU contract before any target run."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 proof=D/'cpu_proof001.json';plan=D/'analysis_code_plan001.json'
 if proof.exists() or plan.exists():raise ValueError('analysis freeze exists')
 files=[D/'analysis_gate.py',D/'report.py',D/'test_exec.py',D/'freeze.py',B/'e4_c_multibudget_analysis192_v2/metrics.py',B/'e4_c_multibudget_analysis192_v2/test_metrics.py',B/'e4_c_multibudget_analysis192_independent/window_detail.py',B/'e4_c_multibudget_analysis192_independent/test_window_detail.py',B/'e4_c_multibudget_environment192_v2/validate.py',B/'e4_c_multibudget_environment192_v2/plan001.json',B/'e4_c_multibudget_expansion192/ANALYSIS_PLAN.md',B/'e4_c_multibudget_expansion192/proposal002.json',B/'e4_c_multibudget_worker192_v6/cpu_plan001.json',B/'e4_c_multibudget_worker192_v6_accept/verification_plan001.json']
 before={str(p.resolve()):sha(p) for p in files};env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 run=subprocess.run([sys.executable,'-m','unittest','-v','test_exec'],cwd=D,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120,env=env)
 after={str(p.resolve()):sha(p) for p in files}
 if before!=after or run.returncode:raise ValueError('analysis synthetic test/source failed: '+run.stdout[-2000:])
 cpu={'status':'passed','returncode':run.returncode,'source_sha256_before':before,'source_sha256_after':after,'tests':run.stdout,'GT_opened':False,'target_predictions_opened':False,'timing_results_opened':False}
 with proof.open('x') as f:json.dump(cpu,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 result={'status':'frozen','source_sha256':dict(after,**{str(proof.resolve()):sha(proof)}),'GT_opened':False,'target_predictions_opened':False,'timing_results_opened':False,'numpy_version':'2.2.6','jobs':7920,'warmups':240,'measured':7680,'scope':'source preparation only; no GPU authority or timing data access'}
 with plan.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('analysis plan SHA256',sha(plan))
if __name__=='__main__':main()
