import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e6/pixel_wavefront_t020_timing_final_v9';A=B/'e6/pixel_wavefront_t020_timing_accept_v5'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if (R/'run_001').exists() or (D/'coverage_plan001.json').exists():raise ValueError('freeze must precede GPU')
 files=[D/n for n in ('coverage.py','coverage_entry.py','test_coverage.py','coverage_freeze.py')];before={str(p.resolve()):sha(p) for p in files};r=subprocess.run([sys.executable,'-m','unittest','-v','test_coverage'],cwd=D,capture_output=True,text=True,timeout=30,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));after={str(p.resolve()):sha(p) for p in files};(D/'coverage_cpu001.log').write_text(r.stdout+r.stderr)
 if r.returncode or before!=after:raise ValueError('coverage CPU test/source')
 v={'scope':'supplemental observer time-window matching only after formal and independent result exits','source_sha256':after,'observer_plan_sha256':sha(D/'plan001.json'),'gpu_plan_sha256':sha(R/'gpu_plan001.json'),'formal_plan_sha256':sha(A/'verification_plan001.json'),'limitations':['not a prerequisite of the primary statistics','no selection or deletion','unobserved intervals and second fsync remain unknown'],'GPU_used':False,'GT_opened':False,'timing_results_opened':False}
 with (D/'coverage_plan001.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
