import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e6/pixel_wavefront_t020_timing_final_v4';A=B/'e6/pixel_wavefront_t020_timing_accept_final'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if (R/'run_001').exists() or (D/'plan001.json').exists():raise ValueError('must freeze before GPU timing')
 files=[D/n for n in ('observe.py','test_observe.py','freeze.py')];before={str(p):sha(p) for p in files};r=subprocess.run([sys.executable,'-m','unittest','-v','test_observe'],cwd=D,capture_output=True,text=True,timeout=30,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));after={str(p):sha(p) for p in files};(D/'cpu_test001.log').write_text(r.stdout+r.stderr)
 if r.returncode or before!=after:raise ValueError('CPU source/test')
 v={'scope':'supplementary low-frequency ps and Docker list observation','source_sha256':after,'gpu_plan_sha256':sha(R/'gpu_plan001.json'),'formal_plan_sha256':sha(A/'verification_plan001.json'),'interval_seconds':60,'duration_seconds':30000,'output':str(D/'observations001.jsonl'),'stop':str(D/'stop001'),'limitations':['not part of GPU gate; separate independent review required','query windows and first fsync bounded; second fsync duration unknown','point observations cannot rule out unobserved background or causally quantify it','observer queries themselves can disturb timing','no GPU query or model access'],'GT_opened':False,'GPU_used':False}
 with (D/'plan001.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
