import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;gate=D.parent/'pixel_wavefront_t020_timing_final_v4/gpu_gate.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources():return {str(p.resolve()):sha(p) for p in [D/'test_gate_full.py',D/'cpu_check.py',gate]}
def main():
 before=sources();r=subprocess.run([sys.executable,'-m','unittest','-v','test_gate_full'],cwd=D,capture_output=True,text=True,timeout=30,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));after=sources();(D/'test001.log').write_text(r.stdout+r.stderr);v={'status':'passed' if r.returncode==0 and before==after else 'failed','source_sha256_before':before,'source_sha256_after':after,'GT_opened':False,'GPU_used':False,'predictions_opened':False}
 with (D/'proof001.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if v['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
