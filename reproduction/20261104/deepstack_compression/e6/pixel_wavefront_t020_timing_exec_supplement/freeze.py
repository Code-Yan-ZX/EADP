"""Fix post-run supervision commands without modifying pre-frozen formal plan."""
import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e6/pixel_wavefront_t020_timing_final_v4';A=B/'e6/pixel_wavefront_t020_timing_accept_final';PY='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if (R/'run_001').exists() or (D/'plan001.json').exists():raise ValueError('must precede run')
 gp=R/'gpu_plan001.json';fp=A/'verification_plan001.json';formal=json.loads(fp.read_text())
 if formal['gpu_plan_sha256']!=sha(gp):raise ValueError('formal/GPU plan mismatch')
 def command(directory,timeout,script,out):return [PY,str(B/'monitor/supervise.py'),'--run-dir',str(directory),'--timeout',str(timeout),'--snapshot-seconds','3600','--poll-seconds','5','--grace-seconds','30','--',PY,str(script),'--out',str(out)]
 v={'scope':'pre-frozen post-run snapshot and formal verifier exact supervision commands','gpu_plan_sha256':sha(gp),'formal_plan_sha256':sha(fp),'freeze_source_sha256':sha(D/'freeze.py'),'snapshot_source_sha256':sha(A/'snapshot_entry.py'),'verifier_source_sha256':sha(A/'verify_entry.py'),'cwd':str(B),'snapshot_command':command(A/'monitor_snapshot001',1800,A/'snapshot_entry.py',R/'run_001_sources'),'verify_command':command(A/'monitor_verify001',1800,A/'verify_entry.py',A/'verification001.json'),'automatic_retry':False,'GPU_used':False,'GT_opened':False,'predictions_opened':False,'limitations':['independent review must bind this supplement; it is not directly enforced by the GPU gate','formal and independent result review are required before interpreting timings']}
 with (D/'plan001.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
