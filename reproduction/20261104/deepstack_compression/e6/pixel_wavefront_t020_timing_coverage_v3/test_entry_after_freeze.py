"""Exercise the frozen actual entry: missing formal proof must deny access."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1]
out=D/'coverage001.json';assert not out.exists()
cmd=[sys.executable,str(D/'coverage_entry.py'),'--out',str(out)]
r=subprocess.run(cmd,cwd=B,capture_output=True,text=True,timeout=60,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
expected='pixel_wavefront_t020_timing_accept_v5/monitor_verify001/started.json'
if r.returncode==0 or expected not in r.stderr or 'coverage source changed' in r.stderr or out.exists():raise ValueError('actual entry did not stop at missing formal proof: '+r.stderr[-1200:])
value={'status':'passed','actual_command':cmd,'exit_nonzero':True,'denial':'missing formal supervisor started.json','plan_sha256':hashlib.sha256((D/'coverage_plan001.json').read_bytes()).hexdigest(),'stdout':r.stdout,'stderr':r.stderr,'timing_results_opened':False,'GPU_used':False,'GT_opened':False}
with (D/'entry_smoke001.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
