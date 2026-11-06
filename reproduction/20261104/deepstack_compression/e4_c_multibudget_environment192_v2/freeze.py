"""Freeze a long-running observer; no observer or GPU work is started here."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 output=D/'plan001.json'
 if output.exists():raise ValueError('already frozen')
 paths=[D/'watch.py',D/'validate.py',D/'test_validate.py',D/'freeze.py',B/'initial_plan.json',B/'monitor/supervise.py',B/'e4_c_latency/timing_supervise.py']
 before={str(p.resolve()):sha(p) for p in paths}
 result=subprocess.run([sys.executable,'-m','unittest','-v','test_validate'],cwd=D,capture_output=True,text=True,timeout=30)
 if result.returncode or before!={str(p.resolve()):sha(p) for p in paths}:raise ValueError('CPU tests or source freeze failed: '+result.stderr)
 py='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python';control='/home/dell/miniconda3/bin/python'
 command=[control,str(B/'e4_c_latency/timing_supervise.py'),'--run-dir',str(D/'monitor_watch001'),'--timeout','36120','--snapshot-seconds','3600','--poll-seconds','5','--grace-seconds','30','--disk','/media/disk2','--',py,str(D/'watch.py'),'--output',str(D/'samples001.jsonl'),'--stop',str(D/'stop001'),'--interval','60','--duration','36000']
 plan={'status':'prepared','GPU_authorized':False,'GT_opened':False,'scoring_performed':False,'source_sha256':before,'command':command,'gpu_outer_timeout_seconds':28800,'minimum_cleanup_seconds':1200,'watcher_duration_seconds':36000,'watcher_outer_timeout_seconds':36120,'interval_seconds':60,'manual_path':str(D/'manual_checks001.jsonl'),'manual_contract':'root logs real hourly monotonic start/end, UTC start/end, time namespace and details; absent hours unknown; manual log fsync unbounded','stop_contract':'root writes stop marker only after GPU/worker supervisor complete; clean watcher end must report stop_marker','unknown':'watcher second fsync, manual log fsync, stop-marker write and unrecorded residual host activity','created_unix':time.time(),'tests':result.stdout+result.stderr}
 with output.open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('observer proposal frozen',sha(output))
if __name__=='__main__':main()
