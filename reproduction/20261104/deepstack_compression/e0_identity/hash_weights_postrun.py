"""Read-only post-run payload SHA256 with before/after stat stability evidence."""
import hashlib,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
PLAN=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')
plan=json.loads(PLAN.read_text());ex=json.loads((HERE/'array16_exit.json').read_text())
if ex['returncode']!=0 or ex['timed_out'] or ex['gpu_pids_after']:raise RuntimeError('run not fully exited')
def stat(p):
 s=p.stat();return dict(resolved=str(p.resolve()),bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
records=[];begin=time.time()
for name,guard in plan['model_stat_guards'].items():
 p=Path(name);before=stat(p); h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 after=stat(p);digest=h.hexdigest()
 records.append(dict(path=name,sha256=digest,expected=guard['prior_verified_sha256'],stat_before=before,stat_after=after,match=(before==after==guard['current_full_stat'] and digest==guard['prior_verified_sha256'])))
 print(name,records[-1]['match'],flush=True)
out=dict(start_unix=begin,finished_unix=time.time(),all_match=all(x['match'] for x in records),records=records,scope='post-run payload hash only; no inference about instantaneous pre-launch payload hash')
with (HERE/'array16_weights_postrun.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
raise SystemExit(0 if out['all_match'] else 1)
