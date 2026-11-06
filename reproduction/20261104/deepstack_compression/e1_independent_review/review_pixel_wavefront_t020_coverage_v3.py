"""Independent review after run, before analysis: supplemental coverage only."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/pixel_wavefront_t020_timing_coverage_v3';O=B/'e6/pixel_wavefront_t020_timing_observer_v2';R=B/'e6/pixel_wavefront_t020_timing_final_v9';A=B/'e6/pixel_wavefront_t020_timing_accept_v5'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(p,script):
 s=read(p/'started.json');t=read(p/'terminal.json');cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in s['command'] if Path(x).name==script.name]==[script.resolve()]
 return {'started_sha256':sha(p/'started.json'),'terminal_sha256':sha(p/'terminal.json')}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);a=ap.parse_args();out=I/'pixel_wavefront_t020_coverage_v3_review.json';assert a.out.resolve()==out and not out.exists() and not (D/'coverage001.json').exists()
 p=read(D/'coverage_plan001.json');sources=p['source_sha256'];assert all(sha(k)==v for k,v in sources.items()) and p['observer_plan_sha256']==sha(O/'plan001.json') and p['gpu_plan_sha256']==sha(R/'gpu_plan001.json') and p['formal_plan_sha256']==sha(A/'verification_plan001.json') and p['timing_results_opened'] is False and p['GT_opened'] is False
 source=(D/'coverage.py').read_text();entry=(D/'coverage_entry.py').read_text();assert sha(D/'coverage.py') in entry and source.index('terminal(A/')<source.index("records=[read(") and "no samples removed" in source and "'unknown'" in source
 frozen=terminal(D/'monitor_freeze001',D/'coverage_freeze.py');smoke=terminal(D/'monitor_entry_smoke001',D/'test_entry_after_freeze.py');v=read(D/'entry_smoke001.json');assert v['status']=='passed' and v['exit_nonzero'] and v['plan_sha256']==sha(D/'coverage_plan001.json') and 'missing formal supervisor' in v['denial'] and v['timing_results_opened'] is False
 z=subprocess.run([sys.executable,'-m','unittest','-v','test_coverage'],cwd=D,capture_output=True,text=True,timeout=30);assert z.returncode==0 and 'OK' in z.stderr
 result={'status':'passed','independent_review':True,'scope':'post-run source review of pre-frozen supplemental coverage; no timing or output data opened','GT_opened':False,'timing_results_opened':False,'GPU_used':False,'coverage_plan_sha256':sha(D/'coverage_plan001.json'),'source_sha256':sources,'reviewer_sha256':sha(__file__),'supervision':{'freeze':frozen,'entry_smoke':smoke},'test_log':z.stderr,'limitations':['coverage is supplemental, not proof of no interference','second fsync, watcher supervisor health and unsampled background remain unknown','all samples retained'],'errors':[]}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
