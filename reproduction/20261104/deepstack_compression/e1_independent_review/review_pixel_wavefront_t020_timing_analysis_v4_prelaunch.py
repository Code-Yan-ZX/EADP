"""Independent pre-run analysis-source audit; never reads measured results."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/pixel_wavefront_t020_timing_analysis_v4';RUN=B/'e6/pixel_wavefront_t020_timing_final_v4';A=B/'e6/pixel_wavefront_t020_timing_accept_final';OUT=I/'pixel_wavefront_t020_timing_analysis_v4_prelaunch_review.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=OUT.resolve() or OUT.exists():raise ValueError('unplanned output')
 plan=read(D/'analysis_plan001.json');cpu=read(D/'cpu_proof001.json');source=plan['source_sha256']
 if len(source)!=7 or any(sha(p)!=v for p,v in source.items()) or cpu.get('status')!='passed' or cpu.get('source_sha256_before')!=source or cpu.get('source_sha256_after')!=source or plan.get('gpu_plan_sha256')!=sha(RUN/'gpu_plan001.json') or plan.get('formal_plan_sha256')!=sha(A/'verification_plan001.json') or plan.get('warmups')!=6 or plan.get('measured')!=18 or plan.get('GT_opened') is not False or plan.get('timing_results_opened') is not False:raise ValueError('analysis freeze')
 spec=importlib.util.spec_from_file_location('_independent_pixel_analysis_gate',D/'gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 g.terminal(D/'monitor_cpu001',D/'cpu_check.py');g.terminal(D/'monitor_freeze001',D/'freeze.py')
 test=subprocess.run([sys.executable,'-m','unittest','-v','test_analysis','test_gate'],cwd=D,capture_output=True,text=True,timeout=120)
 if test.returncode or 'Ran 4 tests' not in test.stderr or any(sha(p)!=v for p,v in source.items()):raise ValueError('analysis CPU tests or source changed')
 q={'status':'passed','independent_review':True,'timing_results_opened':False,'GT_opened':False,'source_sha256':source,'plan_sha256':sha(D/'analysis_plan001.json'),'cpu_sha256':sha(D/'cpu_proof001.json'),'cpu_started_sha256':sha(D/'monitor_cpu001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu001/terminal.json'),'gpu_plan_sha256':sha(RUN/'gpu_plan001.json'),'formal_plan_sha256':sha(A/'verification_plan001.json'),'reviewer_sha256':sha(__file__),'errors':[],'limitations':['Three posthoc anchors; same-repeat pairs are sequential independent processes.','No population inference; unobserved background remains unknown.']}
 with OUT.open('x') as f:json.dump(q,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
