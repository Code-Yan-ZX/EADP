"""Independent CPU-only review of the fixed three-anchor timing analysis."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/pixel_wavefront_t020_timing_analysis_v5';R=B/'e6/pixel_wavefront_t020_timing_final_v9';A=B/'e6/pixel_wavefront_t020_timing_accept_v5';OUT=I/'pixel_wavefront_t020_timing_analysis_v5_prelaunch_review.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or [(cwd/x).resolve() for x in c if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('CPU supervisor')
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);o=a.parse_args().out.resolve()
 if o!=OUT or o.exists():raise ValueError('output')
 terminal(D/'monitor_cpu001',D/'cpu_check.py');terminal(D/'monitor_freeze001',D/'freeze.py')
 plan=read(D/'analysis_plan001.json');cpu=read(D/'cpu_proof001.json');sources={str(p.resolve()):sha(p) for p in D.glob('*.py')}
 if plan.get('source_sha256')!=sources or cpu.get('status')!='passed' or cpu.get('source_sha256_before')!=sources or cpu.get('source_sha256_after')!=sources or cpu.get('GT_opened') is not False or cpu.get('timing_results_opened') is not False or plan.get('GT_opened') is not False or plan.get('timing_results_opened') is not False or plan.get('gpu_plan_sha256')!=sha(R/'gpu_plan001.json') or plan.get('formal_plan_sha256')!=sha(A/'verification_plan001.json') or (plan.get('warmups'),plan.get('measured'))!=(6,18):raise ValueError('frozen analysis plan/source')
 t=subprocess.run([sys.executable,'-m','unittest','-v','test_analysis','test_gate'],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
 if t.returncode or any(sha(k)!=v for k,v in sources.items()):raise ValueError('independent test/source failure '+t.stderr[-1500:])
 fields={'plan_sha256':D/'analysis_plan001.json','cpu_sha256':D/'cpu_proof001.json','cpu_started_sha256':D/'monitor_cpu001/started.json','cpu_terminal_sha256':D/'monitor_cpu001/terminal.json','reviewer_sha256':Path(__file__),'gpu_plan_sha256':R/'gpu_plan001.json','formal_plan_sha256':A/'verification_plan001.json'}
 proof={'status':'passed','independent_review':True,'timing_results_opened':False,'GT_opened':False,'source_sha256':sources,'independent_tests':t.stdout+t.stderr,'scope':'prelaunch analysis sources only; no GPU or timing result authorization','errors':[],**{k:sha(v) for k,v in fields.items()}}
 with o.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
