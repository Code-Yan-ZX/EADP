"""Freeze GPU host guard, real v2 preflight bindings and formal sources."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;V=B/'e4_multistream_expansion352_gpu_v4';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(dir,script):
 s=read(dir/'started.json');t=read(dir/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or len(c)<2 or (cwd/c[1]).resolve()!=script.resolve():raise ValueError('supervision')
def main():
 out=D/'cpu_check001.json'
 if out.exists():raise ValueError('already frozen')
 terminal(D/'monitor_host_smoke002',D/'launch.py');terminal(V/'monitor_preflight001',V/'launch.py');terminal(I/'monitor_e4_multistream_expansion352_v4_preflight001_result',I/'review_e4_multistream_expansion352_v4_preflight001_result.py')
 smoke=read(D/'monitor_host_smoke002/stdout.log');review=read(I/'e4_multistream_expansion352_v4_preflight001_result_review.json');cpu=read(V/'cpu_check002.json')
 if smoke.get('status')!='passed' or smoke.get('GPU_used') is not False or smoke.get('model_loaded') is not False or review.get('status')!='passed' or review.get('independent_review') is not True or review.get('processor_inputs_checked')!=9 or review.get('GT_opened') is not False or review.get('errors')!=[] or cpu.get('status')!='passed' or any(sha(k)!=v for k,v in cpu['source_sha256_after'].items()):raise ValueError('v2 preflight/source')
 paths={p.resolve() for p in D.glob('*.py')}|{(D/'formal_plan001.json').resolve(),(V/'cpu_check002.json').resolve(),(V/'execution_plan001.json').resolve(),(V/'formal_plan001.json').resolve(),(V/'preflight001/worker_preflight.json').resolve(),(V/'preflight001/launch.json').resolve(),(V/'preflight001/container_identity.json').resolve(),(V/'preflight001/exit.json').resolve(),(V/'monitor_preflight001/started.json').resolve(),(V/'monitor_preflight001/terminal.json').resolve(),(I/'e4_multistream_expansion352_v4_preflight001_result_review.json').resolve(),(I/'review_e4_multistream_expansion352_v4_preflight001_result.py').resolve(),(I/'monitor_e4_multistream_expansion352_v4_preflight001_result/started.json').resolve(),(I/'monitor_e4_multistream_expansion352_v4_preflight001_result/terminal.json').resolve(),(D/'monitor_host_smoke002/started.json').resolve(),(D/'monitor_host_smoke002/terminal.json').resolve(),(D/'monitor_host_smoke002/stdout.log').resolve()}
 before={str(p):sha(p) for p in sorted(paths)};t=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py','-v'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=90);after={str(p):sha(p) for p in sorted(paths)}
 if t.returncode or before!=after:raise ValueError('tests or source drift '+t.stderr[-2000:])
 value={'status':'passed','source_sha256_before':before,'source_sha256_after':after,'tests':t.stdout+t.stderr,'GT_opened':False,'GPU_used':False,'model_loaded':False,'scope':'v3 CPU host guard and formal source freeze; real GPU still locked','errors':[]}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
