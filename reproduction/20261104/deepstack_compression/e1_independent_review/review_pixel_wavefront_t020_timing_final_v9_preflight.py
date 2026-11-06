"""Independent CPU-only review of v9 preflight sources and host entry smokes."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/pixel_wavefront_t020_timing_final_v9';OUT=I/'pixel_wavefront_t020_timing_final_v9_preflight_review.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,script,flag=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (flag and cmd.count(flag)!=1):raise ValueError('bad completed supervisor')
 return s

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args()
 if args.out.resolve()!=OUT:raise ValueError('wrong proof path')
 terminal(D/'monitor_cpu001',D/'cpu_check001.py');terminal(D/'monitor_freeze001',D/'freeze_preflight.py')
 p=read(D/'preflight_plan001.json');cpu=read(D/'cpu_proof001.json')
 if cpu.get('status')!='passed' or cpu.get('GPU_used') is not False or cpu.get('GPU_authorized') is not False or cpu.get('GT_opened') is not False or cpu.get('predictions_opened') is not True or cpu['source_sha256_before']!=cpu['source_sha256_after'] or any(sha(k)!=v for k,v in cpu['source_sha256_after'].items()):raise ValueError('CPU source proof')
 for m,script in (('monitor_host_smoke001','launch.py'),('monitor_host_smoke002','preflight_launch.py')):
  s=terminal(D/m,D/script,'--no-gpu-dependency-smoke')
  if Path(s['cwd']).resolve()!=B or '--no-gpu-dependency-smoke' not in s['command']:raise ValueError('host smoke cwd/command')
  o=read(D/m/'stdout.log')
  if o.get('status')!='passed' or o.get('model_loaded') is not False or o.get('gpu_queried',o.get('GPU_queried')) is not False:raise ValueError('host smoke actual result')
 o=read(D/'monitor_host_smoke001/stdout.log');x=read(D/'monitor_host_smoke002/stdout.log')
 if len(o.get('paths',[]))!=7 or Path(o['paths'][-1]).name!='run_panel.py' or Path(x['module']).name!='run_panel.py' or Path(x['diagnose']).name!='diagnose.py':raise ValueError('host import path missing')
 gate_path=D/'preflight_gate.py';expected='6ecdb5f697153bf4e1c0764988f4e546025ddef0b319efb5ab3574f0b6286db1'
 if sha(gate_path)!=expected:raise ValueError('gate source mismatch before import')
 spec=importlib.util.spec_from_file_location('_pixel_final_v9_independent_gate',gate_path);gate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=gate;spec.loader.exec_module(gate)
 gate.cpu();prior,review=gate.prior();actual=gate.sources()
 if p.get('source_sha256')!=actual or any(sha(k)!=v for k,v in actual.items()):raise ValueError('complete frozen source mismatch')
 tests=subprocess.run([sys.executable,'-m','unittest','-v','test_dependencies','test_final_main','test_locks','test_preflight_gate','test_raw_prefix','test_weights'],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
 if tests.returncode or any(sha(k)!=v for k,v in actual.items()):raise ValueError('independent tests or source freeze '+tests.stderr[-1200:])
 plan=D/'preflight_plan001.json';fields={'plan_sha256':plan,'cpu_sha256':D/'cpu_proof001.json','cpu_started_sha256':D/'monitor_cpu001/started.json','cpu_terminal_sha256':D/'monitor_cpu001/terminal.json','prior_review_sha256':I/'pixel_wavefront_t020_timing_run_v2_preflight_result_review.json','reviewer_sha256':Path(__file__)}
 result={'status':'passed','independent_review':True,'preflight_only':True,'gpu_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'prediction_scope':'previously accepted mixed DEV files opened only for source hashes; no new target predictions or labels','source_sha256':actual,'independent_tests':tests.stdout+tests.stderr,'host_smoke_started_sha256':[sha(D/m/'started.json') for m in ('monitor_host_smoke001','monitor_host_smoke002')],'host_smoke_terminal_sha256':[sha(D/m/'terminal.json') for m in ('monitor_host_smoke001','monitor_host_smoke002')],'scope':'only real no-GPU container preflight; no GPU admission','errors':[],**{k:sha(v) for k,v in fields.items()}}
 with args.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
