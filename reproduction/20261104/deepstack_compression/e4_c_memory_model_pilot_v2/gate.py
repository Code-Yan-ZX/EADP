"""Fail-closed two-stage independent admission; no model or GPU side effects."""
import hashlib,json,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;R=B/'e1_independent_review';sys.path.insert(0,str(B/'e4_c_latency'))
_imports=sys.modules.get('imports')
if _imports is not None and (not getattr(_imports,'__file__',None) or Path(_imports.__file__).resolve()!=D/'imports.py'):raise ValueError('foreign imports module')
from imports import checked_import,check_existing,latency_bindings
identity_gate=checked_import('identity_gate',B/'e4_c_latency/identity_gate.py');supervised=identity_gate.supervised

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())

def prior():
 latency_gate=checked_import('latency_gate',B/'e4_c_latency/latency_gate.py')
 light=latency_gate.require_light_identity();proof=R/'e4c_memory_real_handshake_v2_result_review.json';reviewer=R/'review_e4c_memory_real_handshake_v2_result.py';s=R/'monitor_e4c_memory_real_handshake_v2_result/started.json';t=R/'monitor_e4c_memory_real_handshake_v2_result/terminal.json';supervised(s,t,reviewer,proof);r=read(proof)
 started=read(B/'e4_c_memory_real_handshake_v2/monitor_run001/started.json');ended=read(B/'e4_c_memory_real_handshake_v2/monitor_run001/terminal.json');cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
 if ended.get('reason')!='completed' or ended.get('worker_returncode')!=0 or ended.get('remaining_process_group')!=[] or ended.get('worker_pid')!=started.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name=='controller.py']!=[(B/'e4_c_memory_real_handshake_v2/controller.py').resolve()] or cmd[-2:]!=['--run-id','001']:raise ValueError('prior NVML run supervision mismatch')
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('model_loaded') is not False or r.get('GT_opened') is not False or r.get('reviewer_sha256')!=sha(reviewer) or r.get('errors')!=[] or r.get('run_started_sha256')!=sha(B/'e4_c_memory_real_handshake_v2/monitor_run001/started.json') or r.get('run_terminal_sha256')!=sha(B/'e4_c_memory_real_handshake_v2/monitor_run001/terminal.json'):raise ValueError('accepted real NVML handshake proof changed')
 v1=B/'e4_c_memory_model_pilot';formal=B/'e4_c_memory_model_accept/verification001.json';fs=B/'e4_c_memory_model_accept/monitor_verify001/started.json';ft=B/'e4_c_memory_model_accept/monitor_verify001/terminal.json';v1review=R/'e4c_model_memory001_review.json';vr=R/'review_e4c_model_memory_result.py';vs=R/'monitor_e4c_model_memory001/started.json';vt=R/'monitor_e4c_model_memory001/terminal.json'
 supervised(fs,ft,B/'e4_c_memory_model_accept/verify.py',formal);supervised(vs,vt,vr,v1review);f=read(formal);z=read(v1review)
 if f.get('status')!='passed' or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('verifier_sha256')!=sha(B/'e4_c_memory_model_accept/verify.py') or z.get('status')!='passed' or z.get('independent_review') is not True or z.get('GT_opened') is not False or z.get('scoring_performed') is not False or z.get('formal_sha256')!=sha(formal) or z.get('formal_started_sha256')!=sha(fs) or z.get('formal_terminal_sha256')!=sha(ft) or z.get('run_started_sha256')!=sha(v1/'monitor_run001/started.json') or z.get('run_terminal_sha256')!=sha(v1/'monitor_run001/terminal.json') or z.get('reviewer_sha256')!=sha(vr):raise ValueError('v1 model-memory proof mismatch')
 return {'light_identity':light,'nvml_handshake_review_sha256':sha(proof),'nvml_handshake_terminal_sha256':sha(t),'v1_formal_sha256':sha(formal),'v1_review_sha256':sha(v1review),'v1_review_terminal_sha256':sha(vt)}

def source_hashes():
 p=read(D/'plan001.json');sources=p.get('source_sha256')
 if not isinstance(sources,dict) or not sources:raise ValueError('missing sources')
 actual={path:sha(path) for path in sources}
 if actual!=sources or any(actual.get(str(x.resolve()))!=sha(x) for x in [*D.glob('*.py'),D/'PLAN.md']):raise ValueError('source changed or unbound')
 return actual

def verify_plan():
 p=read(D/'plan001.json');bounds={'sample_interval_ns':500000000,'max_sample_gap_ns':5000000000,'worker_timeout_seconds':1200,'ready_timeout_seconds':300,'sampler_timeout_seconds':1500,'inner_timeout_seconds':1560,'outer_timeout_seconds':1680,'grace_seconds':90}
 if p.get('scope')!='one DEV EADP K256 C-ON real model process memory; no GT or timing' or any(p.get(k)!=v for k,v in bounds.items()) or p.get('budget_source_sha256')!=sha(B/'initial_plan.json'):raise ValueError('frozen memory plan mismatch')
 source_hashes();return p

def review(*,preflight_only,host):
 plan=verify_plan();suffix='preflight' if preflight_only else 'final';proof=D/('independent_'+suffix+'_review.json');reviewer=R/('review_e4c_model_memory_v2_'+suffix+'.py');monitor=R/('monitor_e4c_model_memory_v2_'+suffix);cpu=D/'cpu_check005/verification.json';cs=D/'monitor_cpu005/started.json';ct=D/'monitor_cpu005/terminal.json'
 if not proof.exists():raise ValueError('memory pilot locked pending independent '+suffix+' review')
 supervised(cs,ct,D/'cpu_validation.py',cpu.parent);supervised(monitor/'started.json',monitor/'terminal.json',reviewer,proof);c=read(cpu);r=read(proof);identity=prior()
 if c.get('status')!='passed' or c.get('source_sha256_before')!=plan['source_sha256'] or c.get('source_sha256_after')!=plan['source_sha256'] or c.get('GT_opened') is not False or c.get('GPU_used') is not False or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not preflight_only or r.get('reviewer_sha256')!=sha(reviewer) or r.get('source_sha256')!=plan['source_sha256'] or r.get('cpu_sha256')!=sha(cpu) or r.get('cpu_started_sha256')!=sha(cs) or r.get('cpu_terminal_sha256')!=sha(ct) or r.get('prior_identity')!=identity or r.get('GT_opened') is not False:raise ValueError('memory review mismatch')
 if not preflight_only:
  out=D/'run_preflight001';started=D/'monitor_preflight001/started.json';ended=D/'monitor_preflight001/terminal.json';supervised(started,ended,D/'preflight_launch.py',out)
  launch=read(out/'launch.json');ex=read(out/'exit.json');result=read(out/'worker_preflight.json')
  for key,path in (('preflight_launch',out/'launch.json'),('preflight_exit',out/'exit.json'),('preflight_result',out/'worker_preflight.json'),('preflight_started',started),('preflight_terminal',ended)):
   if r.get(key+'_sha256')!=sha(path):raise ValueError('memory preflight binding mismatch')
  if ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or result.get('status')!='passed' or result.get('cuda_initialized') is not False or result.get('model_loaded') is not False or result.get('GT_opened') is not False or result.get('predictions_opened') is not True or result.get('weights_verified')!=4 or result.get('source_sha256')!=plan['source_sha256']:raise ValueError('memory preflight incomplete')
  if host:
   obj=json.loads(subprocess.run(['docker','inspect',launch['name']],capture_output=True,text=True,timeout=15,check=True).stdout)[0];state=obj['State'];ident=ex.get('container_identity') or {}
   if obj['Id']!=ident.get('container_id') or state['Running'] is not False or state['Pid']!=0 or state['ExitCode']!=0:raise ValueError('memory preflight container state')
 return {'review_sha256':sha(proof),'source_sha256':plan['source_sha256'],'prior_identity':identity,'preflight_only':preflight_only}
