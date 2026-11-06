"""Fail-closed preflight/real-model admission; independent proof required."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(dir,script,out=None):
 s=read(dir/'started.json');t=read(dir/'terminal.json');cwd=Path(s['cwd']);c=s['command']
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or len(c)<2 or (cwd/c[1]).resolve()!=Path(script).resolve() or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=Path(out).resolve()))):raise ValueError('supervisor mismatch')
def admit(preflight_only):
 cpu=D/'cpu_check001.json';terminal(D/'monitor_cpu_check001',D/'cpu_check.py');c=read(cpu)
 if c.get('status')!='passed' or c.get('GPU_used') is not False or c.get('GT_opened') is not False or c.get('source_sha256_before')!=c.get('source_sha256_after') or any(sha(k)!=v for k,v in c['source_sha256_after'].items()):raise ValueError('CPU source freeze')
 name='e4_multistream_expansion352_'+('preflight_review' if preflight_only else 'gpu_review');proof=I/(name+'.json');reviewer=I/('review_'+name+'.py');monitor=I/('monitor_'+name)
 if not proof.is_file():raise ValueError('independent review missing')
 terminal(monitor,reviewer,proof);r=read(proof)
 fields={'reviewer_sha256':reviewer,'cpu_sha256':cpu,'cpu_started_sha256':D/'monitor_cpu_check001/started.json','cpu_terminal_sha256':D/'monitor_cpu_check001/terminal.json','selection_sha256':D/'selection001.json'}
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not preflight_only or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('source_sha256')!=c['source_sha256_after'] or any(r.get(k)!=sha(v) for k,v in fields.items())):raise ValueError('independent source review mismatch')
 if not preflight_only:
  # GPU is permanently locked in this version until a future independent
  # GPU admission explicitly verifies actual Docker preflight and final review.
  raise ValueError('GPU runtime admission not implemented')
 return {'preflight_only':True,'source_sha256':c['source_sha256_after'],'review_sha256':sha(proof),'selection_sha256':sha(D/'selection001.json')}
