"""Bind actual noGPU processor acceptance to the execution-entry import probe."""
import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];P=B/'e6/rres_k256_pilot_v2';I=B/'e1_independent_review';E=B/'e6/rres_k256_execution'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 cpu=read(D/'cpu_proof002.json');term=read(D/'monitor_cpu002/terminal.json');p=read(P/'preflight_plan001.json');proof=I/'rres_k256_processor_v2_preflight_result_review.json';q=read(proof);reviewer=I/'review_rres_k256_processor_v2_preflight_result.py';mon=I/'monitor_rres_k256_processor_v2_preflight_result';s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if cpu['status']!='passed' or cpu['before']!=cpu['after'] or term['reason']!='completed' or term['worker_returncode']!=0 or term['remaining_process_group']!=[]:raise ValueError('CPU proof')
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('GPU_authorized') is not False or q.get('worker_sha256')!=sha(P/'run_preflight001/worker_preflight.json') or q.get('reviewer_sha256')!=sha(reviewer) or q.get('errors')!=[] or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=reviewer or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=proof:raise ValueError('prior preflight review')
 source=dict(p['source_sha256']);source.update(cpu['after']);extra=[P/'preflight_plan001.json',P/'run_preflight001/worker_preflight.json',P/'run_preflight001/processor_probe.json',P/'run_preflight001/launch.json',P/'run_preflight001/container_identity.json',P/'run_preflight001/exit.json',P/'monitor_preflight001/started.json',P/'monitor_preflight001/terminal.json',proof,reviewer,mon/'started.json',mon/'terminal.json',D/'cpu_proof002.json',D/'monitor_cpu002/started.json',D/'monitor_cpu002/terminal.json',B/'e4/decode_observer.py',E/'DESIGN.md']
 for path in extra:source[str(path.absolute())]=sha(path)
 if any(sha(k)!=v for k,v in source.items()):raise ValueError('source changed')
 out={'scope':'runtime noGPU import only','source_sha256':source,'GPU_authorized':False,'GT_opened':False,'model_loaded':False,'weights_payload_rehash':False,'note':'three real processor/weight checks occurred in prior independently reviewed container; this probe imports execution modules and author class only'}
 with (D/'plan001.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
