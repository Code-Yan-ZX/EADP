#!/usr/bin/env python3
"""Post-run independent audit of mechanical snapshot/formal guard repair."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;V=B/'e6/pixel_wavefront_t020_full_dev_accept_v2';D=B/'e6/pixel_wavefront_t020_full_dev'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(m,script):
 s=read(m/'started.json');t=read(m/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==script.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out.resolve();assert out==I/'pixel_wavefront_full_accept_v2_prelaunch_review.json' and not out.exists() and not (V/'snapshot002').exists()
 term(V/'monitor_freeze002',V/'freeze.py');term(V/'monitor_cpu002',V/'cpu_validation.py');term(D/'monitor_run001',D/'launch.py')
 p=read(V/'acceptance_plan002.json');c=read(V/'cpu_proof002.json');s=read(V/'guard_smoke002.json');failed=read(D/'monitor_snapshot001/terminal.json')
 assert p['run_plan_sha256']==sha(D/'plan002.json') and all(sha(k)==v for k,v in p['source_sha256'].items()) and all(sha(k)==v for k,v in p['prior_evidence_sha256'].items())
 assert c['status']=='passed' and c['acceptance_plan_sha256']==sha(V/'acceptance_plan002.json') and c['source_sha256']==p['source_sha256'] and c['guard_smoke_sha256']==sha(V/'guard_smoke002.json') and c['test_log_sha256']==sha(V/'test_guard002.log')
 assert s['status']=='passed' and s['candidate_outputs_opened'] is False and s['GT_opened'] is False and s['acceptance_plan_sha256']==sha(V/'acceptance_plan002.json') and failed['reason']=='worker_failed' and failed['worker_returncode']!=0
 old=(D/'finish_guard.py').read_text();new=(V/'finish_guard.py').read_text();assert old.replace('import launch\n    runtime=launch.runtime_module()','import launch as launch_module\n    runtime=launch_module.runtime_module()') in new.replace("import sys\nORIGINAL=Path(__file__).resolve().parent.parent/'pixel_wavefront_t020_full_dev'\nsys.path.insert(0,str(ORIGINAL))\n",'').replace('D=ORIGINAL','D=Path(__file__).resolve().parent')
 # Exercise the real retained container guard with current inspect, no candidate output reads.
 sys.path.insert(0,str(V));import finish_guard
 actual=finish_guard.validate();assert actual==s['actual']
 z=subprocess.run([sys.executable,str(V/'snapshot.py')],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=90)
 assert z.returncode!=0 and not (V/'snapshot002').exists() and 'monitor_pixel_wavefront_full_accept_v2_prelaunch' in z.stderr,z.stderr[-1000:]
 fields={'acceptance_plan_sha256':V/'acceptance_plan002.json','reviewer_sha256':Path(__file__),'cpu_proof_sha256':V/'cpu_proof002.json','cpu_started_sha256':V/'monitor_cpu002/started.json','cpu_terminal_sha256':V/'monitor_cpu002/terminal.json','guard_smoke_sha256':V/'guard_smoke002.json'}
 v={'status':'passed','independent_review':True,'errors':[],'GT_opened':False,'candidate_outputs_opened':False,'checks':['original run source and failed snapshot evidence unchanged','only guard name shadow fixed; separate post-run verifier provenance','real retained GPU container, complete command and stopped state rechecked','snapshot refuses before independent approval','original planned 384-row comparison retained in verifier'],'limits':['verification repair and review after capture; no candidate outcome used','old high-dimensional KV/logits absent; no tensor equivalence claim']};v.update({k:sha(x) for k,x in fields.items()})
 with out.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
