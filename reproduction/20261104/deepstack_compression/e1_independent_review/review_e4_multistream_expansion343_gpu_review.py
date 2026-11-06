#!/usr/bin/env python3
"""Independent final worker GPU review; no new features, labels, or GPU."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e4_multistream_expansion343_gpu';P=B/'e4_multistream_expansion352';H=B/'e4_multistream_expansion343_launch'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(m,script):
 s=read(m/'started.json');t=read(m/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==script.resolve()
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);out=a.parse_args().out.resolve();assert out==I/'e4_multistream_expansion343_gpu_review.json' and not out.exists() and not (D/'run001').exists()
 term(D/'monitor_cpu_check002',D/'cpu_check.py');term(D/'monitor_preflight001',D/'launch.py');term(I/'monitor_e4_multistream_expansion343_preflight001_result',I/'review_e4_multistream_expansion343_preflight001_result.py')
 c=read(D/'cpu_check002.json');p=read(D/'execution_plan001.json');f=read(D/'formal_plan001.json');q=read(I/'e4_multistream_expansion343_preflight001_result_review.json');w=read(D/'preflight001/worker_preflight.json');ex=read(D/'preflight001/exit.json')
 assert c['status']=='passed' and c['source_sha256_before']==c['source_sha256_after'] and all(sha(k)==v for k,v in c['source_sha256_after'].items())
 assert p['images']==343 and p['gpu_inner_seconds']==5400 and p['gpu_outer_seconds']==5700 and p['gpu_stale_seconds']==1800 and p['grace_seconds']==90 and time.time()+p['gpu_outer_seconds']<p['budget_end_unix'] and p['budget_end_unix']==1794065414 and sha(p['budget_source'])==p['budget_source_sha256']
 assert f['images']==343 and f['GT_opened'] is False and f['scoring_performed'] is False and f['automatic_retry'] is False
 assert q['status']=='passed' and q['independent_review'] is True and q['images']==343 and q['processor_inputs_checked']==343 and q['e1_p3d1_complete_references']==343 and not q['errors']
 assert w['status']=='passed' and w['images']==343 and w['processor_inputs_checked']==343 and w['e1_p3d1_complete_references']==343 and w['model_loaded'] is False and w['cuda_initialized'] is False and w['GT_opened'] is False and w['source_sha256']==c['source_sha256_after']
 assert ex['container_stopped_verified'] is True and ex['container_exit_code']==0 and ex['failure_type'] is None
 spec=importlib.util.spec_from_file_location('independent_343_admission',D/'admission.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.admit(True,host=True)
 # Prior independent host inspection checks the exact running image, process, mounts and no device.
 spec=importlib.util.spec_from_file_location('independent_343_host_inspection',H/'gpu_gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);assert g.inspect_preflight(c['source_sha256_after'])['mounts']==12
 fields={'reviewer_sha256':Path(__file__),'cpu_sha256':D/'cpu_check002.json','cpu_started_sha256':D/'monitor_cpu_check002/started.json','cpu_terminal_sha256':D/'monitor_cpu_check002/terminal.json','selection_sha256':P/'selection001.json','preflight_sha256':D/'preflight001/worker_preflight.json','preflight_launch_sha256':D/'preflight001/launch.json','preflight_exit_sha256':D/'preflight001/exit.json','preflight_started_sha256':D/'monitor_preflight001/started.json','preflight_terminal_sha256':D/'monitor_preflight001/terminal.json','preflight_review_sha256':I/'e4_multistream_expansion343_preflight_review.json','formal_plan_sha256':D/'formal_plan001.json'}
 v={'status':'passed','independent_review':True,'preflight_only':False,'GT_opened':False,'GPU_used':False,'new_343_values_opened':False,'errors':[],'source_sha256':c['source_sha256_after'],'checks':['actual full 343 processor no-device preflight and independent verification','CPU frozen source hashes and pilot admission','fresh Docker no-device image/command/environment/mount inspection','fixed deadline and bounded GPU execution','worker and formal contract source reviewed'],'limitations':['no model executed in this review','full tensor to norm linkage remains online identity evidence']}
 v.update({k:sha(x) for k,x in fields.items()})
 with out.open('x') as o:json.dump(v,o,indent=2);o.write('\n');o.flush();os.fsync(o.fileno())
if __name__=='__main__':main()
