#!/usr/bin/env python3
"""Independent review authorizing only same-image no-device runtime import."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_k256_runtime_preflight';E=B/'e6/rres_k256_execution';P=B/'e6/rres_k256_pilot_v2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(m,script):
 s=read(m/'started.json');t=read(m/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==script.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out.resolve();assert out==I/'rres_k256_runtime_preflight_review.json' and not out.exists() and not (D/'run001').exists()
 term(D/'monitor_cpu002',D/'cpu_check.py');term(D/'monitor_freeze001',D/'freeze.py');term(I/'monitor_rres_k256_processor_v2_preflight_result',I/'review_rres_k256_processor_v2_preflight_result.py')
 p=read(D/'plan001.json');c=read(D/'cpu_proof002.json');q=read(I/'rres_k256_processor_v2_preflight_result_review.json')
 assert p['scope']=='runtime noGPU import only' and p['GPU_authorized'] is False and all(sha(k)==v for k,v in p['source_sha256'].items()) and c['status']=='passed' and c['before']==c['after']
 assert q['status']=='passed' and q['independent_review'] is True and q['images']==3 and q['GPU_authorized'] is False and q['worker_sha256']==sha(P/'run_preflight001/worker_preflight.json') and q['reviewer_sha256']==sha(I/'review_rres_k256_processor_v2_preflight_result.py') and not q['errors']
 assert all(str(x.resolve()) in p['source_sha256'] for x in (D/'launch.py',D/'worker.py',D/'gate.py',E/'worker.py',E/'gpu_gate.py',P/'arm.py',P/'capture.py',P/'resolution.py'))
 code=(D/'worker.py').read_text();assert 'runworker.import_gate().admit_gpu()' in code and 'not torch.cuda.is_initialized()' in code and "'GPU_authorized':False" in code
 launcher=(D/'launch.py').read_text();assert "'--device' in command" in launcher and 'run_bounded' in launcher
 z=subprocess.run([sys.executable,str(D/'launch.py'),'--out',str(D/'run001')],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=90)
 assert z.returncode!=0 and 'monitor_rres_k256_runtime_preflight' in z.stderr and not (D/'run001').exists(),z.stderr[-1200:]
 value={'status':'passed','independent_review':True,'GPU_authorized':False,'GT_opened':False,'errors':[],'source_sha256':p['source_sha256'],'plan_sha256':sha(D/'plan001.json'),'reviewer_sha256':sha(__file__),'checks':['prior independently accepted 3-processor/weight preflight','real runtime worker, GPU gate, author imports source bound','actual launch refuses absent proof','device-less command and bounded cleanup source reviewed'],'limits':['only permits real same-image no-device runtime import; no model or GPU','three posthoc samples, 256 tokens does not mean equal compute']}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
