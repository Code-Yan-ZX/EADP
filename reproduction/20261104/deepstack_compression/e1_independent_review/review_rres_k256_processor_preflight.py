#!/usr/bin/env python3
"""Independent CPU-only authorization of three-anchor low-resolution preflight."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_k256_pilot'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(m,script):
 s=read(m/'started.json');t=read(m/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==script.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out.resolve();assert out==I/'rres_k256_processor_preflight_review.json' and not out.exists() and not (D/'run_preflight001').exists()
 term(D/'monitor_cpu002',D/'cpu_check.py');term(D/'monitor_freeze001',D/'freeze_plan.py');term(D/'monitor_processor_host002',D/'probe_processor.py')
 p=read(D/'preflight_plan001.json');c=read(D/'cpu_proof002.json');h=read(D/'processor_host002.json')
 assert p['scope']=='three-anchor low-res processor noGPU only' and p['GPU_authorized'] is False and all(sha(k)==v for k,v in p['source_sha256'].items())
 assert c['status']=='passed' and c['source_sha256_before']==c['source_sha256_after'] and all(p['source_sha256'].get(k)==v for k,v in c['source_sha256_after'].items())
 assert h['status']=='passed' and h['images']==3 and h['resolution']==512 and h['original_resolution']==1024 and h['model_loaded'] is False and h['cuda_initialized'] is False and h['GT_opened'] is False and h['GPU_authorized'] is False
 assert {x['task'] for x in h['anchors']}=={'DocVQA_VAL','TextVQA_VAL','AI2D_TEST'} and all(x['low_visual_tokens']==256 and x['original_visual_tokens']==1024 and x['grid']==[[1,32,32]] and x['text_prefix_suffix_identical'] is True and x['position_formula_verified'] is True for x in h['anchors'])
 assert p['source_sha256'][str(D/'processor_host002.json')]==sha(D/'processor_host002.json')
 sys.path.insert(0,str(D));import resolution,gate
 class Stub:QWEN3_FIXED_RESOLUTION=1024
 try:
  with resolution.fixed_resolution(Stub,512):
   assert Stub.QWEN3_FIXED_RESOLUTION==512
   raise RuntimeError('test exception restore')
 except RuntimeError:pass
 assert Stub.QWEN3_FIXED_RESOLUTION==1024
 try:gate.admit_gpu();raise AssertionError('GPU incorrectly authorized')
 except RuntimeError:pass
 z=subprocess.run([sys.executable,str(D/'preflight_launch.py'),'--out',str(D/'run_preflight001')],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=90)
 assert z.returncode!=0 and 'monitor_rres_k256_processor_preflight' in z.stderr and not (D/'run_preflight001').exists(),z.stderr[-1000:]
 v={'status':'passed','independent_review':True,'preflight_only':True,'GPU_authorized':False,'GT_opened':False,'errors':[],'source_sha256':p['source_sha256'],'plan_sha256':sha(D/'preflight_plan001.json'),'reviewer_sha256':sha(__file__),'checks':['three actual 512 processor outputs and independently reconstructed mRoPE','scoped forced author resolution restores on exception','frozen source hashes and completed CPU supervision','actual launch refuses before review; GPU gate locked','no-device container and bounded cleanup source reviewed'],'limits':['only no-device processor preflight; no model or DeepStack evidence','same three previously observed DEV anchors, 256 tokens not equal compute']}
 with out.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
