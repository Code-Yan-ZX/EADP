"""384-image 512px generation worker; no GPU admission exists yet."""
import argparse,hashlib,importlib.util,json,os,socket,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];F=B/'e6/rres_k256_dev384';P=B/'e6/rres_k256_pilot_v2';ROOT=B.parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def weight_stats(items):
 result=[]
 for x in items:
  p=Path(x['path']);s=p.stat();result.append({'path':str(p),'stat':[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns],'sha256':x['expected_lfs_sha256']})
 return result
def controls_from_e1(completion):
 info=completion['files']['journal'];path=Path(info['path'])
 if sha(path)!=info['sha256']:raise ValueError('E1 journal changed')
 controls={}
 with path.open() as f:
  for line in f:
   item=json.loads(line)
   if item.get('arm')=='P3-D1' and item.get('forced') is False:
    k=(item['task'],str(item['qid']))
    if k in controls:raise ValueError('duplicate E1 control')
    controls[k]={'generation_scalars':item['generation_scalars'],'eos_ids':item['eos_ids']}
 if len(controls)!=384:raise ValueError('E1 control denominator')
 return controls
def execute_admitted(approval,out):
 """Actual model path, callable only after future execution gate is in place."""
 deadline=time.monotonic()+60
 while True:
  try:identity=read(F/'run001/container_identity.json');launch=read(F/'run001/launch.json');break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>deadline:raise TimeoutError('container identity publication')
   time.sleep(.2)
 cid=identity['container_id']
 if len(cid)!=64 or socket.gethostname()!=cid[:12] or cid not in Path('/proc/self/cgroup').read_text() or launch.get('approval')!=approval:raise ValueError('GPU container identity')
 source=approval['source_sha256']
 if any(sha(k)!=v for k,v in source.items()):raise ValueError('source changed before model')
 pre=read(F/'run_preflight001/worker_preflight.json');weights=read(ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json')['files'];initial=weight_stats(weights)
 if initial!=pre['weights']:raise ValueError('weight endpoints changed')
 sys.path[:0]=[str(F),str(P),str(B/'e4_c_latency'),str(B/'e4'),str(ROOT/'_reproduction/author_eadp/Qwen_vl'),str(ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit')]
 import torch,arm,decode_observer,panel_core,vlmeval.vlm.qwen3_vl.model_fixed_res as module
 from input_preflight import load_accepted_e1_references
 for imported,expected in ((arm,P/'arm.py'),(panel_core,F/'panel_core.py'),(decode_observer,B/'e4/decode_observer.py')):
  path=Path(imported.__file__).resolve()
  if path!=expected.resolve() or sha(path)!=source[str(path)]:raise ValueError('foreign runtime import')
 if torch.cuda.is_initialized() or module.QWEN3_FIXED_RESOLUTION!=1024 or torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('device or model state')
 rows,_=load_accepted_e1_references(B/'e1/e1full001_completion.json',B/'e1/monitorverifye1full001/started.json',B/'e1/monitorverifye1full001/terminal.json');controls=controls_from_e1(read(B/'e1/e1full001_completion.json'))
 anchors=read(F/'design001.json')['anchors'];refs=read(F/'run_preflight001/probe/processor_probe.json')['anchors']
 recipe=read(ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')['recipe'];recipe={k:v for k,v in recipe.items() if k not in ('visual_token_num','alpha','beta')}
 wrapper=module.Qwen3VLChatFixedRes(**recipe)
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('model dtype/attention backend')
 result=panel_core.run(rows,anchors,refs,controls,torch,wrapper,module,arm,decode_observer,out)
 if weight_stats(weights)!=initial or any(sha(k)!=v for k,v in source.items()):raise ValueError('source or weight changed during run')
 result['source_and_weight_endpoints_verified']=True;panel_core.write(out/'worker_complete.json',result)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);a=ap.parse_args();out=a.out.resolve()
 if out!=(F/'run001/observations').resolve() or not out.is_dir():raise ValueError('unplanned output')
 path=D/'gpu_gate.py';expected='049c8b353bd7954db2b62576e7ab1d9b0d9a600d6d2535e0d29969dea621ed74'
 if sha(path)!=expected or '_rres384_gpu_gate' in sys.modules:raise ValueError('GPU gate source')
 sys.path.insert(0,str(F));sp=importlib.util.spec_from_file_location('_rres384_gpu_gate',path);gate=importlib.util.module_from_spec(sp);sys.modules[sp.name]=gate;sp.loader.exec_module(gate);approval=gate.admit_gpu()
 try:execute_admitted(approval,out)
 except BaseException as exc:
  failure=out/'worker_failure.json'
  if not failure.exists():
   with failure.open('x') as f:json.dump({'status':'failed','expected':384,'failure_type':type(exc).__name__,'failure':str(exc),'traceback':traceback.format_exc(),'GT_opened':False,'scoring_performed':False},f);f.write('\n');f.flush();os.fsync(f.fileno())
  raise
if __name__=='__main__':main()
