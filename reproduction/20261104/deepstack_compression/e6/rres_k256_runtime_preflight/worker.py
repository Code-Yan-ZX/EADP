"""Container execution-entry imports, without model construction or CUDA."""
import argparse,hashlib,importlib.util,json,os,socket,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/rres_k256_execution';P=B/'e6/rres_k256_pilot_v2';ROOT=B.parents[2];GATE_SHA='f189ff06d6270a380cee70e8557ff444aa0a8f889754917cdad6c14c1da11f30'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==D/'run001/worker.json' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 path=D/'gate.py';assert hashlib.sha256(path.read_bytes()).hexdigest()==GATE_SHA
 spec=importlib.util.spec_from_file_location('_rres_runtime_cpu_gate',path);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);approval=g.admit();deadline=time.monotonic()+60
 while True:
  try:identity=g.read(D/'run001/container_identity.json');launch=g.read(D/'run001/launch.json');break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>deadline:raise TimeoutError('container identity')
   time.sleep(.2)
 cid=identity['container_id'];assert len(cid)==64 and socket.gethostname()==cid[:12] and cid in Path('/proc/self/cgroup').read_text() and launch['approval']==approval
 sys.path[:0]=[str(E),str(P),str(B/'e4_c_latency'),str(B/'e4'),str(ROOT/'_reproduction/author_eadp/Qwen_vl'),str(ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit')]
 import torch,worker as runworker,gpu_gate,arm,capture,resolution,decode_observer,vlmeval.vlm.qwen3_vl.model_fixed_res as author
 for module,expected in ((runworker,E/'worker.py'),(gpu_gate,E/'gpu_gate.py'),(arm,P/'arm.py'),(capture,P/'capture.py'),(resolution,P/'resolution.py'),(decode_observer,B/'e4/decode_observer.py')):
  actual=Path(module.__file__).resolve();assert actual==expected and g.sha(actual)==g.read(D/'plan001.json')['source_sha256'][str(actual)]
 assert Path(author.__file__).resolve()==ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py' and author.QWEN3_FIXED_RESOLUTION==1024 and not torch.cuda.is_initialized()
 try:runworker.import_gate().admit_gpu()
 except FileNotFoundError:locked=True
 else:raise ValueError('GPU unexpectedly unlocked')
 with a.out.open('x') as f:json.dump({'status':'passed','approval':approval,'container_id':cid,'GPU_locked':locked,'GPU_authorized':False,'GT_opened':False,'model_loaded':False,'cuda_initialized':False,'existing_DEV_predictions_opened':False,'imports':['worker','gpu_gate','arm','capture','resolution','decode_observer','author'],'errors':[]},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
