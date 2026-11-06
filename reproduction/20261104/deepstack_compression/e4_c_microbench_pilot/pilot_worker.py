"""Real-model single-image component worker; independently reviewed admission is mandatory."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/'e4_c_latency'))
from pilot_admission import admit,sources
from latency_worker import prepare,load_runtime,save
from real_worker import sha,preflight_one
from input_preflight import tensor_meta
from runner import durable_journal
from reference_contract import choose
from stage_core import execute
from fingerprint import fingerprint

def module(path,name):
 path=Path(path).resolve();spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if Path(m.__file__).resolve()!=path:raise ValueError('module path mismatch')
 return m

def pilot_schedule():
 full=json.loads((BASE/'e4_c_microbench/schedule001.json').read_text())['schedule'];return {'stage':'pilot','images':full['images'][:1],'components':full['components'],'jobs':full['jobs'][:60]}

def archive_cpu(torch,value):
 if torch.is_tensor(value):return value.detach().contiguous().cpu()
 if isinstance(value,dict):return {k:archive_cpu(torch,v) for k,v in value.items()}
 if isinstance(value,list):return [archive_cpu(torch,v) for v in value]
 if isinstance(value,tuple):return tuple(archive_cpu(torch,v) for v in value)
 if value is None or isinstance(value,(str,int,float,bool)):return value
 raise TypeError('unsupported archive value')

def get_reference(row):
 proof=json.loads((BASE/'e1/e1full001_completion.json').read_text());j=proof['files']['journal'];path=Path(j['path'])
 if sha(path)!=j['sha256']:raise ValueError('E1 journal changed')
 with path.open() as f:ref=choose((json.loads(line) for line in f if line.strip()),row)
 if sha(path)!=j['sha256']:raise ValueError('E1 journal changed while reading')
 return ref

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();out=a.out.resolve();approval=admit(preflight_only=a.preflight_only,host=False);launch=json.loads((out/'launch.json').read_text());source={str(p):sha(p) for p in sources()}
 if launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=sha(__file__) or source!=approval['source_sha256']:raise ValueError('launch/source mismatch')
 prepared=prepare('smoke');schedule=pilot_schedule();image=tuple(schedule['images'][0]);rows=[r for r in prepared['rows'] if (r['task'],r['group_id'])==image and str(r['qid'])=='16539']
 if len(rows)!=1 or len(schedule['jobs'])!=60:raise ValueError('frozen image/sample mismatch')
 row=rows[0];torch,native,wrapper_class,versions=load_runtime(prepared)
 # Same exact imports and source paths are checked before returning from CPU preflight.
 capture=module(BASE/'e4_c_microbench/capture.py','component_capture_unique');light=module(BASE/'e4_c_latency/light_adapter.py','component_light_unique');from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized before model construction')
 if a.preflight_only:
  save(out/'worker_preflight.json',{'status':'passed','jobs':60,'sample':{'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id']},'cuda_initialized':False,'model_loaded':False,'predictions_opened':True,'prediction_scope':'accepted E1 DEV mixed journal parsed only to extract input and model identity; no answer or output used','GT_opened':False,'scoring_performed':False,'source_sha256':source,'weight_stats_before':prepared['weights_before'],'versions':versions});return
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 start=time.monotonic();wrapper=wrapper_class(**prepared['plan']['recipe']);load_seconds=time.monotonic()-start
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('model backend mismatch')
 ref=get_reference(row);device=torch.device('cuda:0');stream=torch.cuda.current_stream(device).cuda_stream;handle,journal=durable_journal(out/'journal.jsonl')
 def arch(actual,references):
  path=out/'reference.pt';torch.save({'actual':archive_cpu(torch,actual),'references':archive_cpu(torch,references)},path)
  with path.open('rb') as f:os.fsync(f.fileno())
  journal({'status':'archive_written','path':str(path),'sha256':sha(path)})
 try:
  result=execute(torch,wrapper,native,row,ref,schedule,schedule,preflight_one,Qwen3VLChatCDPruner.generate_inner_transformers_fixed,light.LightAdapter,capture.CaptureInputs,capture.clone_tree,tensor_meta,lambda a,b:capture.equal_tree(torch,a,b),lambda x:fingerprint(torch,x),lambda:torch.cuda.Event(enable_timing=True),lambda:torch.cuda.synchronize(device),lambda:torch.cuda.current_stream(device).cuda_stream==stream,journal,arch)
 finally:handle.close()
 if any(sha(p)!=d for p,d in source.items()):raise ValueError('source changed during pilot')
 for path,before in prepared['weights_before'].items():
  s=Path(path).stat()
  if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weight identity changed')
 save(out/'worker_complete.json',{'status':'completed','result':result,'source_sha256':source,'schedule':schedule,'model_load_seconds':load_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'versions':versions,'GT_opened':False,'scoring_performed':False,'scope':'isolated warm-cache component diagnostic; DS staging excludes decoder addition, not TTFT'})
if __name__=='__main__':main()
