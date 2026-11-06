"""Thirty-two-image CPU-preflight / GPU zero-threshold anchored-selector diagnostic; no model."""
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];P=B/'e6/external_sources/pixelprune_8d347eb';sys.path.insert(0,str(P))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def emit(p,x):
 with Path(p).open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
def check_plan():
 p=json.loads((D/'plan001.json').read_text());assert p['threshold']==0.0 and len(p['anchors'])==32 and all(sha(k)==v for k,v in p['source_sha256'].items());return p
def load_candidate():
 path=B/'e6/pixel_exact_zero_cpu/candidate.py';spec=importlib.util.spec_from_file_location('eadp_zero_candidate',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);assert Path(m.__file__).resolve()==path;return m.select
def imports():
 import torch
 from pixelprune.methods import pred_2d
 from pixelprune.patches import qwen3_vl_hf
 assert Path(pred_2d.__file__).resolve()==P/'pixelprune/methods/pred_2d.py' and Path(qwen3_vl_hf.__file__).resolve()==P/'pixelprune/patches/qwen3_vl_hf.py'
 return torch,pred_2d.Pred2DSelector,qwen3_vl_hf.normalize_pixel_values_for_selector,load_candidate()
def preflight(plan):
 torch,author,normalize,candidate=imports();assert not torch.cuda.is_initialized();cases=[]
 for x in plan['anchors']:
  data=torch.load(x['capture_path'],map_location='cpu',weights_only=True)['actual']['processor'];assert data['image_grid_thw'].tolist()==[[1,64,64]] and data['pixel_values'].dtype==torch.bfloat16 and torch.isfinite(data['pixel_values']).all()
  v=normalize(data['pixel_values']).reshape(1024,-1);selected=candidate(v,32,32).tolist();assert selected==x['expected_indices'];cases.append({'task':x['task'],'qid':x['qid'],'k':len(selected),'capture_sha256':sha(x['capture_path'])})
 assert not torch.cuda.is_initialized();return {'status':'passed','plan_sha256':sha(D/'plan001.json'),'cases':cases,'cuda_initialized':False,'model_loaded':False,'GT_opened':False,'new_predictions_opened':False,'gpu_used':False,'torch_version':torch.__version__}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();plan=check_plan()
 if a.preflight_only:
  if a.out.resolve()!=D/'preflight001.json':raise ValueError('output')
  save(a.out,preflight(plan));return
 # This review is made by a separate agent after the real no-GPU preflight.
 from gate import require_final
 if Path(sys.modules['gate'].__file__).resolve()!=D/'gate.py':raise ValueError('foreign gate')
 require_final();torch,author,normalize,candidate=imports()
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 torch.cuda.set_per_process_memory_fraction(.5,0);a.out.mkdir(exist_ok=False);journal=a.out/'progress.jsonl';outcomes=[]
 for i,x in enumerate(plan['anchors']):
  key=x['task']+'/'+x['qid'];emit(journal,{'event':'case_started','key':key});case={'key':key,'arms':[],'status':'failed'}
  try:
   data=torch.load(x['capture_path'],map_location='cpu',weights_only=True)['actual']['processor'];pixel=data['pixel_values'].to('cuda');v=normalize(pixel).reshape(1024,-1).contiguous();assert torch.isfinite(v).all().item()
   # Same three-by-three independent warmup for each implementation, outside timing.
   tiny=v[:9];sel=author(method='max',threshold=0.0,anchored=True)
   with torch.inference_mode():
    sel._select_2d_loco_anchored(tiny,3,3,tiny.device);candidate(tiny,3,3);torch.cuda.synchronize()
    names=['author','zero_parallel'] if i%2==0 else ['zero_parallel','author']
    for name in names:
     emit(journal,{'event':'arm_started','key':key,'arm':name});torch.cuda.synchronize();t=time.perf_counter_ns()
     idx=sel._select_2d_loco_anchored(v,32,32,v.device) if name=='author' else candidate(v,32,32)
     torch.cuda.synchronize();end=time.perf_counter_ns();elapsed=end-t;ids=idx.detach().cpu().tolist();record={'arm':name,'wall_ns_synchronized':elapsed,'start_monotonic_ns':t,'end_monotonic_ns':end,'indices':ids,'matches_cpu':ids==x['expected_indices']};case['arms'].append(record);emit(journal,{'event':'arm_completed','key':key,**record})
   case['ordered_equal']=case['arms'][0]['indices']==case['arms'][1]['indices'];case['status']='completed'
  except Exception as e:
   case['failure']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
  outcomes.append(case);emit(journal,{'event':'case_recorded','key':key,'status':case['status']})
 check_plan();save(a.out/'complete.json',{'status':'attempts_completed','cases':outcomes,'GT_opened':False,'model_loaded':False,'new_predictions_opened':False,'limitations':['one warmup and one synchronized host-wall measurement per arm per image; not an end-to-end model result','same input includes no processor or host-to-device transfer; author .item synchronization is included','GPU results apply only to these 32 post-hoc images and threshold','allocator limit is not a device-wide or NVML peak']})
if __name__=='__main__':main()
