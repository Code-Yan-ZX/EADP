"""Prepared PriorTR DEV32 worker; hard locked by gate.execution_gate.

--preflight-only exercises the same input/source/weights and isolated imports,
without loading a model, initializing CUDA, parsing predictions or labels.
"""
import argparse, hashlib, importlib.util, inspect, json, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];PREP=HERE.parent/'preparation';INSTALL=HERE.parent/'install001'
sys.path.insert(0,str(PREP))
import gate,source_inventory
from panel_core import run
from runtime import invoker

def lines(p):
 with open(p,encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]

def load_exact(name,path):
 path=Path(path)
 if name in sys.modules:raise ImportError('local module name already registered')
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
 try:
  spec.loader.exec_module(mod)
  if Path(mod.__file__).resolve()!=path.resolve():raise ImportError('unexpected local import')
  return mod
 except BaseException:
  sys.modules.pop(name,None);raise

def validate_inputs(sample,e1):
 rows=lines(e1['files']['inputs']['path']);panel=lines(e1['files']['panel']['path']);bykey={(x['task'],str(x['qid'])):x for x in rows};dev={(x['task'],str(x['qid'])):x for x in panel if x.get('split')=='DEV'}
 if len(rows)!=384 or len(bykey)!=384 or len(dev)!=384:raise ValueError('E1 inventory')
 selected=[]
 for r in sample:
  key=(r['task'],str(r['qid']));source=bykey.get(key);p=dev.get(key)
  if source is None or p is None or any(source.get(k)!=r.get(k) for k in ('task','qid','group_id','image_path','image_sha256','message','message_sha256')) or source['group_id']!=p['group_id']:raise ValueError('sample changed')
  if gate.sha(r['image_path'])!=r['image_sha256'] or hashlib.sha256(json.dumps(r['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()!=r['message_sha256']:raise ValueError('sample image or prompt differs')
  if [x['value'] for x in r['message'] if x.get('type')=='image']!=[r['image_path']]:raise ValueError('image message differs')
  selected.append(r)
 if len(selected)!=32 or len({(x['task'],str(x['qid'])) for x in selected})!=32 or len({x['group_id'] for x in selected})!=32:raise ValueError('sample denominator')
 return selected

def validate_launch_binding(out,approval):
 out=Path(out).resolve();launch=gate.read(out/'launch.json');cmd=launch.get('command',[])
 if (not out.is_dir() or (out/'data').exists() or (out/'frozen.json').exists() or launch.get('approval')!=approval or
     launch.get('worker_sha256')!=gate.sha(HERE/'panel_worker.py') or launch.get('launcher_sha256')!=gate.sha(HERE/'launch.py') or
     launch.get('budget_sha256')!=gate.sha(HERE/'resource_plan.json') or not isinstance(launch.get('controller'),dict) or
     cmd.count(str(HERE/'panel_worker.py'))!=1 or cmd.count('--out')!=1 or Path(cmd[cmd.index('--out')+1]).resolve()!=out or '--execute' not in cmd or '--preflight-only' in cmd):
  raise ValueError('run launch binding invalid')
 return launch

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--preflight-only',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if not a.execute:raise ValueError('explicit execution required')
 approval=gate.execution_gate(preflight=a.preflight_only)
 if not a.preflight_only:validate_launch_binding(a.out,approval)
 upstream=gate.upstream();plan=json.loads((HERE/'sample_001.json').read_text());e1=json.loads((BASE/'e1/e1full001_completion.json').read_text());rows=validate_inputs(plan['sample'],e1)
 weight_file=BASE.parents[1]/'20261028/qwen/qwen_weight_identity.json';weights=json.loads(weight_file.read_text());stats={}
 if len(weights['files'])!=4:raise ValueError('weight inventory')
 for z in weights['files']:
  path=Path(z['path']);st=path.stat()
  if st.st_size!=z['bytes'] or gate.sha(path)!=z['expected_lfs_sha256']:raise ValueError('weight identity')
  stats[str(path)]=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
 import torch,transformers,visual_token_pruning
 import transformers.models.qwen3_vl.modeling_qwen3_vl as actual_qwen
 from transformers import AutoProcessor
 from visual_token_pruning.config import VTRConfig
 from visual_token_pruning.model.vtr_qwen3_vl import VTRQwen3VLForConditionalGeneration
 prepare=load_exact('_priortr_dev32_preprocess',PREP/'preprocess.py').prepare
 Trace=load_exact('_priortr_dev32_trace',PREP/'trace.py').Trace
 verify_env=load_exact('_priortr_dev32_env_contract',PREP/'env_contract.py').verify_env
 env_fingerprint=verify_env()
 provenance=json.loads((INSTALL/'provenance.json').read_text())
 if provenance['priortr_commit']!='e442a4c14af1b333067a6b8310c9111cad922033' or provenance['transformers_commit']!='f8f2834e1a82970f93e06f9bb076a361ca2d70c0':raise ValueError('source revisions changed')
 for name,digest in provenance['original_source_sha256'].items():
  if gate.sha(INSTALL/'source'/name)!=digest:raise ValueError('isolated source changed')
 if (Path(sys.prefix).resolve()!=(INSTALL/'env').resolve() or torch.__version__!='2.10.0+cu128' or transformers.__version__!='5.2.0.dev0' or
     Path(visual_token_pruning.__file__).resolve()!=(INSTALL/'source/visual_token_pruning/__init__.py').resolve() or Path(actual_qwen.__file__).resolve()!=(INSTALL/'source/qwen3/modeling_qwen3_vl.py').resolve() or
     Path(inspect.getfile(prepare)).resolve()!=(PREP/'preprocess.py').resolve() or Path(inspect.getfile(Trace)).resolve()!=(PREP/'trace.py').resolve()):raise ValueError('runtime import identity')
 if a.preflight_only:
  if torch.cuda.is_initialized():raise ValueError('preflight initialized CUDA')
  print(json.dumps({'status':'preflight_passed','cuda_initialized':False,'model_loaded':False,'predictions_parsed':False,'labels_opened':False,'samples':len(rows),'env_fingerprint_sha256':env_fingerprint,'approval':approval,'upstream':upstream}),flush=True);return
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('wrong GPU')
 refs={}
 for r in lines(e1['files']['journal']['path']):
  if r.get('forced') is False and r.get('arm')=='P3-D1':
   key=(r['task'],str(r['qid']))
   if key in refs:raise ValueError('duplicate E1 reference')
   refs[key]=r
 if len(refs)!=384 or any((r['task'],str(r['qid'])) not in refs for r in rows):raise ValueError('E1 reference inventory')
 out=a.out.resolve()
 if not out.is_dir() or (out/'data').exists() or (out/'frozen.json').exists():raise ValueError('invalid run output')
 recipe=json.loads((BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json').read_text())['recipe']
 frozen={'approval':approval,'upstream':upstream,'weight_identity_sha256':gate.sha(weight_file),'env_fingerprint_sha256':env_fingerprint,'provenance_sha256':gate.sha(INSTALL/'provenance.json'),'weight_stats_before':stats,'GT_opened':False,'scoring_performed':False,'source_sha256':{str(x):gate.sha(x) for x in source_inventory.sources()}}
 with (out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 processor=AutoProcessor.from_pretrained(recipe['model_path'],local_files_only=True,trust_remote_code=False)
 model=VTRQwen3VLForConditionalGeneration.from_pretrained(recipe['model_path'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
 if model.dtype!=torch.bfloat16 or model.config._attn_implementation!='sdpa':raise ValueError('backend differs')
 model.vtr_config=VTRConfig(enabled=False);model.vtr_strategy=model._create_strategy('priortr')
 # run() creates data and arrays before invoking the first arm.
 runner=invoker(torch=torch,model=model,processor=processor,VTRConfig=VTRConfig,Trace=Trace,prepare=prepare,arrays_dir=out/'data/arrays')
 result=run(rows=rows,e1_references=refs,out=out/'data',invoke=runner)
 for path,expected in stats.items():
  st=Path(path).stat()
  if [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]!=expected:raise ValueError('weight changed')
 if any(gate.sha(path)!=digest for path,digest in frozen['source_sha256'].items()):raise ValueError('source changed')
 with (out/'worker_complete.json').open('x') as f:json.dump(result,f);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
