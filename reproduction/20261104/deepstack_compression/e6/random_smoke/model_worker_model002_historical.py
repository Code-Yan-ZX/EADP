"""Random K256 diagnostic worker. Missing review keeps all execution locked."""
import argparse,hashlib,importlib.util,inspect,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];LAT=B/'e4_c_latency';E4=B/'e4';RB=B/'e6/random_baselines'
def load(name,path):
 path=Path(path).resolve();prior=sys.modules.get(name)
 if prior is not None:
  if not getattr(prior,'__file__',None) or Path(prior.__file__).resolve()!=path:raise ValueError('foreign module '+name)
  return prior
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def save(path,data):
 with Path(path).open('x') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();out=a.out.resolve()
 if not out.is_dir():raise ValueError('launcher directory missing')
 # The reviewed execution plan verifies smoke_gate.py before it is invoked;
 # this fixed hash must be filled only at final source freeze.
 if hashlib.sha256((D/'smoke_gate.py').read_bytes()).hexdigest()!='90d5dbfeb4c9315958233bdc5651e928bae3e1ae21dec8a491fa6338bc4181e7':raise ValueError('model worker not frozen')
 gate=load('_random_model_gate',D/'smoke_gate.py');approval=gate.admit(a.preflight_only)
 launch=gate.read(out/'launch.json')
 if launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=gate.sha(__file__) or launch.get('run_out')!=str(out):raise ValueError('worker launch binding')
 identity_path=out/'container_identity.json';deadline=time.monotonic()+30
 while True:
  try:cid=gate.read(identity_path);break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>=deadline:raise ValueError('container identity unavailable')
   time.sleep(.1)
 container=cid.get('container_id');own=Path('/proc/self/cgroup').read_text()
 if not isinstance(container,str) or len(container)!=64 or container not in own or container not in cid.get('host_cgroup','') or not (cid.get('host_init') or {}).get('start_ticks'):raise ValueError('actual container identity')
 plan=gate.plan();sys.path[:0]=[str(RB),str(E4),str(LAT),str(D)]
 for name,path in {'selector':RB/'selector.py','budget_adapter':E4/'budget_adapter.py','decode_observer':E4/'decode_observer.py','identity_records':E4/'identity_records.py','off_adapter':E4/'off_adapter.py','skip_deepstack':E4/'skip_deepstack.py'}.items():
  m=sys.modules.get(name)
  if m is not None and (not getattr(m,'__file__',None) or Path(m.__file__).resolve()!=path.resolve()):raise ValueError('foreign preloaded module '+name)
 latency=load('_random_latency_worker',LAT/'latency_worker.py');e1=load('_random_e1_admission',B/'m3_oracle_exec/e1_admission.py');refs=load('_random_e1_refs',D/'e1_refs.py');manifest=load('_random_manifest',D/'manifest_load.py');inv=load('_random_inventory',D/'inventory.py');identity=load('_random_identity_contract',D/'identity_contract.py');core=load('_random_smoke_core',D/'smoke_core.py');diagnostic=load('_random_diagnostic_run',D/'diagnostic_run.py');model=load('_random_model_core',D/'model_core.py')
 selected=load('_random_selector',RB/'selector.py');adapter=load('_random_model_adapter',RB/'adapter.py');runner=load('_random_budget_runner',E4/'budget_runner.py');cache=load('_random_fixed_common',B/'e4_c_multibudget_fixed32/fixed_common.py')
 if Path(inspect.getfile(adapter.RandomAdapter)).resolve()!=RB/'adapter.py' or Path(sys.modules['selector'].__file__).resolve()!=RB/'selector.py':raise ValueError('actual random class/source')
 prepared=latency.prepare('smoke');receipt=e1.validate(metadata_only=False);random_plan=gate.read(D/'plan001.json');choices=manifest.load(D/'selection_manifest001.jsonl',random_plan)
 rows=[]
 for anchor in random_plan['anchors']:
  matches=[r for r in prepared['rows'] if r['task']==anchor['task'] and str(r['qid'])==str(anchor['qid'])]
  if len(matches)!=1:raise ValueError('anchor selection')
  rows.append(matches[0])
 reference=refs.load(B/'e1/e1full001/raw_predictions.jsonl',receipt,rows);inv.validate_inventory(rows,reference,choices,random_plan['anchors'])
 torch,native,wrapper,versions=latency.load_runtime(prepared)
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner
 if a.preflight_only:
  if torch.cuda.is_initialized():raise ValueError('unexpected CUDA initialization')
  save(out/'worker_preflight.json',{'status':'passed','plan_sha256':gate.sha(D/'execution_plan001.json'),'anchors':3,'random_choices':9,'model_loaded':False,'cuda_initialized':False,'GT_opened':False,'predictions_opened':True,'prediction_scope':'accepted E1 DEV mixed predictions parsed for baseline identity only','versions':versions});return
 model.run(prepared,torch,native,wrapper,lambda _receipt,_rows:reference,receipt,random_plan,choices,out,Qwen3VLChatCDPruner.generate_inner_transformers_fixed,selected.selection,runner,adapter.RandomAdapter,cache.CacheFingerprints,identity.validate_e1_identity,core.run_arm,inv,diagnostic,latency.preflight_one,gate.plan)
if __name__=='__main__':main()
