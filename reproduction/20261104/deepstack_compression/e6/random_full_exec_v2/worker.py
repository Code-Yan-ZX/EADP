"""DEV-only 384 x 3 continuation; execution gated by independently reviewed plan."""
import argparse, hashlib, importlib.util, inspect, json, os, sys, time, traceback
from pathlib import Path
D=Path(__file__).resolve().parent; B=D.parents[1]; ROOT=B.parents[2]
LAT=B/'e4_c_latency'; E4=B/'e4'; RB=B/'e6/random_baselines'; RS=B/'e6/random_smoke'

def load(name,path):
    path=Path(path).resolve(); prior=sys.modules.get(name)
    if prior is not None:
        if not getattr(prior,'__file__',None) or Path(prior.__file__).resolve()!=path: raise ValueError('foreign module '+name)
        return prior
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    try: spec.loader.exec_module(mod)
    except BaseException: del sys.modules[name];raise
    return mod

def save(path,obj):
    with Path(path).open('x') as f: json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def load_gate(path=D/'gate.py'):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!='f1648d328a45e538dc39fbeaf75902ba87bda6454fed9fd3d4e1128c6363328b':raise ValueError('unreviewed gate source')
    return load('_random_full_gate',path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    out=a.out.resolve()
    if not out.is_dir():raise ValueError('launcher output required')
    gate=load_gate();approval=gate.admit(a.preflight_only)
    launch=gate.read(out/'launch.json')
    if launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=gate.sha(__file__) or launch.get('run_out')!=str(out):raise ValueError('launch mismatch')
    deadline=time.monotonic()+30
    while True:
        try: cid=gate.read(out/'container_identity.json');break
        except (FileNotFoundError,json.JSONDecodeError):
            if time.monotonic()>=deadline:raise ValueError('container identity missing')
            time.sleep(.1)
    container=cid.get('container_id');own=Path('/proc/self/cgroup').read_text()
    if not isinstance(container,str) or len(container)!=64 or container not in own or container not in cid.get('host_cgroup','') or not (cid.get('host_init') or {}).get('start_ticks'):raise ValueError('container identity')
    sys.path[:0]=[str(RB),str(E4),str(LAT),str(RS)]
    expected={'selector':RB/'selector.py','budget_adapter':E4/'budget_adapter.py','decode_observer':E4/'decode_observer.py','identity_records':E4/'identity_records.py','off_adapter':E4/'off_adapter.py','skip_deepstack':E4/'skip_deepstack.py',**{x:LAT/(x+'.py') for x in ('latency_gate','real_worker','runner','stage_runner','schedule','contract','timer','input_preflight','e1_reference_gate','identity_gate','light_adapter','memory_probe')}}
    def check_modules():
        for name,path in expected.items():
            old=sys.modules.get(name)
            if old is not None and (not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=path.resolve()):raise ValueError('foreign module '+name)
    check_modules()
    latency=load('_random_full_latency',LAT/'latency_worker.py');e1=load('_random_full_e1',B/'m3_oracle_exec/e1_admission.py')
    refs=load('_random_full_refs',D/'references.py');manifest=load('_random_full_manifest',RS/'manifest_load.py')
    inventory=load('_random_full_inventory',B/'e6/random_full/inventory.py');core=load('_random_full_core',D/'core.py')
    selected=load('_random_full_selector',RB/'selector.py');adapter=load('_random_full_adapter',RB/'adapter.py')
    runner=load('_random_full_budget_runner',E4/'budget_runner.py');identity=load('_random_full_identity',RS/'identity_contract.py')
    smoke=load('_random_full_smoke_core',RS/'smoke_core.py');cache=load('_random_full_cache',B/'e4_c_multibudget_fixed32/fixed_common.py')
    check_modules()
    if Path(inspect.getfile(adapter.RandomAdapter)).resolve()!=(RB/'adapter.py').resolve() or Path(sys.modules['selector'].__file__).resolve()!=(RB/'selector.py').resolve():raise ValueError('actual random class/source')
    prepared=latency.prepare('smoke');receipt=e1.validate(metadata_only=False)
    reference=refs.load(B/'e1/e1full001/raw_predictions.jsonl',receipt,prepared['rows'])
    choices=manifest.load(RS/'selection_manifest001.jsonl',gate.read(RS/'plan001.json'))
    inventory.schedule(prepared['rows'],choices)
    torch,native,wrapper_class,versions=latency.load_runtime(prepared)
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner
    if a.preflight_only:
        if torch.cuda.is_initialized():raise ValueError('unexpected CUDA')
        save(out/'worker_preflight.json',{'status':'passed','questions':384,'choices':1152,'GT_opened':False,'predictions_opened':True,'prediction_scope':'accepted DEV mixed prediction journal parsed only for identity','model_loaded':False,'cuda_initialized':False,'versions':versions});return
    if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
    torch.cuda.set_per_process_memory_fraction(.9,0)
    def progress(event,**kw):
        with (out/'progress.jsonl').open('a') as f: json.dump({'event':event,'unix':time.time(),**kw},f);f.write('\n');f.flush();os.fsync(f.fileno())
    start=time.monotonic();progress('model_loading')
    try:
        wrapper=wrapper_class(**prepared['plan']['recipe']);counter={'steps':0}
        def heartbeat(_module,_args,_kwargs,_output):
            counter['steps']+=1
            if counter['steps']%32==0:progress('forward_progress',steps=counter['steps'])
        hook=wrapper.model.model.language_model.register_forward_hook(heartbeat,with_kwargs=True)
        save(out/'model_ready.json',{'model_load_seconds':time.monotonic()-start,'versions':versions,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'timing_eligible':False})
        def generate(w,row):
            with torch.inference_mode():return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
        def complete():
            source=gate.plan();after={}
            for path,before in prepared['weights_before'].items():
                s=Path(path).stat();after[path]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
                if after[path]!=before:raise ValueError('weights changed')
            save(out/'completion_provenance.json',{'status':'passed','source_sha256':source['source_sha256'],'weight_stats_before':prepared['weights_before'],'weight_stats_after':after,'GT_opened':False,'scoring_performed':False})
        try:
            core.run(prepared,choices,reference,out,torch,native,wrapper,generate,selected.selection,runner,adapter.RandomAdapter,cache.CacheFingerprints,identity.validate_e1_identity,smoke.run_arm,inventory,sys.modules['identity_records'].tensor_meta,latency.preflight_one,complete)
        finally:hook.remove()
        progress('completed')
    except BaseException as exc:
        progress('failed',error_type=type(exc).__name__)
        if not (out/'failure.json').exists():save(out/'failure.json',{'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'GT_opened':False,'scoring_performed':False})
        raise

if __name__=='__main__': main()
