"""Real formal entry/import order, no GPU, model, results or new predictions."""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];A=B/'e6/pixel_wavefront_t020_timing_accept_v5';R=B/'e6/pixel_wavefront_t020_timing_final_v9';ROOT=B.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA visibility not disabled')
 assert Path.cwd().resolve()==B and not (R/'run_001').exists() and not (A/'verification001.json').exists()
 # Import the actual entry and execute its top-level code, exactly as a non-main module.
 sp=importlib.util.spec_from_file_location('_formal_smoke_entry',A/'verify_entry.py');entry=importlib.util.module_from_spec(sp);sp.loader.exec_module(entry);v=sys.modules['_pixel_t020_formal_verify'];assert Path(v.__file__).resolve()==(A/'verify.py').resolve()
 formal=read(A/'verification_plan001.json');gpu=read(R/'gpu_plan001.json');fs=formal['source_sha256'];gs=gpu['source_sha256'];assert formal['gpu_plan_sha256']==sha(R/'gpu_plan001.json')
 specs=[('_pixel_t020_formal_gpu_gate',R/'gpu_gate.py',gs),('_pixel_t020_formal_core',A/'verify_candidate.py',fs),('_pixel_t020_formal_pipeline',A/'pipeline.py',fs),('_pixel_t020_formal_docker',A/'docker_contract.py',fs),('_pixel_t020_formal_completion',A/'completion_contract.py',fs),('_pixel_t020_formal_window',A/'window_overlap.py',fs),('_pixel_t020_formal_jobcheck',B/'e6/pixel_wavefront_t020_timing_controller_draft/job_contract.py',gs),('_pixel_t020_formal_prefix',R/'raw_prefix.py',gs),('_pixel_t020_formal_refs',B/'e6/pixel_wavefront_t020_model_timing_v2/reference.py',gs)]
 loaded=[]
 for name,path,inventory in specs:
  m=v.load(name,path,inventory);loaded.append({'name':name,'path':str(Path(m.__file__).resolve()),'sha256':sha(path)})
 path=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_panel.py';runtime=v.load('_pixel_t020_formal_runtime',path,gs);diagnose=sys.modules.get('diagnose');expected=path.parent/'diagnose.py'
 if diagnose is None or not getattr(diagnose,'__file__',None) or Path(diagnose.__file__).resolve()!=expected.resolve() or gs.get(str(expected.resolve()))!=sha(expected):raise ValueError('actual diagnose source mismatch')
 assert runtime.PLAN==expected.parent/'run_plan.json'
 # A real absent-result read fails closed; do not mock file APIs or create records.
 missing=R/'run_001/weight_stats_before.json'
 try:v.read(missing)
 except FileNotFoundError:pass
 else:raise ValueError('unexpected run result exists')
 torch=sys.modules.get('torch');initialized=bool(torch.cuda.is_initialized()) if torch is not None else False
 assert not initialized
 value={'status':'passed','scope':'actual formal entry and real dependency import order without proof gate or run data','GPU_used':False,'model_loaded':False,'cuda_initialized':False,'GT_opened':False,'predictions_opened':False,'formal_plan_sha256':sha(A/'verification_plan001.json'),'gpu_plan_sha256':sha(R/'gpu_plan001.json'),'entry_sha256':sha(A/'verify_entry.py'),'verifier_sha256':sha(A/'verify.py'),'diagnose_path':str(expected),'diagnose_sha256':sha(expected),'runtime_path':str(path),'runtime_sha256':sha(path),'loaded':loaded,'actual_missing_result_rejected':True,'limits':['GPU proof admission and actual post-run verification are not executed; only import feasibility and absent result tested.','Does not establish safety against injected modules in another process; formal execution must use the verified clean environment.'],'smoke_sha256':sha(__file__),'errors':[]}
 with (D/'proof001.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
