"""Model worker candidate. CLI is deliberately locked; no GPU authority."""
import importlib.util,inspect,json,sys,time
from pathlib import Path
import process_core
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PROJECT=ROOT.parents[2]
def execute_after_future_admission(job,prepared,out):
 """Not callable from CLI. Future launcher MUST establish admission first."""
 import torch
 author=PROJECT/'_reproduction/author_eadp/Qwen_vl';sys.path[:0]=[str(author),str(author/'VLMEvalKit'),str(ROOT/'e6/rres_k256_pilot_v2')]
 import vlmeval.vlm.qwen3_vl.model_fixed_res as fixed,resolution
 from model import pruner
 from transformers import StoppingCriteriaList
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes,Qwen3VLChatEADP,Qwen3VLChatCDPruner
 adapter_path=ROOT/'e4_c_budget_pilot_v2/adapter.py';spec=importlib.util.spec_from_file_location('_paired_timing_adapter_worker',adapter_path);adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
 if Path(inspect.getfile(Qwen3VLChatFixedRes)).resolve()!=(author/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve():raise ValueError('author class location')
 recipe=json.loads((PROJECT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json').read_text())['recipe']
 if job['arm']=='rres':
  recipe={k:v for k,v in recipe.items() if k not in ('visual_token_num','alpha','beta')};model=Qwen3VLChatFixedRes(**recipe)
  def author_generate(w,row):
   with torch.no_grad():return w.generate_inner(row['message'],dataset=row['task'])
 elif job['arm']=='eadp':
  model=Qwen3VLChatEADP(**recipe)
  def author_generate(w,row):
   with torch.no_grad():return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
 else:raise ValueError('arm')
 if model.model.dtype!=torch.bfloat16 or model.model.config._attn_implementation!='flash_attention_2':raise ValueError('model backend')
 started=time.monotonic_ns()
 result=process_core.run_process(job,prepared['rows'],prepared['references'],torch=torch,wrapper=model,module=fixed,pruner=pruner,adapter_type=adapter.BudgetLightAdapter,criteria_type=StoppingCriteriaList,fixed_resolution=resolution.fixed_resolution,author_generate=author_generate,out=out)
 return {**result,'model_ready_monotonic_ns':started,'model_load_excluded':True}
def main():raise RuntimeError('GPU worker is locked pending complete execution, preflight and independent admission')
if __name__=='__main__':main()
