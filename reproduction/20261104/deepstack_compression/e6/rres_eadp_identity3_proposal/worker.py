"""Three-anchor diagnostic worker candidate; GPU entry remains hard locked."""
import inspect,json,os,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PROJECT=ROOT.parents[2];TIMING=ROOT/'e6/rres_eadp_timing_pilot_v4'
sys.path.insert(0,str(TIMING))
import identity_core,module_load,timing_imports

ANCHORS=(('DocVQA_VAL','64588'),('TextVQA_VAL','36511'),('AI2D_TEST','207406'))
def save(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
def hook_counts(model):return [(name,len(module._forward_pre_hooks),len(module._forward_hooks)) for name,module in model.named_modules()]
def execute_after_future_admission(*,arm,prepared,out):
 """Only a future independently admitted controller may call this function."""
 if arm not in ('rres','eadp'):raise ValueError('arm')
 qwen_request,process_core,_runner=timing_imports.load()
 out=Path(out)
 if not out.is_dir():raise ValueError('output')
 import torch
 author=PROJECT/'_reproduction/author_eadp/Qwen_vl';sys.path[:0]=[str(author),str(author/'VLMEvalKit'),str(ROOT/'e6/rres_k256_pilot_v2'),str(ROOT/'e4')]
 import resolution,decode_observer,vlmeval.vlm.qwen3_vl.model_fixed_res as fixed
 from model import pruner
 from transformers import StoppingCriteriaList
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes,Qwen3VLChatEADP,Qwen3VLChatCDPruner
 capture=module_load.load('_paired_identity_capture',ROOT/'e6/rres_k256_pilot_v2/capture.py');adapter=module_load.load('_paired_identity_adapter',ROOT/'e4_c_budget_pilot_v2/adapter.py')
 if Path(inspect.getfile(Qwen3VLChatFixedRes)).resolve()!=(author/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve():raise ValueError('class identity')
 recipe=json.loads((PROJECT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json').read_text())['recipe'];process_core.append(out/'progress.jsonl',{'event':'model_loading','arm':arm,'monotonic_ns':time.monotonic_ns()})
 if arm=='rres':
  wrapper=Qwen3VLChatFixedRes(**{k:v for k,v in recipe.items() if k not in ('visual_token_num','alpha','beta')})
  def author_generate(w,row):
   with torch.no_grad():return w.generate_inner(row['message'],dataset=row['task'])
 else:
  wrapper=Qwen3VLChatEADP(**recipe)
  def author_generate(w,row):
   with torch.no_grad():return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('backend')
 process_core.append(out/'progress.jsonl',{'event':'model_ready','arm':arm,'monotonic_ns':time.monotonic_ns()});results=[]
 for task,qid in ANCHORS:
  row=prepared['rows'][(task,qid)];ref=prepared['references'][(arm,task,qid)];name=task+'_'+qid;process_core.append(out/'progress.jsonl',{'event':'started','arm':arm,'task':task,'qid':qid})
  before_hooks=hook_counts(wrapper.model);original_generate=wrapper.model.generate;original_processor=wrapper._processor_inputs;original_greedy=pruner._greed_select_impl
  def restored():
   try:return hook_counts(wrapper.model)==before_hooks and wrapper.model.generate==original_generate and wrapper._processor_inputs==original_processor and pruner._greed_select_impl is original_greedy and fixed.QWEN3_FIXED_RESOLUTION==1024
   except Exception:return False
  def persist(raw):save(out/(name+'_raw.json'),{'arm':arm,'task':task,'qid':qid,'full_sequence':raw.detach().cpu().tolist()})
  try:
   result=identity_core.run_one(torch=torch,wrapper=wrapper,module=fixed,pruner=pruner,row=row,ref=ref,arm=arm,qwen_request=qwen_request,adapter_type=adapter.BudgetLightAdapter,criteria_type=StoppingCriteriaList,fixed_resolution=resolution.fixed_resolution,author_generate=author_generate,capture_module=capture,decode=decode_observer,persist=persist)
   result['checks']['hooks_and_methods_restored']=restored();result['errors']=[k for k,v in result['checks'].items() if not v];result['status']='passed' if not result['errors'] else 'failed'
   save(out/(name+'.json'),result);process_core.append(out/'progress.jsonl',{'event':'completed','arm':arm,'task':task,'qid':qid,'status':result['status']});results.append(result)
  except BaseException as exc:
   save(out/(name+'_failure.json'),{'arm':arm,'task':task,'qid':qid,'failure_type':type(exc).__name__,'failure':str(exc),'traceback':traceback.format_exc(),'hooks_and_methods_restored':restored()});process_core.append(out/'progress.jsonl',{'event':'failed','arm':arm,'task':task,'qid':qid});raise
 result={'status':'passed' if len(results)==3 and all(x['status']=='passed' for x in results) else 'failed','arm':arm,'anchors':3,'timing_claims':False,'GT_opened':False,'existing_predictions_opened':True};save(out/'worker_complete.json',result);return result
def main():raise RuntimeError('GPU identity worker locked until separate plan, preflight and independent admission')
if __name__=='__main__':main()
