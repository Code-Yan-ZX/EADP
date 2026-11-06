"""Freeze CPU/preflight inputs, not a GPU launch authorization."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[2]
sys.path.insert(0,str(HERE));from contract import make_schedule

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()

def main():
 old=json.loads((BASE/'e5/graph_ttft_pilot_v3/plan001.json').read_text());anchors=[{**a,'split':'DEV'} for a in old['anchors']]
 source=[HERE/x for x in ('contract.py','observer.py','worker.py','test_contract.py','test_worker.py','preflight.py','preflight_launch.py','freeze_plan.py','cpu_check.py','DESIGN.md')]
 source += [BASE/'e4_c_latency'/x for x in ('timer.py','runner.py','light_adapter.py','identity_gate.py','input_preflight.py','e1_reference_gate.py','real_worker.py','timing_launch.py')]
 source += [BASE/'e4/skip_deepstack.py',BASE/'monitor/supervise.py',BASE/'initial_plan.json',BASE/'e5/graph_ttft_pilot_v3/plan001.json']
 source += [ROOT/'reproduction/20261028/qwen'/x for x in ('gpu_formula_facility_panel_run02/run_plan.json','gpu_formula_facility_panel_run02/run_panel.py','qwen_weight_identity.json')]
 source += [ROOT/'_reproduction/author_eadp/Qwen_vl/model/pruner.py',ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py']
 source += list((BASE/'e6/external_sources/pixelprune_8d347eb/pixelprune').rglob('*.py'))
 for sub in ('pixelprune_dev384','pixelprune_t045_dev384'):
  source += [BASE/'e6'/sub/'plan001.json',BASE/'e6'/sub/'worker.py']
 for sub in ('pixelprune_dev384_accept','pixelprune_t045_accept'):
  source += [BASE/'e6'/sub/'verification001.json',BASE/'e6'/sub/'monitor_verify001/started.json',BASE/'e6'/sub/'monitor_verify001/terminal.json']
 for sub in ('pixel_dev384','pixel_t045_dev384'):
  source += [BASE/'e1_independent_review'/(sub+'_result_review.json'),BASE/'e1_independent_review'/('monitor_'+sub+'_result')/'started.json',BASE/'e1_independent_review'/('monitor_'+sub+'_result')/'terminal.json']
 source += [BASE/'e1/e1full001_completion.json',BASE/'e1/monitorverifye1full001/started.json',BASE/'e1/monitorverifye1full001/terminal.json',BASE/'e1/input_preparation_002/inputs.jsonl']
 hashes={str(p.resolve()):sha(p) for p in source}
 if len(hashes)!=len(source):raise ValueError('duplicate source path')
 plan={'stage':'CPU and no-GPU preflight only; GPU execution locked','anchors':anchors,'jobs':make_schedule(anchors),
       'seed':20261105,'sha256':hashes,'budget_source_sha256':sha(BASE/'initial_plan.json'),
       'GT_opened':False,'predictions_opened_for_identity':True,
       'limitations':['no GPU authorization','measured Python K observer has nonzero unknown overhead',
          'natural full generated outputs must match same-arm accepted DEV outputs','no actual NVML/process memory proof']}
 with (HERE/'plan001.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
 print('sources',len(hashes),'plan_sha256',sha(HERE/'plan001.json'))
if __name__=='__main__':main()
