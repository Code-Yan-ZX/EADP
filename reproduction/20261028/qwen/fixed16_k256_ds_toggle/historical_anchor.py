"""K256 historical anchor; call only after the new raw response is preserved.

Old Qwen feature cache did not generate text. Historical full-run literal output
has no saved per-item output token IDs. This anchor cannot prove they agree.
"""
import hashlib,json
from pathlib import Path
import numpy as np
Q=Path(__file__).resolve().parent.parent
FORMULA=Q/'gpu_formula_facility_panel_run02';FULL16=Q/'fixed16_ds_factorial';AUDIT=Q/'independent_audit'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def bits(torch,t):
 x=t.detach().cpu().contiguous()
 return x.view(torch.uint16).numpy() if x.dtype==torch.bfloat16 else x.numpy()
def validate(torch,panel_index,item,rec):
 plan=json.loads((FORMULA/'run_plan.json').read_text());panel=plan['panel'];expected=panel[panel_index]
 assert item['position']==expected['position'] and item['index']==expected['index']
 proof=json.loads((FORMULA/'root_terminal_process_proof.json').read_text());summary_path=FORMULA/'execution/summary.json'
 assert proof['root_owned_exec_actual_exit_code']==0 and proof['all_original_proc_paths_absent'] and proof['summary_sha256']==sha(summary_path)
 summary=json.loads(summary_path.read_text());records_path=FORMULA/'execution/per_question.jsonl'
 assert summary['status']=='complete' and summary['per_question_sha256']==sha(records_path) and summary['run_plan_sha256']==sha(FORMULA/'run_plan.json')==proof['run_plan_sha256']
 records=[json.loads(s) for s in records_path.open()];record=records[panel_index]
 cache=FORMULA/'execution'/('question_%02d.npz'%panel_index);assert record['cache']['sha256']==sha(cache)
 prior=FULL16/'execution'/('%02d_eadp1024.npz'%panel_index);prior_record=json.loads((FULL16/'execution'/('%02d_eadp1024.json'%panel_index)).read_text());assert prior_record['archive_sha256']==sha(prior)
 audit=json.loads((AUDIT/'public_fa2_full_run02.terminal_audit.report.json').read_text());reference=AUDIT/'public_fa2_full_run02_DocVQA_VAL.per_question.jsonl'
 assert audit['audit_terminated'] and audit['coverage_complete'] and audit['generation_valid'] and audit['public_launcher_exited_successfully']
 assert audit['existing_CPU_verification']['datasets']['DocVQA_VAL']['per_question']['sha256']==sha(reference)
 rows=[json.loads(s) for s in reference.open()]
 pred=rows[item['position']];assert pred['index']==item['index'] and not pred['prediction_changed_by_native_na_loader']
 assert rec['native_failure'] is None and rec['response']==pred['raw_prediction']
 assert rec['selection_calls']==1 and not rec['selection_errors']
 selected=rec['selection_results'][0]['ordered_ids'];assert rec['selection_results'][0]['quotas'].detach().cpu().tolist()==[256]
 spatial=selected[0].sort().values
 with np.load(cache,allow_pickle=False) as a,np.load(prior,allow_pickle=False) as b:
  checks=[(selected,a['native_ordered_ids']),(spatial,a['native_spatial_ids']),(rec['pruner_input'],a['visual_features_BF16_bits']),
    (rec['pruner_output'],a['native_pruned_BF16_bits']),(rec['processor_inputs']['input_ids'],a['full_prompt_input_ids']),
    (rec['processor_inputs']['image_grid_thw'],a['image_grid_thw'])]
  checks.extend((rec['processor_inputs'][k],b['processor_'+k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw'))
  checks.extend((rec['vision'][1][j],b['visual_deepstack_'+str(j)]) for j in range(3))
  checks.append((rec['vision'][0],b['visual_features']))
  expected_types={np.dtype('uint16'):torch.bfloat16,np.dtype('int64'):torch.int64,np.dtype('int32'):torch.int32,np.dtype('float32'):torch.float32,np.dtype('bool'):torch.bool}
  assert all(y.dtype in expected_types and x.dtype==expected_types[y.dtype] and np.array_equal(bits(torch,x),y) for x,y in checks)
 return dict(historical_literal_answer_equal=True,old_native_selection_and_feature_equal=True,old_full_visual_and_three_deepstack_equal=True,
  old_generation_token_ids_unavailable=True,old_generation_prefill_unavailable=True)
