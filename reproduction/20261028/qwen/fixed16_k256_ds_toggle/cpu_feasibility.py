"""Read only: audit historical cache geometry and complete FA2 K256 predictions."""
import hashlib,json
from pathlib import Path
import numpy as np
Q=Path(__file__).resolve().parent.parent
F=Q/'gpu_formula_facility_panel_run02'; P=Q/'fixed16_ds_factorial'; A=Q/'independent_audit'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 from contracts import compare_selection
 panel=json.loads((F/'run_plan.json').read_text())['panel']
 records=[json.loads(s) for s in (F/'execution/per_question.jsonl').open()]
 predictions=[json.loads(s) for s in (A/'public_fa2_full_run02_DocVQA_VAL.per_question.jsonl').open()]
 report=json.loads((A/'public_fa2_full_run02.terminal_audit.report.json').read_text())
 assert report['audit_terminated'] and report['coverage_complete'] and report['generation_valid']
 assert report['public_launcher_exited_successfully'] and len(panel)==len(records)==16 and len(predictions)==5349
 for i,(item,record) in enumerate(zip(panel,records)):
  assert item['position']==record['position'] and item['index']==record['index']
  row=predictions[item['position']];assert row['index']==item['index'] and isinstance(row['raw_prediction'],str)
  assert not row['prediction_changed_by_native_na_loader']
  cache=F/'execution'/('question_%02d.npz'%i);assert sha(cache)==record['cache']['sha256']
  fixed=P/'execution'/('%02d_eadp1024.npz'%i);fixed_meta=json.loads((P/'execution'/('%02d_eadp1024.json'%i)).read_text());assert sha(fixed)==fixed_meta['archive_sha256']
  with np.load(cache,allow_pickle=False) as a,np.load(fixed,allow_pickle=False) as b:
   ordered=a['native_ordered_ids'][0].tolist();spatial=a['native_spatial_ids'].tolist();compare_selection(ordered,spatial)
   assert np.array_equal(a['visual_features_BF16_bits'],b['visual_features'])
   assert np.array_equal(a['native_pruned_BF16_bits'],b['visual_features'][spatial])
   assert np.array_equal(a['full_prompt_input_ids'],b['processor_input_ids'])
   assert all(b['visual_deepstack_'+str(j)].shape==(1024,4096) and b['visual_deepstack_'+str(j)].dtype==np.uint16 for j in range(3))
 print(json.dumps(dict(success=True,n=16,author_selected_rows_valid=16,full_visual_bytes_match=16,selected_visual_bytes_match=16,processor_ids_match=16,three_ds_shapes_match=16,historical_raw_texts_exist=16,prior_k256_DSinjection_measured=False,model_loaded=False,GPU_started=False)))
if __name__=='__main__':main()
