"""CPU-only independent re-execution of fixed one-image author selector thresholds."""
import argparse,json,math,sys
from pathlib import Path
import torch
B=Path(__file__).resolve().parent.parent;R=B/'e1_independent_review';D=B/'e6/pixelprune_threshold_probe';O=D/'probe001';sys.path[:0]=[str(D),str(B/'e6/external_sources/pixelprune_8d347eb')]
import worker
from pixelprune.core import compute_merged_keep_indices
from pixelprune.patches.qwen3_vl_hf import normalize_pixel_values_for_selector
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==R/'pixel_threshold_probe001_review.json'
def read(p):return json.loads(p.read_text())
s=read(D/'monitor_probe001/started.json');t=read(D/'monitor_probe001/terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==D/'worker.py'
plan=read(D/'plan001.json');worker.source_gate(plan);done=read(O/'completed.json');assert done['status']=='completed' and done['plan_sha256']==worker.sha(D/'plan001.json') and done['journal_sha256']==worker.sha(O/'journal.jsonl') and done['GT_opened'] is False and done['predictions_opened'] is False and done['gpu_used'] is False
journal=[json.loads(x) for x in (O/'journal.jsonl').read_text().splitlines()];assert len(journal)==12
capture=torch.load(worker.CAP,map_location='cpu',weights_only=True);pixels=capture['actual']['processor']['pixel_values'];grid=capture['actual']['processor']['image_grid_thw'];assert worker.metadata(pixels)==plan['processor']['pixel_values'] and worker.metadata(grid)==plan['processor']['image_grid_thw'];torch.set_num_threads(1);norm=normalize_pixel_values_for_selector(pixels);counts=[]
for i,threshold in enumerate(plan['thresholds']):
 assert journal[2*i]['event']=='started' and journal[2*i+1]['event']=='completed' and journal[2*i+1]['returncode']==0 and all(journal[j]['threshold']==threshold and journal[j]['index']==i for j in (2*i,2*i+1));record=read(O/('threshold_%02d.json'%i));ids=compute_merged_keep_indices(norm,grid,spatial_merge_size=2,method='pred_2d',metric='max',threshold=threshold,anchored=True)[0].tolist();assert ids==record['selected_indices'] and len(ids)==record['selected_count'] and ids==sorted(set(ids)) and record['threshold']==threshold and record['device']=='cpu' and record['cuda_initialized'] is False and record['GT_opened'] is False and record['predictions_opened'] is False and math.isfinite(record['wall_seconds']) and record['wall_seconds']>=0;counts.append({'threshold':threshold,'selected_count':len(ids)})
r={'status':'passed','independent_review':True,'images':1,'thresholds':counts,'plan_sha256':worker.sha(D/'plan001.json'),'journal_sha256':worker.sha(O/'journal.jsonl'),'result_sha256':{str(x):worker.sha(x) for x in sorted(O.glob('threshold_*.json'))},'run_started_sha256':worker.sha(D/'monitor_probe001/started.json'),'run_terminal_sha256':worker.sha(D/'monitor_probe001/terminal.json'),'reviewer_sha256':worker.sha(__file__),'GT_opened':False,'predictions_opened':False,'gpu_used':False,'limitations':['independent rerun uses the same author implementation on CPU','one chosen image; no claim of monotonicity, GPU parity, accuracy, or latency'], 'errors':[]};a.out.write_text(json.dumps(r,indent=2)+'\n');print('passed')
