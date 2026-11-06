"""Independently reproduce 32-image CPU-only author threshold calibration."""
import argparse,collections,inspect,json,math,statistics,sys
from pathlib import Path
import torch
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixelprune_threshold32';R=B/'e1_independent_review';O=D/'run001';PIX=B/'e6/external_sources/pixelprune_8d347eb';sys.path[:0]=[str(D),str(PIX)];import worker
from pixelprune.core import compute_merged_keep_indices
from pixelprune.patches.qwen3_vl_hf import normalize_pixel_values_for_selector
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==R/'pixel_threshold32_review.json'
def read(p):return json.loads(p.read_text())
s=read(D/'monitor_run001/started.json');t=read(D/'monitor_run001/terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==D/'worker.py' and s['timeout_seconds']==1800
plan=read(D/'plan001.json');worker.validate_plan(plan);done=read(O/'completed.json');assert done['status']=='completed' and done['plan_sha256']==worker.sha(D/'plan001.json') and done['journal_sha256']==worker.sha(O/'journal.jsonl') and done['GT_opened'] is False and done['predictions_opened'] is False and done['gpu_used'] is False
assert Path(inspect.getfile(compute_merged_keep_indices)).resolve()==PIX/'pixelprune/core.py' and Path(inspect.getfile(normalize_pixel_values_for_selector)).resolve()==PIX/'pixelprune/patches/qwen3_vl_hf.py'
outer=[json.loads(x) for x in (O/'journal.jsonl').read_text().splitlines()];assert len(outer)==64 and collections.Counter(x['identity']['task'] for x in plan['samples'])=={'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10};torch.set_num_threads(1);rows=[];files={}
for i,sample in enumerate(plan['samples']):
 assert outer[2*i]['event']=='started' and outer[2*i]['image_index']==i and outer[2*i]['sample']==sample['identity'] and outer[2*i+1]['image_index']==i and outer[2*i+1]['event']=='completed' and outer[2*i+1]['returncode']==0
 path=Path(sample['capture_path']);assert worker.sha(path)==sample['capture_sha256'];capture=torch.load(path,map_location='cpu',weights_only=True);pixel=capture['actual']['processor']['pixel_values'];grid=capture['actual']['processor']['image_grid_thw'];assert worker.meta(pixel)==sample['processor']['pixel_values'] and worker.meta(grid)==sample['processor']['image_grid_thw'];norm=normalize_pixel_values_for_selector(pixel)
 sub=O/('image_%03d'%i);complete=read(sub/'complete.json');assert complete['status']=='completed' and complete['thresholds']==10 and complete['sample']==sample['identity'] and complete['cuda_initialized'] is False and complete['GT_opened'] is False and complete['predictions_opened'] is False
 journal=[json.loads(x) for x in (sub/'journal.jsonl').read_text().splitlines()];assert len(journal)==20;files[str(sub/'complete.json')]=worker.sha(sub/'complete.json');files[str(sub/'journal.jsonl')]=worker.sha(sub/'journal.jsonl')
 for j,threshold in enumerate(plan['thresholds']):
  x,y=journal[2*j:2*j+2];assert x['event']=='started' and y['event']=='completed' and x['threshold_index']==y['threshold_index']==j and x['threshold']==y['threshold']==threshold and y['selector_path']==('fast' if threshold==0 else 'anchored_python') and math.isfinite(y['wall_seconds']) and y['wall_seconds']>=0
  ids=compute_merged_keep_indices(norm,grid,spatial_merge_size=2,method='pred_2d',metric='max',threshold=threshold,anchored=True)[0].tolist();assert ids==y['selected_indices'] and len(ids)==y['selected_count'] and ids==sorted(set(ids)) and all(0<=x<1024 for x in ids);rows.append({'task':sample['identity']['task'],'qid':str(sample['identity']['qid']),'group_id':sample['group_id'],'threshold':threshold,'K':len(ids)})
assert len(rows)==320
summary={}
for threshold in plan['thresholds']:
 sub=[x for x in rows if x['threshold']==threshold];values=sorted(x['K'] for x in sub);tasks={}
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  v=sorted(x['K'] for x in sub if x['task']==task);tasks[task]={'n':len(v),'min':v[0],'median':statistics.median(v),'max':v[-1]}
 summary[str(threshold)]={'n':len(values),'min':values[0],'median':statistics.median(values),'max':values[-1],'within_224_288':sum(224<=x<=288 for x in values),'tasks':tasks}
nonmonotonic=[{'task':s['identity']['task'],'qid':str(s['identity']['qid'])} for s in plan['samples'] if any(a['K']<b['K'] for a,b in zip([x for x in rows if x['group_id']==s['group_id']],[x for x in rows if x['group_id']==s['group_id']][1:]))]
r={'status':'passed','independent_review':True,'images':32,'threshold_cases':320,'task_counts':{'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10},'summary':summary,'nonmonotonic_images':nonmonotonic,'per_image':rows,'plan_sha256':worker.sha(D/'plan001.json'),'source_sha256':plan['source_sha256'],'run_started_sha256':worker.sha(D/'monitor_run001/started.json'),'run_terminal_sha256':worker.sha(D/'monitor_run001/terminal.json'),'run_journal_sha256':worker.sha(O/'journal.jsonl'),'image_files_sha256':files,'reviewer_sha256':worker.sha(__file__),'GT_opened':False,'predictions_opened':False,'gpu_used':False,'limitations':['same author selector independently rerun on CPU only; GPU selection identity and anchored GPU runtime untested','32-image sample and threshold grid were fixed after DEV results/one-image probe; any threshold choice and subsequent quality are exploratory','no quality or end-to-end timing measured'], 'errors':[]};a.out.write_text(json.dumps(r,indent=2)+'\n');print('passed')
