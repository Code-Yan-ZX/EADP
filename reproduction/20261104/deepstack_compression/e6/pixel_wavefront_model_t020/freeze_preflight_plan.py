"""Freeze reviewed sources only after synthetic CPU validation has exited."""
import argparse,json,os
from pathlib import Path
from source_inventory import sha,source_hashes
from reference_contract import reference
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';S=B/'e6/pixelprune_t020_smoke'
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=D/'preflight_plan002.json':raise ValueError('fixed plan output')
 cpu=D/'cpu_check004/verification.json';s=json.loads((D/'monitor_cpu004/started.json').read_text());t=json.loads((D/'monitor_cpu004/terminal.json').read_text());c=json.loads(cpu.read_text());sources=source_hashes()
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or t.get('worker_pid')!=s.get('worker_pid') or c.get('status')!='passed' or c.get('existing_DEV_predictions_opened') is not True or c.get('source_sha256_before')!=sources or c.get('source_sha256_after')!=sources:raise ValueError('CPU proof incomplete')
 refs=reference(I/'pixel_t020_result_review.json',I/'review_pixel_t020_result.py',I/'monitor_pixel_t020_result/started.json',I/'monitor_pixel_t020_result/terminal.json',S/'run_smoke001/observations')
 component=json.loads((I/'pixel_wavefront_t020_gpu32_result_review.json').read_text());rs=json.loads((I/'monitor_pixel_wavefront_t020_gpu32_result/started.json').read_text());rt=json.loads((I/'monitor_pixel_wavefront_t020_gpu32_result/terminal.json').read_text())
 if component.get('status')!='passed' or component.get('independent_review') is not True or component.get('ordered_equal')!=32 or component.get('images')!=32 or component.get('reviewer_sha256')!=sha(I/'review_pixel_wavefront_t020_gpu32_result.py') or rt.get('reason')!='completed' or rt.get('worker_returncode')!=0 or rt.get('remaining_process_group')!=[] or rs.get('worker_pid')!=rt.get('worker_pid'):raise ValueError('component identity proof incomplete')
 # Bind the actual three reviewed component records to the same t=.2 model anchors.
 complete=B/'e6/pixel_wavefront_t020_gpu32/run_smoke001/observations/complete.json'
 if component['evidence_sha256'].get(str(complete))!=sha(complete):raise ValueError('component records unbound')
 cases=json.loads(complete.read_text())['cases']
 for entry in refs['images']:
  anchor=entry['anchor'];expected=entry['expected']['selected_indices'];key=anchor['task']+'/'+str(anchor['qid'])
  matches=[x for x in cases if x.get('key')==key]
  if len(matches)!=1 or matches[0].get('status')!='completed' or len(matches[0].get('arms',[]))!=2 or any(x.get('indices')!=expected for x in matches[0]['arms']):raise ValueError('component/model smoke anchor mismatch')
 old=json.loads((S/'plan001.json').read_text());aliases={path:digest for path,digest in old['sha256'].items() if path.startswith('/home/dell/.cache/huggingface/')}
 if len(aliases)!=7 or any(sources.get(path)!=digest or sha(path)!=digest for path,digest in aliases.items()):raise ValueError('seven model metadata mount aliases and content must match')
 plan={'stage':'three-anchor wavefront model no-GPU preflight only','threshold':.2,'images':refs['images'],'source_sha256':sources,'model_metadata_aliases_sha256':aliases,'cpu_sha256':sha(cpu),'cpu_started_sha256':sha(D/'monitor_cpu004/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu004/terminal.json'),'component_review_sha256':sha(I/'pixel_wavefront_t020_gpu32_result_review.json'),'budget_source_sha256':sha(B/'initial_plan.json'),'existing_DEV_predictions_opened':True,'GT_opened':False,'scoring_performed':False,'gpu_authorized':False,'limitations':['preflight only; full model identity worker, GPU gate, and separately reviewed timing remain required','existing DEV references and component results informed this post-hoc three-image choice','seven model metadata paths use preserved absolute container mount aliases, with content SHA checked']}
 with a.out.open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
