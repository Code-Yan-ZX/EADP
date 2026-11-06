"""Posthoc 128-image selector timing summary; no GPU or GT access."""
import argparse,json,os,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent.parent;P=BASE/'e5/selector_real32';N=BASE/'e5/selector_real128';sys.path[:0]=[str(BASE/'e5/selector_real32_failure'),str(BASE/'e4_c_latency')]
from timing_summary import summarize
from identity_gate import sha,supervised

def distribution(v):return {'n':len(v),'median_ms':statistics.median(v),'min_ms':min(v),'max_ms':max(v)} if v else {'n':0}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();combined=HERE/'combined001.json';supervised(HERE/'monitor_combine001/started.json',HERE/'monitor_combine001/terminal.json',HERE/'combine_run.py',combined);proof=json.loads(combined.read_text())
 if proof.get('status')!='completed' or proof.get('GT_opened') is not False or proof.get('scoring_performed') is not False:raise ValueError('combined evidence incomplete')
 outputs=[]
 for folder,count,field in ((P,32,'prior_evidence'),(N,96,'expansion_evidence')):
  journal=folder/'run_001/journal.jsonl'
  if sha(journal)!=proof[field]['journal_sha256']:raise ValueError('journal changed')
  events=[json.loads(x) for x in journal.read_text().splitlines() if x.strip()];outputs.append(summarize(events,image_count=count))
 report={}
 for budget in ('128','256','512'):
  per={f'{group}:{i}':row for group,result in zip(('observed_pilot','added'),outputs) for i,row in result[budget]['per_image'].items()};paired=[v for v in per.values() if all(v[m] is not None for m in ('author','fused','graph'))]
  all_avail={m:distribution([v[m] for v in per.values() if v[m] is not None]) for m in ('author','fused','graph')};paired_dist={m:distribution([v[m] for v in paired]) for m in ('author','fused','graph')};missing=[k for k,v in per.items() if any(v[m] is None for m in ('author','fused','graph'))]
  if len(per)!=128 or len(paired)+len(missing)!=128:raise ValueError('missing planned observations')
  counts=proof['counts']['combined_128'][budget]
  for method in ('fused','graph'):
   if len(missing)!=counts[method]['strict_order_fail']+counts[method]['unavailable']:raise ValueError('timing omission differs from strict identity failure')
  setup={}
  for group,result in zip(('observed_pilot','added'),outputs):
   s=result[budget]['graph_setup'];setup[group]={'capture_wall_ms':s['capture_wall_ms'],'allocated_before_bytes':s['before_bytes']['allocated'],'allocated_after_bytes':s['after_bytes']['allocated'],'allocated_delta_bytes':s['after_bytes']['allocated']-s['before_bytes']['allocated'],'reserved_before_bytes':s['before_bytes']['reserved'],'reserved_after_bytes':s['after_bytes']['reserved']}
  report[budget]={'planned_images':128,'all_available':all_avail,'paired':paired_dist,'excluded_from_paired':missing,'per_image_medians_ms':per,'graph_setup':setup}
 out={'status':'completed','scope':'posthoc component CUDA-event timing of fixed 32 observed pilot plus 96 added DEV images','results':report,'combined_sha256':sha(combined),'source_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'limitations':['Each image contributes its median of five calls; distributions report across-image medians and range.','K512 fails exact order on one pilot image; its fused and graph timings were not taken; candidate and paired summaries use 127/128, with possible selection bias.','96-image expansion was decided after seeing 32-image pilot; full 128 is not an independent confirmation.','Fused and graph share the same fused selector kernel; setup is separate and is not end-to-end TTFT.','During the measurement window lightweight code editing and background activity occurred with no per-sample interference windows recorded; their impact is unknown.','PyTorch allocated/reserved setup delta is not device-wide peak memory.']}
 with a.out.open('x') as h:json.dump(out,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
if __name__=='__main__':main()
