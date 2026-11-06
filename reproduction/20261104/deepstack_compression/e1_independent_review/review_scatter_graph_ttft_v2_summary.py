"""Independently check graph TTFT pilot per-anchor and aggregate arithmetic."""
import argparse,hashlib,json,math,os,statistics
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e5/scatter_graph_ttft_v2';R=B/'e1_independent_review'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):return math.isclose(a,b,rel_tol=1e-13,abs_tol=1e-12)
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==R/'scatter_graph_ttft_v2_summary_review.json'
mon=R/'monitor_scatter_graph_ttft_v2_result002';s=read(mon/'started.json');t=read(mon/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];prior=read(R/'scatter_graph_ttft_v2_result_review.json');f=read(D/'smoke001_verification.json');assert prior['status']=='passed' and prior['independent_review'] is True and prior['formal_sha256']==sha(D/'smoke001_verification.json') and prior['unequal_comparisons']==f['unequal_comparisons']
per=f['per_anchor'];assert len(per)==3
for v in per.values():
 o=v['original'];g=v['graph'];assert len(o['raw_ms'])==len(g['raw_ms'])==5 and o['median_ms']==statistics.median(o['raw_ms']) and g['median_ms']==statistics.median(g['raw_ms']) and close(v['relative_change_pct'],(g['median_ms']/o['median_ms']-1)*100)
agg=f['aggregate_only_if_all_pairs_equal']
if f['unequal_comparisons']==0:
 o=statistics.median(x['original']['median_ms'] for x in per.values());g=statistics.median(x['graph']['median_ms'] for x in per.values());assert agg['original_median_of_anchor_medians_ms']==o and agg['graph_median_of_anchor_medians_ms']==g and close(agg['relative_change_of_medians_pct'],(g/o-1)*100)
else:assert not agg
z=read(D/'run_smoke001/observations/graph_setup.json');setup=f['graph_setup'];assert setup['capture_wall_ms']==z['capture_wall_ms'] and setup['allocated_before']==z['before_bytes']['allocated'] and setup['allocated_after']==z['after_bytes']['allocated'] and setup['allocated_delta']==z['after_bytes']['allocated']-z['before_bytes']['allocated'] and setup['reserved_delta']==z['after_bytes']['reserved']-z['before_bytes']['reserved']
out={'status':'passed','independent_review':True,'formal_sha256':sha(D/'smoke001_verification.json'),'prior_review_sha256':sha(R/'scatter_graph_ttft_v2_result_review.json'),'prior_terminal_sha256':sha(mon/'terminal.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'limitations':['three exploratory anchors; unmeasured background influence unknown','capture excluded from first-token interval; setup and allocator reported separately','per-anchor repeats do not establish full-panel efficiency'],'errors':[]}
with a.out.open('x') as h:json.dump(out,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
