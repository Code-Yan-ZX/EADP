"""Post-hoc nine-process separate NVML design; no GPU or prediction loading."""
import hashlib,json,random
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];T=B/'e6/pixel_wavefront_t020_model_timing_v2/design001.json';I=B/'e1_independent_review'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def make(anchors):
 if len(anchors)!=3 or len({(a['task'],str(a['qid']),a['group_id']) for a in anchors})!=3 or len({a['task'] for a in anchors})!=3:raise ValueError('three unique task anchors')
 jobs=[dict(a,arm=arm) for a in anchors for arm in ('t020_author','t020_wavefront','t0_author')]
 random.Random(20261105).shuffle(jobs)
 for i,j in enumerate(jobs):j['order']=i
 return jobs

def plan():
 anchors=json.loads(T.read_text())['anchors'];jobs=make(anchors)
 proof_files=[B/'e6/pixel_wavefront_t020_accept_v2/verification001.json',I/'pixel_wavefront_t020_run_v2_result_review.json',I/'monitor_pixel_wavefront_t020_run_v2_result/started.json',I/'monitor_pixel_wavefront_t020_run_v2_result/terminal.json',B/'e6/pixelprune_dev384_accept/verification001.json',I/'pixel_dev384_result_review.json',I/'monitor_pixel_dev384_result/started.json',I/'monitor_pixel_dev384_result/terminal.json']
 return {'scope':'post-hoc independent nine-process model NVML design, not an executable admission','design_sha256':sha(T),'seed':20261105,'jobs':jobs,'prior_proof_sha256':{str(p):sha(p) for p in proof_files},'separate_from_ttft':True,'gpu_authorized':False,'GT_opened':False,'predictions_opened':False,'limitations':['proposed nine fresh processes; not a model or GPU run','sample query windows bracket load, request and teardown; unknown is not zero, MiB granularity and sampled peak can miss transients','target process and whole device are distinct; graphics background retained and foreign compute rejected','actual processor, generated IDs, EOS and K must match the accepted same-arm same-anchor reference','t=.2 prior identity provides complete ordered selection; t0 full DEV prior records K and diagnostics but not necessarily raw complete ordered selection, so archive actual indices without asserting retrospectively proved identical ordering','capture and CPU synchronization can affect memory; no TTFT is measured in this run','three post-hoc anchors and dynamic K; not equal compute to fixed K or to one another','final executable worker, gate, container preflight, bounded cleanup, source and formal-verifier freeze and independent reviews are required']}
if __name__=='__main__':raise SystemExit('design only; GPU locked')
