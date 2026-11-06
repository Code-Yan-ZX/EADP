"""Freeze GPU and formal plans only after independent no-device import review."""
import json,os,sys
from pathlib import Path
T=Path(__file__).resolve().parent;E=T.parent;B=E.parents[1];F=B/'e6/rres_k256_dev384';I=B/'e1_independent_review';sys.path.insert(0,str(F));import pilot_gate
read=pilot_gate.read;sha=pilot_gate.sha;terminal=pilot_gate.terminal
prior=pilot_gate.require();runtime=read(E/'runtime_preflight_plan001.json');proof=I/'rres_k256_dev384_runtime_result_review.json';reviewer=I/'review_rres_k256_dev384_runtime_result.py';monitor=I/'monitor_rres_k256_dev384_runtime_result';terminal(monitor,reviewer,proof);terminal(E/'monitor_runtime_preflight001',E/'runtime_preflight_launch.py',E/'runtime_preflight001');q=read(proof)
fields={'plan_sha256':E/'runtime_preflight_plan001.json','worker_preflight_sha256':E/'runtime_preflight001/worker_preflight.json','launch_sha256':E/'runtime_preflight001/launch.json','identity_sha256':E/'runtime_preflight001/container_identity.json','exit_sha256':E/'runtime_preflight001/exit.json','started_sha256':E/'monitor_runtime_preflight001/started.json','terminal_sha256':E/'monitor_runtime_preflight001/terminal.json','reviewer_sha256':reviewer}
if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('errors')!=[] or any(q.get(k)!=sha(p) for k,p in fields.items()):raise ValueError('runtime noGPU independent proof')
source=runtime['source_sha256'].copy()
for p in (*fields.values(),proof,monitor/'started.json',monitor/'terminal.json',Path(__file__)):
 source[str(p.resolve())]=sha(p)
pilot=read(B/'e6/rres_k256_execution_v2/verification001.json')
if pilot.get('status')!='passed' or len(pilot['results'])!=3:raise ValueError('three-anchor formal')
for item in pilot['results']:
 name=item['task']+'_'+str(item['qid']);root=B/'e6/rres_k256_execution_v2/run001/observations'
 for p in (root/(name+'.json'),root/(name+'_generated.json')):source[str(p.resolve())]=sha(p)
if any(sha(k)!=v for k,v in source.items()):raise ValueError('source drift')
score=I/'rres_k256_full_score_prelaunch_review.json';pre=I/'rres_k256_dev384_preflight_result_review.json';initial=read(B/'initial_plan.json')
gpu={'scope':'rres384_gpu_generation','images':384,'resolution':512,'visual_tokens':256,'source_sha256':source,'pilot_review_sha256':prior['review_sha256'],'preflight_result_sha256':sha(pre),'runtime_result_sha256':sha(proof),'score_review_sha256':sha(score),'budget_end_unix':initial['budget_end_unix'],'budget_source_sha256':sha(B/'initial_plan.json'),'inner_timeout_seconds':21000,'outer_timeout_seconds':21600,'grace_seconds':300,'GT_opened':False,'scoring_performed':False,'GPU_used_at_freeze':False,'existing_DEV_predictions_opened':True,'prediction_scope':'accepted E1/other DEV records parsed or source-hashed; three accepted low-res pilot outputs hashed for later anchor comparison; no new predictions at freeze','limitations':['post-hoc DEV cohort, not confirmation','512 resize changes visual encoder computation; K256 is not equal compute','online DS/position/KV checks and hashes do not archive full high-dimensional tensors','source snapshots are post-run and endpoint weight stats do not exclude transient changes']}
gp=F/'gpu_plan001.json'
with gp.open('x') as f:json.dump(gpu,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
formal={'scope':'rres384_formal_noGT','gpu_plan_sha256':sha(gp),'verifier_sha256':sha(E/'verify.py'),'cpu_proof_sha256':sha(E/'cpu_proof001.json'),'GT_opened':False,'scoring_performed':False,'GPU_used_at_freeze':False,'contract':'all 384 identities and raw outputs, exact processor, DS/mRoPE/KV, natural EOS/truncation, real container and source snapshot; report all pilot anchor differences without exclusion'}
with (F/'verification_plan001.json').open('x') as f:json.dump(formal,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
