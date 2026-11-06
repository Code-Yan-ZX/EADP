"""Independent no-label M3 single-image GPU gradient evidence review."""
import argparse,hashlib,json,math,pathlib,subprocess
import numpy as np,torch
B=pathlib.Path(__file__).resolve().parents[1];D=B/'m3_gpu_smoke';R=D/'smoke_001';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(pathlib.Path(p).read_text())
p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();assert not a.out.exists()
proof=D/'smoke_001_verification.json';fs=D/'monitor_verify001/started.json';ft=D/'monitor_verify001/terminal.json';v=read(proof);st=read(fs);te=read(ft)
assert v['status']=='passed' and v['stage']=='synthetic_target_gradient_smoke' and v['gradients']==4 and v['target_ids']==[100,200,300] and not v['GT_opened'] and not v['scoring_performed'] and not v['generation_performed']
assert te['reason']=='completed' and te['worker_returncode']==0 and not te['remaining_process_group'] and st['worker_pid']==te['worker_pid'] and any(pathlib.Path(x).name=='verify.py' for x in st['command']) and (pathlib.Path(st['cwd'])/st['command'][st['command'].index('--out')+1]).resolve()==proof.resolve()
for path,h in v['files_sha256'].items():assert sha(path)==h
manifest=read(D/'smoke_001_sources/manifest.json');cpu=read(D/'cpu_check_006/verification.json');assert set(cpu['source_sha256_after']).issubset(manifest)
for source,entry in manifest.items():assert sha(source)==entry['sha256']==sha(entry['snapshot'])
assert v['verifier_sha256']==manifest[str(D/'verify.py')]['sha256']
r=read(R/'result.json');f=read(R/'frozen.json');runstart=read(D/'monitor_smoke001/started.json');runend=read(D/'monitor_smoke001/terminal.json');exit=read(R/'exit.json');plan=read(D/'plan.json')
assert r['target_ids']==f['target_ids']==plan['target_token_ids']==[100,200,300] and not r['gt_read'] and not r['scoring'] and not r['generation'] and r['model_fully_frozen'] and r['model_eval'] and r['gradient_autograd_enabled'] and math.isfinite(r['synthetic_loss'])
assert runend['reason']=='completed' and runend['worker_returncode']==0 and not runend['remaining_process_group'] and runstart['worker_pid']==runend['worker_pid'] and exit['controller']['pid']==runend['worker_pid']
assert exit['returncode']==exit['container_exit_code']==0 and exit['container_stopped_verified'] and not exit['timed_out'] and exit['interrupted'] is None and exit['gpu_pids_after']==[]
i=json.loads(subprocess.check_output(['docker','inspect',exit['container_identity']['container_id']]))[0];assert not i['State']['Running'] and i['State']['Pid']==0 and i['State']['ExitCode']==0 and not i['State']['OOMKilled']
events=[json.loads(x) for x in (R/'events.jsonl').read_text().splitlines()];assert [x['event'] for x in events]==['worker_started','model_loaded','processing_started','native_prefill_started','backward_started','gradients_saved','worker_complete'] and all(events[i]['unix']<=events[i+1]['unix'] for i in range(6))
refs=[json.loads(x) for x in (B/'e4_native/native_001/records.jsonl').read_text().splitlines()];matches=[x for x in refs if x['task']==r['task'] and str(x['qid'])==str(r['qid']) and x['group_id']==r['group_id'] and x['path']=='stock_native'];native=next(x for x in matches if not x['forced']);forced=next(x for x in matches if x['forced'])
assert r['processor']==native['processor'] and r['features']==[native['main_visual'],*native['full_deepstack']] and r['native_reference_logits']==native['prefill_logits']
for k in range(2):
 pos=r['prompt_length']+r['native_rope_delta']+k;assert [v[0][k] for v in r['teacher_prefix_positions']]==[pos]*3 and forced['decode_trace'][k+1]['position_ids']==[[[pos]]]*3
path=pathlib.Path(r['arrays']['path']);assert path.resolve()==(R/'arrays.npz').resolve() and sha(path)==r['arrays']['sha256'];stats=[]
with np.load(path,allow_pickle=False) as z, np.load(native['arrays']['path'],allow_pickle=False) as nz:
 assert set(z.files)=={*('feature'+str(i) for i in range(4)),*('gradient'+str(i) for i in range(4)),'native_actual_logits','native_reference_logits','teacher_first_logits'}
 assert np.array_equal(z['native_reference_logits'],nz['prefill_logits'])
 tensors={}
 for name in z.files:
  meta=r['features'][int(name[-1])] if name.startswith('feature') else r['gradients'][int(name[-1])] if name.startswith('gradient') else r[name]
  assert z[name].dtype==np.uint8 and hashlib.sha256(z[name].tobytes()).hexdigest()==meta['sha256'];dtype={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32}[meta['dtype']];tensor=torch.from_numpy(z[name].copy()).view(dtype).reshape(meta['shape']);assert torch.isfinite(tensor).all();tensors[name]=tensor
 for i in range(4):
  feature=tensors['feature'+str(i)];grad=tensors['gradient'+str(i)];assert tuple(feature.shape)==tuple(grad.shape)==(1024,4096) and int(torch.count_nonzero(grad))==r['gradient_nonzero_counts'][i]>0
  u=(feature.float()*grad.float()).abs().mean(-1);assert r['utility'][i]['length']==1024 and math.isclose(float(u.min()),r['utility'][i]['min'],rel_tol=1e-5,abs_tol=1e-12) and math.isclose(float(u.max()),r['utility'][i]['max'],rel_tol=1e-5,abs_tol=1e-12);stats.append({'nonzero':r['gradient_nonzero_counts'][i],'utility_min':float(u.min()),'utility_max':float(u.max())})
 reference=tensors['native_reference_logits'];actual=tensors['native_actual_logits'];teacher=tensors['teacher_first_logits'];native_max=float((actual.float()-reference.float()).abs().max());teacher_max=float((teacher.float()-reference.float()).abs().max());limit=plan['native_first_logits_max_abs_tolerance'];assert native_max==v['native_first_logits_comparison']['max_abs'] and teacher_max==v['teacher_first_logits_comparison']['max_abs'] and teacher_max<=limit and int(actual.argmax())==int(reference.argmax())==int(teacher.argmax())
result={'status':'passed','independent_review':True,'scope':'one DEV image, three fixed synthetic tokens; gradient feasibility only','formal_proof_sha256':sha(proof),'formal_started_sha256':sha(fs),'formal_terminal_sha256':sha(ft),'source_manifest_sha256':sha(D/'smoke_001_sources/manifest.json'),'reviewer_sha256':sha(__file__),'arrays_sha256':sha(path),'gradient_streams':stats,'native_first_logits_max_abs':native_max,'teacher_first_logits_max_abs':teacher_max,'frozen_tolerance':limit,'teacher_at_tolerance_boundary':teacher_max==limit,'GT_opened':False,'scoring_performed':False,'generation_performed':False,'limitations':['synthetic target, no oracle accuracy or deployment claim','teacher logits maximum difference equals the frozen threshold; no stronger equality is claimed','KV contents were not independently reconstructed']}
a.out.write_text(json.dumps(result,indent=2)+'\n');print('passed',sha(a.out))
