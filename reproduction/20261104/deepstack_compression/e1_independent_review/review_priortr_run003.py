"""Independent CPU review of completed PriorTR single-question run."""
import argparse,hashlib,json,pathlib,subprocess
import numpy as np,torch
from transformers import AutoTokenizer
B=pathlib.Path(__file__).resolve().parents[1];P=B/'e6/priortr/preparation';R=P/'smoke_001';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(pathlib.Path(p).read_text());rows=lambda p:[json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x]
a=argparse.ArgumentParser();a.add_argument('--out',type=pathlib.Path,required=True);args=a.parse_args();assert not args.out.exists()
formal=P/'smoke_001_verification.json';fs=P/'monitor_verify_smoke001/started.json';ft=P/'monitor_verify_smoke001/terminal.json';v=load(formal);s=load(fs);t=load(ft)
assert v['status']=='passed' and not v['GT_opened'] and not v['scoring_performed'] and t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid']
assert any(pathlib.Path(x).name=='smoke_verify.py' for x in s['command']) and (pathlib.Path(s['cwd'])/s['command'][s['command'].index('--out')+1]).resolve()==formal.resolve()
for p,h in v['files_sha256'].items():assert sha((pathlib.Path(s['cwd'])/p).resolve())==h
frozen=load(R/'frozen.json');plan=load(P/'smoke_single_plan002.json');ex=load(R/'exit.json');st=load(P/'monitor_smoke001/started.json');end=load(P/'monitor_smoke001/terminal.json');assert not frozen['gt_read'] and not frozen['scoring'] and end['reason']=='completed' and end['worker_returncode']==0 and not end['remaining_process_group'] and st['worker_pid']==end['worker_pid']
assert ex['returncode']==0 and ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['gpu_pids_after']==[] and not ex['timed_out'] and ex['interrupted'] is None
ins=json.loads(subprocess.check_output(['docker','inspect',ex['container_identity']['container_id']]))[0];assert not ins['State']['Running'] and ins['State']['Pid']==0 and ins['State']['ExitCode']==0 and not ins['State']['OOMKilled']
manifest=load(P/'smoke_001_sources/manifest_fixed.json');assert all(sha(P/'smoke_001_sources'/n)==h for n,h in manifest['files'].items()) and sha(P/'smoke_001_sources/manifest.json')==manifest['original_manifest_sha256']
log=R/'launcher.log';assert 'Specified kernel cache directory could not be created' in log.read_text()
records=rows(R/'records.jsonl');journal=rows(R/'generated.jsonl');names=[c['name'] for c in plan['cases']];assert [x['case'] for x in records]==[x['case'] for x in journal]==names
ref=next(x['actual'] for x in load(P/'processor_check002.json')['checks'] if x['task']==plan['selected']['task'] and str(x['qid'])==str(plan['selected']['qid']))
model=load(B.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')['recipe']['model_path'];tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False);force=tok.encode(' the',add_special_tokens=False)[0]
for x,j in zip(records,journal):
 assert x['ids']==j['ids'] and x['forced']==j['forced'] and x['decoded']==tok.decode(x['ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
 path=pathlib.Path(x['arrays']['path']).resolve();assert path.parent==(R/'arrays').resolve() and sha(path)==x['arrays']['sha256'];meta=x['arrays']['tensors'];case=x['case'];seq=x['calls'][0]['tokens']; k=256 if case in ('k256','k256_forced') else int(1024*.1111) if case=='author_default' else 1024
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)==set(meta)
  for name,item in meta.items():assert z[name].dtype==np.uint8 and hashlib.sha256(z[name].tobytes()).hexdigest()==item['sha256']
  for name,item in ref.items():assert meta['processor_'+name]==item
  if x['prunes']:
   selected=np.frombuffer(z['selected_ids_0'].tobytes(),dtype=np.int64);assert len(selected)==len(set(selected))==k and np.all(np.diff(selected)>=0) and np.all((selected>=0)&(selected<1024))
   dtype={'torch.bfloat16':torch.bfloat16,'torch.float16':torch.float16,'torch.float32':torch.float32}[meta['selection_scores_0']['dtype']];scores=torch.from_numpy(z['selection_scores_0'].copy()).view(dtype).float().numpy();mask=np.ones(1024,dtype=bool);mask[selected]=False;assert np.isfinite(scores).all() and (not mask.any() or scores[selected].min()>=scores[mask].max())
   pre=x['prunes'][0];start,stop=pre['range_before'];assert stop-start==1024 and pre['range_after']==[start,start+k] and pre['layer']==2;keep=np.r_[np.arange(start),start+selected,np.arange(stop,seq)]
   for name,axis in (('position',2),('hidden',1)):
    bef=z['prune_before_'+name+'_0'];aft=z['prune_after_'+name+'_0']
    if name=='position':bef=np.frombuffer(bef.tobytes(),np.int64).reshape(meta['prune_before_position_0']['shape']);aft=np.frombuffer(aft.tobytes(),np.int64).reshape(meta['prune_after_position_0']['shape'])
    assert np.array_equal(np.take(bef,keep,axis=axis),aft)
   for i in range(3):assert np.array_equal(z['injected_ds_'+str(i)],z['full_ds_'+str(i)] if i<2 else z['full_ds_'+str(i)][selected])
  logit_names=['logits_'+str(i) for i in range(len(x['calls']))] if x['forced'] else ['logits_0']
  for name in logit_names:
   dtype={'torch.bfloat16':torch.bfloat16,'torch.float16':torch.float16,'torch.float32':torch.float32}[meta[name]['dtype']];assert torch.isfinite(torch.from_numpy(z[name].copy()).view(dtype)).all()
 counts=[y['tokens'] for y in x['layers'] if y['step']==0];L=seq-1024+k;assert counts==[seq]*3+[L]*33 and [y['rows'] for y in x['injections']]==[1024,1024,k] and [y['mask_count'] for y in x['injections']]==[1024,1024,k]
 calls=x['calls'];assert len(calls)==len(x['ids']) and calls[0]['cache_after']==[L]*36
 for n,c in enumerate(calls[1:]):assert c['cache_before']==[L+n]*36 and c['cache_after']==[L+n+1]*36 and np.all(np.asarray(c['position_ids'])[-3:]==seq+int(x['rope_delta'][0][0])+n)
 if x['forced']:assert x['ids']==[force]*33
 else:assert x['ids'] and (x['ids'][-1] in x['eos_ids'] or len(x['ids'])==2048)
r={x['case']:x for x in records}
for left,right in [('stock','stock_forced'),('k256','k256_forced')]:
 common={'full_visual','original_prefill_position','original_prefill_embeds','original_visual_mask','logits_0'}|{'full_ds_'+str(i) for i in range(3)}|{'injected_ds_'+str(i) for i in range(3)}|{'injected_mask_'+str(i) for i in range(3)}|{'processor_'+n for n in ref}
 if left=='k256':common|={'selected_ids_0','selection_scores_0','prune_before_position_0','prune_after_position_0','prune_before_hidden_0','prune_after_hidden_0','prune_after_mask_0'}
 for n in common:assert r[left]['arrays']['tensors'][n]==r[right]['arrays']['tensors'][n]
checks={name:{'same_ids_as_stock':r[name]['ids']==r['stock']['ids'],'same_first_logits_as_stock':r[name]['arrays']['tensors']['logits_0']==r['stock']['arrays']['tensors']['logits_0']} for name in ('patched_disabled','priortr_keep_all')}
proof={'status':'passed','independent_review':True,'formal_proof_sha256':sha(formal),'formal_started_sha256':sha(fs),'formal_terminal_sha256':sha(ft),'source_manifest_sha256':sha(P/'smoke_001_sources/manifest_fixed.json'),'launcher_log_sha256':sha(log),'reviewer_sha256':sha(__file__),'cases':names,'endpoints':checks,'GT_opened':False,'scoring_performed':False,'limits':['one DEV question only','KV lengths and positions verified; KV contents not archived','SDPA and newer torch/transformers differ from E1 FA2','CUDA kernel caching disabled by runtime warning; no timing comparison']}
args.out.write_text(json.dumps(proof,indent=2)+'\n');print('passed',sha(args.out))
