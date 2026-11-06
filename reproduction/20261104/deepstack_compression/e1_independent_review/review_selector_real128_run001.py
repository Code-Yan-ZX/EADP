"""Independent no-GT review of all 96 expansion selector cases."""
import argparse,hashlib,json,math,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e5/selector_real128';R=B/'e1_independent_review';O=D/'run_001';sys.path[:0]=[str(D),str(B/'e4_c_latency'),str(B/'e4_c_microbench_pilot')]
import torch
from input_preflight import load_accepted_e1_references
from pilot_worker import get_reference
from panel_reference import accepted,get_records
from admission import validate_container
from identity_gate import sha
from expansion_contract import load_manifest,selected
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==R/'selector_real128_run001_review.json'
def read(p):return json.loads(p.read_text())
def terminal(mon,script):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in s['command'] if Path(x).name==script.name]==[script.resolve()];return s
def meta(x):
 y=x.detach().cpu().contiguous();return {'dtype':str(y.dtype),'shape':list(y.shape),'sha256':hashlib.sha256(y.view(torch.uint8).numpy().tobytes()).hexdigest()}
f=read(D/'run_001_verification.json');assert f['status']=='passed' and f['images']==96 and f['cases']==288 and f['GT_opened'] is False and f['scoring_performed'] is False and f['verifier_sha256']==sha(D/'verify.py')
sv=terminal(D/'monitor_verify_run001',D/'verify.py');sr=terminal(D/'monitor_run001',D/'launch.py');launch=read(O/'launch.json');ex=read(O/'exit.json');assert sr['worker_pid']==launch['controller']['pid']==ex['controller']['pid'] and ex['container_stopped_verified'] and ex['returncode']==ex['container_exit_code']==0 and ex['gpu_pids_after']==[]
obj=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];container=validate_container(launch,ex,obj,gpu=True)
assert len(f['array_sha256'])==384 and all(sha(p)==h for p,h in f['array_sha256'].items()) and all(sha(p)==h for p,h in f['files_sha256'].items()) and all(sha(p)==h for p,h in f['source_sha256'].items())
manifest,samples=load_manifest(B/'e5/selector_real32/sample_manifest_001.json');assert len(samples)==96 and len({x['group_id'] for x in samples})==96
refs,_=load_accepted_e1_references(B/'e1/e1full001_completion.json',B/'e1/monitorverifye1full001/started.json',B/'e1/monitorverifye1full001/terminal.json');refs={(x['task'],str(x['qid'])):x for x in refs};e4=get_records(samples,accepted())
events=[json.loads(x) for x in (O/'journal.jsonl').read_text().splitlines() if x.strip()];assert events[0]['status']=='synthetic_completed';synthetic=events[0]['cases'];assert len(synthetic)==7
for x in synthetic:
 assert x['graph_equal'] is (x['author_ids']==x['graph_ids'])
 if 'fused_equal' in x:assert x['fused_equal'] is (x['author_ids']==x['fused_ids'])
chunks=[];cur=None
for e in events[1:]:
 if e['status']=='image_started':assert cur is None and e['image_index']==len(chunks);cur=[e]
 elif e['status']=='image_completed':assert cur is not None and e['image_index']==len(chunks);cur.append(e);chunks.append(cur);cur=None
 else:assert cur is not None and e['image_index']==len(chunks);cur.append(e)
assert cur is None and len(chunks)==96
formal_outcomes={(x['image_index'],x['budget']):x for x in f['outcomes']};assert len(formal_outcomes)==288
mismatch=[];counts={'fused':0,'graph':0};nmeasured={'author':0,'fused':0,'graph':0};setup={}
for i,(sample,chunk) in enumerate(zip(samples,chunks)):
 for edge in (chunk[0],chunk[-1]):assert all(str(edge.get(k))==str(sample[k]) for k in ('task','qid','group_id'))
 for e in chunk:
  if e.get('status')=='graph_setup':assert e['budget'] not in setup;setup[e['budget']]=e['result']
 x=torch.load(O/('image_%03d'%i)/'capture.pt',map_location='cpu',weights_only=True);actual=x['actual'];args=x['selector_args'];ref=get_reference(refs[(sample['task'],str(sample['qid']))]);visual=actual['visual'];pre=actual['prefill']
 assert all(meta(actual['processor'][k])==ref['processor'][k] for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')) and meta(visual[0])==ref['visual_sha256'] and [meta(t) for t in visual[1]]==ref['deepstack_full_sha256']
 assert actual['ordered'].reshape(-1).tolist()==ref['ordered_ids'] and meta(pre['inputs_embeds'])==ref['prefill_embed_sha256'] and meta(pre['attention_mask'])==ref['attention_mask_sha256'] and pre['position_ids'].tolist()==ref['prefill_position_ids'] and actual['rope_delta'].tolist()==ref['rope_delta'] and pre['visual_pos_masks'].tolist()==ref['injected_mask'] and [meta(t) for t in pre['deepstack_visual_embeds']]==ref['injected_deepstack']
 ids=actual['processor']['input_ids'][0].tolist();instr=x['instruction_ids'];starts=[j for j in range(len(ids)-len(instr)+1) if ids[j:j+len(instr)]==instr];assert len(starts)==1;start=ref['image_start'];positions=[j if j<start else j-768 for j in range(starts[0],starts[0]+len(instr))];assert torch.equal(pre['inputs_embeds'][:,positions,:],args[2]) and torch.equal(args[0],visual[0])
 cases={e['budget']:e for e in chunk if e.get('status')=='case_completed'};assert set(cases)=={128,256,512};arrays={}
 for k in (128,256,512):
  z=torch.load(O/('image_%03d'%i)/('case_%d.pt'%k),map_location='cpu',weights_only=True);arrays[k]=z;ordered=z['ordered'][0].tolist();rec=e4[(sample['task'],str(sample['qid']),k)];assert ordered==rec['ordered_ids'] and sorted(ordered)==rec['spatial_ids'] and meta(z['pruned'][0])==rec['pruned_visual'] and meta(visual[0])==rec['main_visual'] and [meta(t) for t in visual[1]]==rec['full_deepstack'] and all(meta(actual['processor'][j])==rec['processor'][j] for j in rec['processor'])
  result=cases[k]['result'];assert result['status']=='measured' and result['identity']['author_equals_endpoint'] is True and len(result['measurements'])==15
  for m in ('fused','graph'):
   got=result['identity'][m+'_ordered_ids'];same=got==ordered;assert result['identity'][m+'_equals_author'] is same
   observed=formal_outcomes[(i,k)];assert observed['status']=='measured' and observed['task']==sample['task'] and str(observed['qid'])==str(sample['qid']);comparison=selected(ordered,got,k);assert all(observed[m][q]==v for q,v in comparison.items())
   if not same:counts[m]+=1;mismatch.append({'task':sample['task'],'qid':sample['qid'],'budget':k,'method':m,'first_order_difference':next(j for j,(u,v) in enumerate(zip(ordered,got)) if u!=v),'author_len':len(ordered),'candidate_len':len(got),'same_set':set(ordered)==set(got)})
  for m in ('author','fused','graph'):
   entries=[z for z in result['measurements'] if z['implementation']==m];assert len(entries)==5 and sorted(z['repeat'] for z in entries)==list(range(5))
   for z in entries:
    if z['status']=='measured':
     assert z['ordered_ids']==ordered and z['equals_author'] is True and all(math.isfinite(z[t]) and z[t]>=0 for t in ('gpu_event_ms','host_wall_ms'));nmeasured[m]+=1
     if m!='author':
      recorded=formal_outcomes[(i,k)][m]['repeats'][z['repeat']];assert recorded['repeat']==z['repeat'] and recorded['status']=='measured' and recorded['comparison']==selected(ordered,z['ordered_ids'],k)
    else:assert m!='author' and result['identity'][m+'_equals_author'] is False and z['status']=='unavailable'
 for k in (128,512):assert torch.equal(arrays[k]['similarity'],arrays[256]['similarity']) and torch.equal(arrays[k]['importance'],arrays[256]['importance'])
assert set(setup)=={128,256,512} and f['counts']['fused_mismatch_or_unavailable']==counts['fused'] and f['counts']['graph_mismatch_or_unavailable']==counts['graph']
assert nmeasured=={'author':1440,'fused':1440,'graph':1440} and len(f['outcomes'])==288 and all(x['status']=='measured' for x in f['outcomes'])
r={'status':'passed','independent_review':True,'images':96,'cases':288,'mismatch_cases':mismatch,'measured_counts':nmeasured,'formal_sha256':sha(D/'run_001_verification.json'),'formal_started_sha256':sha(D/'monitor_verify_run001/started.json'),'formal_terminal_sha256':sha(D/'monitor_verify_run001/terminal.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'reviewer_sha256':sha(__file__),'array_sha256':f['array_sha256'],'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'container':container,'limitations':['96 further images selected from a fixed 128-image expansion after earlier pilot timing','GPU per-round greedy scores were not archived; no offline GPU causal attribution','E4 full comparison uses recorded tensor hashes/metadata and IDs','runtime mean equality is not reproduced by CPU BF16 reductions','not end-to-end TTFT; graph capture/setup excluded from replay'], 'errors':[]};a.out.write_text(json.dumps(r,indent=2)+'\n');print('passed')
