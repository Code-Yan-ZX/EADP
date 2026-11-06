import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from light_verify_records import validate_light,tensor_hash

class LightVerifyTests(unittest.TestCase):
 def test_real_launcher_reaches_only_mocked_container_boundary_and_rejects_busy_gpu(self):
  import types,sys
  from unittest.mock import patch
  import light_smoke_launch,light_admission
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);reference=root/'reference.json';reference.write_text('{}');(root/'light_smoke_worker.py').write_text('worker');(root/'light_smoke_budget.json').write_text('{}')
   approval={'reference_path':str(reference),'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest()}
   budget={'minimum_free_bytes':0,'inner_timeout_seconds':3600};calls=[]
   module=types.ModuleType('run_panel')
   module.container_command=lambda plan,args,gpu,name:(calls.append((args,gpu,name)) or ['docker','run','--rm','image',*args])
   module.gpu_pids=lambda:[];module.identity=lambda pid:{'pid':pid}
   bounded=lambda *a,**kw:0
   oldpath=list(sys.path)
   try:
    with patch.object(light_smoke_launch,'HERE',root),patch.object(light_smoke_launch,'load_budget',return_value=budget),patch.object(light_smoke_launch,'check_supervisor'),patch.object(light_smoke_launch,'run_bounded',side_effect=bounded),patch.object(light_admission,'admit',return_value=approval),patch.dict(sys.modules,{'run_panel':module}),patch('sys.argv',['light_smoke_launch.py','--run-id','test','--references',str(reference)]):
     self.assertEqual(light_smoke_launch.main(),0)
     launch=json.loads((root/'light_smoke_test/launch.json').read_text());self.assertNotIn('--rm',launch['command']);self.assertEqual(len(calls),1);self.assertIs(calls[0][1],True)
     module.gpu_pids=lambda:[777]
     with patch('sys.argv',['light_smoke_launch.py','--run-id','busy','--references',str(reference)]):
      with self.assertRaisesRegex(ValueError,'GPU occupied'):light_smoke_launch.main()
     self.assertFalse((root/'light_smoke_busy').exists());self.assertEqual(len(calls),1)
   finally:sys.path[:]=oldpath
 def test_preflight_stop_timeout_signal_and_kill_fallback(self):
  from unittest.mock import patch
  from types import SimpleNamespace
  import subprocess
  from light_preflight_launch import cleanup_container,stop_reason
  self.assertIsNone(stop_reason(None,9,10));self.assertEqual(stop_reason(None,10,10),'timed_out');self.assertEqual(stop_reason(15,2,10),'interrupted')
  def inspect(running):
   item={'Id':'id','Image':'img','State':{'Running':running,'Pid':5 if running else 0},'Config':{'Image':'img','Entrypoint':[],'Cmd':[]},'HostConfig':{'Devices':[],'DeviceRequests':None}}
   return SimpleNamespace(returncode=0,stdout=json.dumps([item]),stderr='')
  for failure in (subprocess.TimeoutExpired(['docker','stop'],30),SimpleNamespace(returncode=1,stdout='',stderr='stop failed')):
   results=[failure,inspect(True),SimpleNamespace(returncode=0,stdout='id',stderr=''),inspect(False)]
   def run(*_a,**_kw):
    value=results.pop(0)
    if isinstance(value,Exception):raise value
    return value
   record={}
   with patch('light_preflight_launch.subprocess.run',side_effect=run):last=cleanup_container('name',record)
   self.assertEqual([x['action'] for x in record['cleanup_attempts']],['stop','kill']);self.assertFalse(last['state']['Running']);self.assertEqual(results,[])
 def test_worker_preflight_requires_own_independent_review(self):
  from unittest.mock import patch
  import light_admission,light_preflight_launch
  with patch.object(light_admission,'PRE_REVIEW',Path('/nonexistent-light-preflight-review-test')),patch('sys.argv',['light_preflight_launch.py','--run-id','test','--worker-preflight','--references','unused']):
   with self.assertRaisesRegex(ValueError,'independent review missing'):light_preflight_launch.main()
 def test_source_inventory_missing_or_changed_fails_closed(self):
  from light_admission import require_source_coverage
  expected={'a':'one','b':'two'};require_source_coverage({'a':'one','b':'two','extra':'ok'},expected,'proof')
  for observed in ({'a':'one'},{'a':'one','b':'bad'}):
   with self.assertRaisesRegex(ValueError,'inventory incomplete or changed'):require_source_coverage(observed,expected,'proof')
 def test_reviewer_terminal_cannot_belong_to_other_output(self):
  from light_admission import reviewer_command,REVIEW,REVIEWER
  s={'cwd':'/','command':['python',str(REVIEWER),'--out',str(REVIEW)]}
  reviewer_command(s);s['command'][-1]='/tmp/wrong.json'
  with self.assertRaisesRegex(ValueError,'reviewer process or output'):reviewer_command(s)
 def test_incomplete_supervisor_rejected_before_record_read(self):
  from light_verify import validate_completed
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);started=root/'started.json';terminal=root/'terminal.json'
   started.write_text(json.dumps({'worker_pid':42}));terminal.write_text(json.dumps({'worker_pid':42,'reason':'progress_stale','worker_returncode':0,'remaining_process_group':[]}))
   with self.assertRaisesRegex(ValueError,'supervised process incomplete'):validate_completed(root,started,terminal)
 def test_journal_replay_rejects_changed_raw_output(self):
  from unittest.mock import patch
  from light_smoke_plan import make_plan
  from light_verify_core import verify_records
  anchors=[{'task':t,'qid':str(i),'group_id':str(i),'message_sha256':'m','image_sha256':'i'} for i,t in enumerate(('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'))]
  plan=make_plan(anchors);jobs=plan['jobs'];b=[];old=[];records=[];raw=[];comparisons=[]
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'arrays').mkdir()
   for j in jobs:
    mode=j['path'];forced=j['forced'];ident={k:j[k] for k in ('task','qid','group_id','forced')}
    rec=dict(ident,mode=mode,output_ids=[1],eos_ids=[1],generation_scalars={},min_new_tokens=33 if forced else None,arrays={'path':str(root/'arrays'/(str(j['order'])+'.npz'))})
    if mode=='fresh_on_reference':old.append(dict(ident,prefill_logits='present'));rec['prefill_logits']='present';refname='accepted_e4_on'
    elif mode=='C-ON':refname='fresh_on'
    else:b.append(dict(ident,mode=mode));refname='accepted_b_'+mode
    records.append({'job':j,'record':rec});raw.append(dict(ident,path=mode,output_ids=[1],eos_ids=[1],generation_scalars={},min_new_tokens=rec['min_new_tokens']))
    comparisons.append({'job':j,'reference':refname,'metadata':{'ok':True},'arrays':{'ok':True}})
   def write(name,items):(root/name).write_text(''.join(json.dumps(x)+'\n' for x in items))
   write('records.jsonl',records);write('generated.jsonl',raw);write('comparisons.jsonl',comparisons)
   ref={'status':'passed','plan':plan,'b_records':b,'old_on':old}
   with patch('light_verify_core.compare_metadata',return_value={'ok':True}),patch('light_verify_core.compare_archives',return_value={'ok':True}),patch('light_verify_core.validate_light'):
    self.assertEqual(verify_records(root,ref,1)['generations'],16)
    raw[-1]['output_ids']=[999];write('generated.jsonl',raw)
    with self.assertRaisesRegex(ValueError,'generation journal mismatch'):verify_records(root,ref,1)
 def test_identity_launcher_requires_exact_supervisor_and_budget(self):
  from light_smoke_launch import check_supervisor
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);source=root/'initial.json';source.write_text(json.dumps({'budget_start_unix':100,'budget_end_unix':10000}))
   plan={'budget_source':str(source),'budget_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'budget_start_unix':100,'budget_end_unix':10000,
         'inner_timeout_seconds':3600,'outer_timeout_seconds':3720,'grace_seconds':90,'journal_stale_seconds':1200,'snapshot_seconds':3600,'poll_seconds':15}
   started={'worker_pid':10,'timeout_seconds':3720,'stale_seconds':1200,'progress_paths':[str(root/'generated.jsonl')],'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':90}
   p=root/'started.json';p.write_text(json.dumps(started));self.assertEqual(check_supervisor(p,root/'generated.jsonl',plan,200,10)['inner_timeout_seconds'],3600)
   started['stale_seconds']=1;p.write_text(json.dumps(started))
   with self.assertRaisesRegex(ValueError,'supervisor'):check_supervisor(p,root/'generated.jsonl',plan,200,10)
   started['stale_seconds']=1200;p.write_text(json.dumps(started))
   with self.assertRaisesRegex(ValueError,'budget'):check_supervisor(p,root/'generated.jsonl',plan,7000,10)
 def test_identity_launcher_is_locked(self):
  from unittest.mock import patch
  import light_admission,light_smoke_launch
  with patch.object(light_admission,'REVIEW',Path('/nonexistent-light-review-test')),patch('sys.argv',['light_smoke_launch.py','--run-id','test','--references','unused']):
   with self.assertRaisesRegex(ValueError,'independent review missing'):light_smoke_launch.main()
 def test_actual_arrays_selection_position_and_delta_counterexamples(self):
  s=1;L=1026;spatial=list(range(256));ret=[0]+[1+x for x in spatial]+[1025];n=len(ret)
  full=np.tile(np.arange(L,dtype=np.int64),(3,1));pos=full[:,ret][:,None,:].tolist()
  visual=np.arange(2048,dtype=np.uint8).reshape(1024,2);pruned=visual[spatial];prefill=np.zeros((1,n,2),dtype=np.uint8);prefill[0,1:257]=pruned
  logits=np.array([1.,2.],dtype=np.float32).view(np.uint8)
  tensors={'visual':visual,'pruned':pruned,'prefill':prefill,'prefill_logits':logits}
  for k in ('input_ids','attention_mask','pixel_values','image_grid_thw'):tensors['processor_'+k]=np.array([1],dtype=np.uint8)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'a.npz';np.savez(path,**tensors)
   meta=lambda x:{'sha256':tensor_hash(x)}
   r={'mode':'C-SKIP','forced':False,'forced_token':None,'output_ids':[2],'eos_ids':[2],'max_new_tokens':2048,'min_new_tokens':None,'generation_scalars':{'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0},'eos':True,'truncated':False,
      'image_start':s,'original_length':L,'ordered_ids':spatial,'spatial_ids':spatial,'retained_ids':ret,
      'full_position_ids':full[:,None,:].tolist(),'full_delta':[[0]],'adjusted_delta':[[768]],'prefill_position_ids':pos,
      'decode_trace':[{'rope_delta':[[768]],'layer_cache_lengths':[],'layer_cache_lengths_after':[n]*36,'cache_position':list(range(n)),'position_ids':pos}],
      'arrays':{'kind':'full_arrays','path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
      'main_visual':meta(visual),'pruned_visual':meta(pruned),'prefill_embeds':meta(prefill),'prefill_logits':dict(meta(logits),dtype='torch.float32'),
      'processor':{k:meta(tensors['processor_'+k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},'injected_mask':None,'injected_deepstack':None,
      'cache_fingerprints':[[{'key':{'sha256':'a'*64,'shape':[1],'dtype':'torch.bfloat16'},'value':{'sha256':'b'*64,'shape':[1],'dtype':'torch.bfloat16'}} for _ in range(36)]]}
   self.assertEqual(validate_light(r,path,4)['status'],'passed')
   bad=copy.deepcopy(r);bad['output_ids']=[1]*2048+[2]
   with self.assertRaisesRegex(ValueError,'generation recipe'):validate_light(bad,path,4)
   bad=copy.deepcopy(r);bad['full_delta']=[[999231]];bad['adjusted_delta']=[[999999]]
   with self.assertRaisesRegex(ValueError,'position or delta'):validate_light(bad,path,4)
   bad=copy.deepcopy(r);bad['decode_trace'][0]['layer_cache_lengths_after'][0]-=1
   with self.assertRaisesRegex(ValueError,'decode/cache'):validate_light(bad,path,4)
   bad=copy.deepcopy(r);bad['arrays']['sha256']='0'*64
   with self.assertRaisesRegex(ValueError,'archive identity'):validate_light(bad,path,4)
