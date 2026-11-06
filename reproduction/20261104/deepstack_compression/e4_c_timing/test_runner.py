import copy,json,os,subprocess,sys,types,unittest
from pathlib import Path
import torch
from torch import nn
from timer import FirstTokenBoundary,measure
from light_adapter import LightAdapter
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'e4'))
from test_skip_deepstack import FakeVision
class FakePruner(nn.Module):
 def __init__(self,namespace):super().__init__();self.namespace=namespace
 def forward(self,x):
  selected,_=self.namespace._greed_select_impl();return x.index_select(0,selected[0].sort().values),[256]
class FakeCore:
 def __init__(self):self.language_model=None;self.rope_deltas=None
 def get_rope_index(self,input_ids,_grid,_video,attention_mask):
  length=input_ids.shape[1];return torch.arange(length).view(1,1,length).expand(3,1,length),torch.tensor([[0]])
class FakeModel:
 def __init__(self):self.visual=FakeVision();self.model=FakeCore();self.config=types.SimpleNamespace(image_token_id=999);self.embed=nn.Embedding(1000,4)
 def get_input_embeddings(self):return self.embed
class FakeWrapper:
 def __init__(self):
  self.namespace=types.SimpleNamespace(_greed_select_impl=lambda: (torch.arange(255,-1,-1).reshape(1,256),torch.tensor([256])))
  self.model=FakeModel();self.pruner=FakePruner(self.namespace)
 def _processor_inputs(self,_message):
  ids=torch.tensor([[1,2]+[999]*1024+[3,4,5]]);return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'pixel_values':torch.ones(1024,4),'image_grid_thw':torch.tensor([[1,64,64]])}


class Capture(nn.Module):
 def __init__(self):super().__init__();self.kw=None
 def forward(self,inputs_embeds,attention_mask,position_ids=None,visual_pos_masks=None,deepstack_visual_embeds=None):
  self.kw={'position_ids':position_ids,'visual_pos_masks':visual_pos_masks,'deepstack_visual_embeds':deepstack_visual_embeds};return inputs_embeds.mean(dim=1)

def run_fake(mode):
 torch.manual_seed(7);w=FakeWrapper();w.model.model.language_model=Capture()
 with LightAdapter(torch,w,w.namespace,mode,2) as a:
  x=w._processor_inputs([]);visual,streams=w.model.visual(x['pixel_values']);pruned,_=w.pruner(visual);embedded=w.model.get_input_embeddings()(x['input_ids']);embeds=torch.cat((embedded[:,:2],pruned[None],embedded[:,1026:]),dim=1)
  y=w.model.model.language_model(inputs_embeds=embeds,attention_mask=torch.ones((1,261),dtype=torch.long));kw=w.model.model.language_model.kw
  w.model.model.language_model(inputs_embeds=torch.ones(1,1,4),attention_mask=torch.ones((1,262),dtype=torch.long))
 return visual,streams,y,kw,w
class AdapterTests(unittest.TestCase):
 def test_same_selection_positions_and_skip_drop_primary(self):
  vd,ds,yd,kd,wd=run_fake('C-DROP');vs,skip,ys,ks,ws=run_fake('C-SKIP');vo,on,yo,ko,wo=run_fake('C-ON')
  self.assertTrue(torch.equal(vd,vs) and torch.equal(vd,vo));self.assertTrue(torch.equal(yd,ys) and torch.equal(yd,yo));self.assertTrue(torch.equal(kd['position_ids'],ks['position_ids']) and torch.equal(kd['position_ids'],ko['position_ids']))
  self.assertIsNone(kd['visual_pos_masks']);self.assertIsNone(ks['deepstack_visual_embeds']);self.assertEqual(int(ko['visual_pos_masks'].sum()),256)
  for a,b in zip(ko['deepstack_visual_embeds'],on):self.assertTrue(torch.equal(a,b[:256]))
  self.assertTrue(all(x is None for x in skip));self.assertFalse(wd.model.visual._forward_hooks);self.assertFalse(ws.model.model.language_model._forward_pre_hooks)
class TimerTests(unittest.TestCase):
 def test_first_callback_occurs_after_token_append_in_real_transformers(self):
  from transformers import GPT2Config,GPT2LMHeadModel
  from transformers.generation.stopping_criteria import StoppingCriteriaList
  torch.manual_seed(42);model=GPT2LMHeadModel(GPT2Config(vocab_size=32,n_positions=32,n_embd=16,n_layer=2,n_head=2,eos_token_id=None,pad_token_id=0)).eval();ids=torch.tensor([[1,2,3]])
  with torch.inference_mode():baseline=model.generate(input_ids=ids,do_sample=False,max_new_tokens=4)
  sync=[];clock=iter([1000,6000,12000]).__next__
  def run(boundary):
   with torch.inference_mode():return model.generate(input_ids=ids,do_sample=False,max_new_tokens=4,stopping_criteria=StoppingCriteriaList([boundary]))
  actual,times=measure(run,lambda:sync.append('sync'),clock)
  self.assertTrue(torch.equal(actual,baseline));self.assertEqual(times['first_criteria_length'],4);self.assertEqual(times['criteria_calls'],4);self.assertEqual(times['first_token_selected_ns'],6000);self.assertEqual(len(sync),3)
 def test_missing_first_token_aborts(self):
  with self.assertRaises(ValueError):measure(lambda boundary:'no-generation',lambda:None)
class MemoryProbeTests(unittest.TestCase):
 def test_kv_bytes_and_hook_removal_without_gpu(self):
  from memory_probe import KVProbe
  class Model(nn.Module):
   def forward(self,x):return types.SimpleNamespace(past_key_values=[(x,x),(x[:1],x[:1])])
  m=Model();x=torch.zeros(2,3,dtype=torch.float32)
  with KVProbe(m) as probe:
   m(x);self.assertEqual(probe.latest_bytes,72);self.assertEqual(probe.calls,1);m(x[:1]);self.assertEqual(probe.max_bytes,72)
  self.assertFalse(m._forward_hooks)
 def test_missing_cache_fails_and_removes_hook(self):
  from memory_probe import KVProbe
  class Model(nn.Module):
   def forward(self,x):return types.SimpleNamespace(past_key_values=None)
  m=Model()
  with self.assertRaisesRegex(ValueError,'KV cache'):
   with KVProbe(m):m(torch.zeros(1))
  self.assertFalse(m._forward_hooks)
class InterferenceTests(unittest.TestCase):
 def test_health_window_and_overlap_annotation_retain_all(self):
  from timing_supervise import timed_health
  from interference import annotate
  clock=iter([100,200]).__next__;snap=timed_health(lambda:{'gpu':{}},clock,lambda:'time:[1]')()
  rows=[{'t0_before_image_read_ns':50,'generation_complete_ns':100},{'t0_before_image_read_ns':201,'generation_complete_ns':300}]
  result=annotate(rows,[snap],'time:[1]');self.assertEqual(result['n_records'],2);self.assertEqual(result['n_monitor_overlaps'],1)
  self.assertTrue(result['records'][0]['monitor_overlap']);self.assertFalse(result['records'][1]['monitor_overlap'])
  with self.assertRaisesRegex(ValueError,'clock'):annotate(rows,[snap],'time:[2]')
  with self.assertRaisesRegex(ValueError,'clock'):annotate(rows,[{'utc':'old'}],'time:[1]')
  from interference import annotate_events
  event={'status':'generated','result':{'boundaries':dict(rows[0],time_namespace='time:[1]',clock_implementation='clock_gettime(CLOCK_MONOTONIC)')}}
  snapshots=[{'health_snapshot_window_monotonic_ns':[0,40],'time_namespace':'time:[1]'},snap,{'event':'terminal_snapshot','health_snapshot_window_monotonic_ns':[300,400],'time_namespace':'time:[1]'}]
  started={'worker_pid':123,'snapshot_seconds':3600,'poll_seconds':15};terminal={'worker_pid':123,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
  output=annotate_events([event,{'status':'failed'}],snapshots,'time:[1]',started,terminal);self.assertEqual(output['n_events'],2);self.assertTrue(output['events'][0]['monitor_overlap']);self.assertEqual(output['events'][1],{'status':'failed'})
  bad=copy.deepcopy(event);bad['result']['boundaries']['time_namespace']='time:[2]'
  with self.assertRaisesRegex(ValueError,'clocks'):annotate_events([bad],snapshots,'time:[1]',started,terminal)
  for broken in ([],snapshots[:-1],snapshots[1:]):
   with self.assertRaises(ValueError):annotate_events([event],broken,'time:[1]',started,terminal)
  with self.assertRaises(ValueError):annotate_events([event],snapshots,'time:[1]',started,dict(terminal,worker_returncode=1))
class LightCaptureTests(unittest.TestCase):
 def test_capture_observes_actual_prefill_and_releases_hooks(self):
  from light_capture import LightCapture
  for mode in ('C-ON','C-DROP','C-SKIP'):
   torch.manual_seed(7);w=FakeWrapper();w.model.model.language_model=Capture();w.model.lm_head=nn.Linear(4,3)
   with LightCapture(torch,w,w.namespace,mode,2) as capture:
    x=w._processor_inputs([]);visual,streams=w.model.visual(x['pixel_values']);pruned,_=w.pruner(visual);embedded=w.model.get_input_embeddings()(x['input_ids']);embeds=torch.cat((embedded[:,:2],pruned[None],embedded[:,1026:]),dim=1)
    w.model.model.language_model(inputs_embeds=embeds,attention_mask=torch.ones((1,261),dtype=torch.long));w.model.lm_head(torch.zeros(1,1,4))
    state=capture.state;self.assertTrue(torch.equal(state['visual'],visual));self.assertTrue(torch.equal(state['pruned'],pruned));self.assertEqual(state['prefill_position_ids'].shape,(3,1,261))
    self.assertEqual(state['logits'].shape,(1,3));self.assertEqual(state['adjusted_delta'].tolist(),[[768]])
    self.assertEqual(state['block_calls'],[1]*5);self.assertEqual(state['main_merger_calls'],1);self.assertEqual(state['ds_calls'],[1]*3);self.assertEqual(state['ds_none'],[mode=='C-SKIP']*3)
    if mode=='C-ON':self.assertEqual(len(state['injected_ds']),3);self.assertEqual(int(state['prefill_visual_pos_masks'].sum()),256)
    else:self.assertIsNone(state['injected_ds']);self.assertIsNone(state['prefill_visual_pos_masks'])
   self.assertFalse(w.model.visual._forward_hooks);self.assertFalse(w.model.model.language_model._forward_pre_hooks);self.assertFalse(w.model.lm_head._forward_hooks)
class LightArmTests(unittest.TestCase):
 def test_forced_prefill_and_32_decode_with_actual_hook_path(self):
  from light_arm import run_arm
  class Cache:
   def __init__(self,n):
    self.n=n;self.layers=[types.SimpleNamespace(keys=torch.zeros(1,1,n,1),values=torch.zeros(1,1,n,1)) for _ in range(36)]
   def get_seq_length(self,i):return self.n
  class Language(nn.Module):
   def __init__(self):super().__init__();self.layers=[None]*36
   def forward(self,inputs_embeds,attention_mask,position_ids=None,visual_pos_masks=None,deepstack_visual_embeds=None,past_key_values=None,cache_position=None):
    n=inputs_embeds.shape[1] if past_key_values is None else past_key_values.n+1
    return types.SimpleNamespace(past_key_values=Cache(n))
  for mode in ('C-ON','C-DROP','C-SKIP'):
   torch.manual_seed(7);w=FakeWrapper();w.model.model.language_model=Language();w.model.lm_head=nn.Linear(4,3)
   w.model.generation_config=types.SimpleNamespace(eos_token_id=[5]);w.visual_token_num=256;w.pruner.visual_token_num=256
   def model_generate(**kwargs):
    x=w._processor_inputs([]);v,_=w.model.visual(x['pixel_values']);pruned,_=w.pruner(v);e=w.model.get_input_embeddings()(x['input_ids']);emb=torch.cat((e[:,:2],pruned[None],e[:,1026:]),dim=1)
    out=w.model.model.language_model(inputs_embeds=emb,attention_mask=torch.ones(1,261,dtype=torch.long),cache_position=torch.arange(261));w.model.lm_head(torch.zeros(1,1,4))
    for i in range(32):
     pos=261+i;out=w.model.model.language_model(inputs_embeds=torch.zeros(1,1,4),attention_mask=torch.ones(1,pos+1,dtype=torch.long),position_ids=torch.tensor([[[pos+768]]]*3),cache_position=torch.tensor([pos]),past_key_values=out.past_key_values)
     w.model.lm_head(torch.zeros(1,1,4))
    return torch.tensor([[6]*33])
   w.model.generate=model_generate
   def generate(wrapper,row):
    wrapper.model.generate(max_new_tokens=2048,do_sample=False,top_p=.001,top_k=1,temperature=.01,repetition_penalty=1.0)
    return 'forced'
   row={'task':'T','qid':'1','group_id':'g','image_sha256':'i','message_sha256':'m','split':'DEV','image_start':2};journal=[]
   record=run_arm(torch,w,w.namespace,row,mode,generate,journal.append,lambda arrays:{k:list(v.shape) for k,v in arrays.items()},6)
   self.assertEqual(record['output_ids'],[6]*33);self.assertEqual(len(record['decode_trace']),33);self.assertEqual(len(record['cache_fingerprints']),33)
   self.assertEqual(record['prefill_embeds']['shape'],[1,261,4]);self.assertEqual(record['arrays']['prefill'],[1,261,4]);self.assertEqual(len(record['prefill_position_ids'][0][0]),261)
   self.assertEqual(record['ds_none'],[mode=='C-SKIP']*3);self.assertEqual(len(journal),1)
   if mode=='C-ON':self.assertEqual(len(record['injected_deepstack']),3);self.assertEqual(sum(record['injected_mask'][0]),256)
   else:self.assertIsNone(record['injected_deepstack']);self.assertIsNone(record['injected_mask'])
   self.assertFalse(w.model.visual._forward_hooks);self.assertFalse(w.model.model.language_model._forward_pre_hooks)
class LightSmokePlanTests(unittest.TestCase):
 def test_three_anchors_make_sixteen_fixed_no_gt_jobs(self):
  from light_smoke_plan import make_plan
  anchors=[{'task':t,'qid':str(i),'group_id':str(i),'message_sha256':'m','image_sha256':'i'} for i,t in enumerate(('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'))]
  plan=make_plan(anchors);self.assertEqual(len(plan['jobs']),16);self.assertEqual(sum(x['forced'] for x in plan['jobs']),4)
  self.assertEqual(sum(x['path']=='fresh_on_reference' for x in plan['jobs']),4);self.assertEqual(plan,make_plan(anchors))
  bad=copy.deepcopy(anchors);bad[1]['group_id']=bad[0]['group_id']
  with self.assertRaises(ValueError):make_plan(bad)
class LightSmokeCoreTests(unittest.TestCase):
 def test_nonboolean_forced_reference_is_rejected(self):
  from light_smoke_core import key
  with self.assertRaisesRegex(ValueError,'boolean'):key({'task':'T','qid':'1','forced':'false'})
 def test_identity_worker_is_locked_before_gpu_import(self):
  from unittest.mock import patch
  import light_admission
  with patch.object(light_admission,'REVIEW',Path('/nonexistent-light-review-test')):
   with self.assertRaisesRegex(ValueError,'identity smoke locked'):light_admission.admit(Path('/nonexistent-light-review-test'))
 def test_all_16_jobs_order_and_partial_failure_journal(self):
  import tempfile
  from unittest.mock import patch
  from light_smoke_core import run_diagnostics
  from light_smoke_plan import make_plan
  anchors=[{'task':t,'qid':str(i),'group_id':str(i),'message_sha256':'m','image_sha256':'i'} for i,t in enumerate(('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'))]
  rows=[dict(x,split='DEV',image_start=2) for x in anchors];plan=make_plan(anchors)
  b=[{'task':j['task'],'qid':j['qid'],'forced':j['forced'],'mode':j['path']} for j in plan['jobs'] if j['path'] in ('C-DROP','C-SKIP')]
  old=[{'task':j['task'],'qid':j['qid'],'forced':j['forced'],'prefill_logits':None if j['task']!='DocVQA_VAL' else 'logits'} for j in plan['jobs'] if j['path']=='fresh_on_reference']
  counter=[]
  def fresh(*args,**kwargs):
   counter.append('fresh');args[7]({'output_ids':[1]});return {'prefill_logits':'logits','arrays':{}}
  def light(*args,**kwargs):
   counter.append(args[4]);args[6]({'output_ids':[1]});return {'arrays':{}}
  with tempfile.TemporaryDirectory() as d,patch('light_smoke_core.compare_metadata',return_value={'status':'ok'}),patch('light_smoke_core.compare_archives',return_value={'n':1}):
   result=run_diagnostics(None,None,None,rows,anchors,b,old,d,None,fresh,light,lambda n:None,1)
   self.assertEqual(result['jobs'],16);self.assertEqual(len((Path(d)/'generated.jsonl').read_text().splitlines()),16);self.assertEqual(counter[:4],['fresh','C-ON','C-DROP','C-SKIP'])
  with tempfile.TemporaryDirectory() as d,patch('light_smoke_core.compare_metadata',side_effect=ValueError('mismatch')):
   with self.assertRaisesRegex(ValueError,'mismatch'):run_diagnostics(None,None,None,rows,anchors,b,old,d,None,fresh,light,lambda n:None,1)
   self.assertEqual(len((Path(d)/'generated.jsonl').read_text().splitlines()),1);self.assertEqual(len((Path(d)/'errors.jsonl').read_text().splitlines()),1)
class LightComparisonTests(unittest.TestCase):
 def test_mismatched_outputs_and_positions_rejected(self):
  from light_compare import COMMON,compare_metadata
  for mode in ('C-ON','C-DROP','C-SKIP'):
   actual={k:'same' for k in COMMON};actual.update(mode=mode,forced=False,prefill_logits='logits',ds_none=[mode=='C-SKIP']*3,injected_mask='mask' if mode=='C-ON' else None,injected_deepstack='ds' if mode=='C-ON' else None,cache_fingerprints='kv')
   ref={k:actual[k] for k in COMMON}
   if mode=='C-ON':ref.update(budget=256,ds_on=True,actual_injection_verified=True,prefill_logits='logits',injected_mask='mask',injected_deepstack='ds',cache_fingerprints='kv')
   else:ref.update(mode=mode,actual_ds_off=True,first_logits='logits',cache_fingerprints='kv')
   self.assertEqual(compare_metadata(actual,ref)['status'],'identity_passed_bitwise')
   if mode=='C-ON':
    no_kv=dict(ref);del no_kv['cache_fingerprints']
    with self.assertRaisesRegex(ValueError,'fingerprints missing'):compare_metadata(actual,no_kv)
    self.assertFalse(compare_metadata(actual,no_kv,allow_missing_reference_kv=True)['cache_fingerprints_compared'])
    no_ref=dict(ref,prefill_logits=None)
    with self.assertRaises(ValueError):compare_metadata(actual,no_ref)
    self.assertFalse(compare_metadata(actual,no_ref,allow_missing_reference_logits=True)['prefill_logits_compared'])
    with self.assertRaises(ValueError):compare_metadata(dict(actual,forced=True),dict(no_ref,forced=True))
   for key in ('prefill_position_ids','output_ids','decode_trace','ordered_ids','cache_fingerprints'):
    bad=copy.deepcopy(actual);bad[key]='wrong'
    with self.assertRaises(ValueError):compare_metadata(bad,ref)
 def test_array_payload_mismatch_rejected(self):
  import tempfile,numpy as np
  from light_compare import compare_archives,sha
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'a.npz';b=Path(d)/'b.npz';data={x:np.array([1,2]) for x in ('visual','prefill','prefill_logits')};np.savez(a,**data);np.savez(b,**data)
   aa={'arrays':{'kind':'full_arrays','path':str(a),'sha256':sha(a)}};rr={'arrays':{'kind':'full_arrays','path':str(b),'sha256':sha(b)}}
   self.assertEqual(compare_archives(aa,rr)['n_compared'],3)
   np.savez(b,visual=np.array([0]),prefill=data['prefill'],prefill_logits=data['prefill_logits']);rr['arrays']['sha256']=sha(b)
   with self.assertRaisesRegex(ValueError,'differs'):compare_archives(aa,rr)
class AuthorPreflightTests(unittest.TestCase):
 def test_real_author_message_builder_precedes_processor(self):
  import real_worker
  from unittest.mock import patch
  root='/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/Qwen_vl'
  sys.path[:0]=[root,str(Path(root)/'VLMEvalKit')]
  from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner
  class Wrapper:
   system_prompt=None
   model=types.SimpleNamespace(config=types.SimpleNamespace(image_token_id=999))
   def __init__(self):self.received=None;self.dataset=None
   def _prepare_content(self,message,dataset=None):self.dataset=dataset;return [{'type':'text','text':'processed'}]
   def _build_messages(self,message,dataset=None):return Qwen3VLChatCDPruner._build_messages(self,message,dataset=dataset)
   def _processor_inputs(self,messages):self.received=messages;return {'input_ids':'synthetic'}
  w=Wrapper();row={'task':'DocVQA_VAL','message':[{'type':'text','value':'raw'}]};ref={'processor':{},'image_start':2}
  with patch.object(real_worker,'validate_processor',return_value='accepted') as validate:
   self.assertEqual(real_worker.preflight_one(torch,w,row,ref),'accepted')
  self.assertEqual(w.dataset,'DocVQA_VAL');self.assertEqual(w.received,[{'role':'user','content':[{'type':'text','text':'processed'}]}]);self.assertNotEqual(w.received,row['message']);validate.assert_called_once()
class InputPreflightTests(unittest.TestCase):
 def test_e1_reference_bound_by_question_and_message(self):
  from input_preflight import bind_e1_references
  rows=[{'task':'T','qid':str(i),'group_id':str(i),'split':'DEV','message_sha256':'m'+str(i),'image_sha256':'i'+str(i)} for i in range(384)]
  records=[dict(task='T',qid=str(i),group_id=str(i),message_sha256='m'+str(i),arm='P3-D0',forced=False,processor={k:{'dtype':'torch.int64','shape':[1],'sha256':'a'*64} for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},image_start=2,answer='not retained') for i in range(384)]
  bound=bind_e1_references(rows,records);self.assertEqual(len(bound),384);self.assertNotIn('answer',bound[('T','0')])
  for bad in (records[:-1],records+[records[0]]):
   with self.assertRaises(ValueError):bind_e1_references(rows,bad)
  bad=copy.deepcopy(records);bad[0]['message_sha256']='wrong'
  with self.assertRaises(ValueError):bind_e1_references(rows,bad)
  bad=copy.deepcopy(records);bad[0]['processor']={}
  with self.assertRaises(ValueError):bind_e1_references(rows,bad)
 def test_unavailable_e1_proof_never_opens_journal(self):
  import input_preflight
  with self.assertRaises((ValueError,FileNotFoundError)):
   input_preflight.load_accepted_e1_references('/tmp/no-e1-proof-xyz','/tmp/no-e1-start-xyz','/tmp/no-e1-end-xyz')
 def test_expected_token_span_and_processor_hashes(self):
  from input_preflight import validate_processor,tensor_meta
  actual={'input_ids':torch.tensor([[1,2]+[999]*1024+[3]]),'attention_mask':torch.ones(1,1027,dtype=torch.long),'pixel_values':torch.ones(2,3),'image_grid_thw':torch.tensor([[1,64,64]])}
  expected={k:tensor_meta(v) for k,v in actual.items()}
  self.assertEqual(validate_processor(torch,actual,expected,999,2)['image_start'],2)
  with self.assertRaisesRegex(ValueError,'span mismatch'):validate_processor(torch,actual,expected,999,3)
  bad=dict(expected);bad['pixel_values']=dict(bad['pixel_values'],sha256='wrong')
  with self.assertRaisesRegex(ValueError,'processor identity'):validate_processor(torch,actual,bad,999,2)
  changed=dict(actual);changed['input_ids']=actual['input_ids'].clone();changed['input_ids'][0,2]=1
  changed_expected={k:tensor_meta(v) for k,v in changed.items()}
  with self.assertRaisesRegex(ValueError,'1024'):validate_processor(torch,changed,changed_expected,999,2)
class NVMLSampleTests(unittest.TestCase):
 def test_proc_identity_requires_exact_start_ticks(self):
  from nvml_sampler import identity_alive
  stat=Path('/proc/self/stat').read_text();start_ticks=int(stat[stat.rfind(')')+2:].split()[19])
  self.assertTrue(identity_alive(os.getpid(),start_ticks));self.assertFalse(identity_alive(os.getpid(),start_ticks+1))
 def test_parse_units_process_and_conflicts(self):
  from nvml_sampler import parse_sample
  sample='GPU-a, 123, 19\nGPU-a, 124, 7\nGPU-b, 123, 8\n'
  used,others=parse_sample(sample,'GPU-a',123)
  self.assertEqual(used,19*1048576);self.assertEqual(others,[{'pid':124,'used_bytes':7*1048576}])
  self.assertEqual(parse_sample(sample,'GPU-c',123),(None,[]))
  for invalid in ('GPU-a, 123, [N/A]','GPU-a, 123, 1\nGPU-a, 123, 2','bad'):
   with self.assertRaises(ValueError):parse_sample(invalid,'GPU-a',123)
class BIdentityTests(unittest.TestCase):
 def test_b_proof_mismatch_and_evidence_tamper_rejected(self):
  import identity_gate
  from unittest.mock import patch
  base=Path('/tmp/fake-eadp-b');b=base/'e4_b_identity';required=[b/'b_001'/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','records.jsonl','generated.jsonl','comparisons.jsonl')]
  files={str(p.resolve()):'h' for p in required};arrays={'/tmp/fake-array-'+str(i):'h' for i in range(8)}
  f={'status':'passed','stage':'skip_drop_identity','natural_pairs':3,'forced_pairs':1,'generations':8,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'verifier_sha256':'h','files_sha256':files,'array_sha256':arrays}
  r={'status':'passed','independent_review':True,'n_generations':8,'natural_pairs':3,'forced_pairs':1,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'formal_proof_sha256':'h','formal_started_sha256':'h','formal_terminal_sha256':'h','reviewer_sha256':'h','files_sha256':files,'array_sha256':arrays}
  with patch.object(identity_gate,'supervised'),patch.object(identity_gate,'sha',return_value='h'),patch.object(identity_gate,'read',side_effect=lambda p:f if Path(p).name=='b_001_verification.json' else r):
   self.assertEqual(identity_gate.validate_b_identity(base)['formal_proof_sha256'],'h')
   bad=copy.deepcopy(r);bad['reviewer_sha256']='wrong'
   with patch.object(identity_gate,'read',side_effect=lambda p:f if Path(p).name=='b_001_verification.json' else bad):
    with self.assertRaises(ValueError):identity_gate.validate_b_identity(base)
   bad=copy.deepcopy(f);bad['array_sha256'].pop(next(iter(bad['array_sha256'])))
   with patch.object(identity_gate,'read',side_effect=lambda p:bad if Path(p).name=='b_001_verification.json' else r):
    with self.assertRaises(ValueError):identity_gate.validate_b_identity(base)
   changed=next(iter(files))
   with patch.object(identity_gate,'sha',side_effect=lambda p:'tampered' if str(p)==changed else 'h'):
    with self.assertRaises(ValueError):identity_gate.validate_b_identity(base)
 def test_rejects_absent_proof_and_keeps_gpu_locked(self):
  import tempfile,identity_gate
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises((FileNotFoundError,ValueError)):identity_gate.validate_b_identity(Path(d))
  with self.assertRaisesRegex(ValueError,'locked'):identity_gate.require_gpu_identity()
class ProcessTests(unittest.TestCase):
 def test_memory_workers_are_distinct_and_cannot_mix_modes(self):
  script=HERE/'runner.py';jobs=[{'mode':'C-ON','order':0},{'mode':'C-ON','order':1}];cmd=[sys.executable,str(script),'--role','memory_isolated','--mode','C-ON','--fake']
  results=[]
  for _ in range(2):
   r=subprocess.run(cmd,input=''.join(json.dumps(x)+'\n' for x in jobs),text=True,capture_output=True,check=True);results.append([json.loads(x) for x in r.stdout.splitlines()])
  self.assertNotEqual(results[0][0]['pid'],results[1][0]['pid']);self.assertEqual({x['pid'] for x in results[0]},{results[0][0]['pid']})
  bad=subprocess.run(cmd,input=json.dumps({'mode':'C-DROP','order':0})+'\n',text=True,capture_output=True);self.assertNotEqual(bad.returncode,0)
 def test_schedule_durable_failure_and_mode_separation(self):
  import tempfile
  import contract,runner
  rows=[{'task':task,'qid':str(i),'group_id':task+str(i),'split':'DEV'} for task in contract.TASKS for i in range(128)]
  schedule=contract.make_schedule(rows)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'partial.jsonl';handle,journal=runner.durable_journal(path);seen=[]
   def execute(row,mode,protocol):
    seen.append((mode,protocol))
    if len(seen)==3:raise RuntimeError('synthetic abort')
    return {'mode':mode,'protocol':protocol}
   with self.assertRaisesRegex(RuntimeError,'synthetic abort'):runner.execute_schedule(schedule,rows,'memory_isolated','C-ON',execute,journal)
   handle.close();events=[json.loads(x) for x in path.read_text().splitlines()]
   self.assertEqual([e['status'] for e in events],['generated','completed','generated','completed','failed']);self.assertEqual({x[0] for x in seen},{'C-ON'})
   with self.assertRaises(FileExistsError):runner.durable_journal(path)
   with self.assertRaises(ValueError):runner.execute_schedule(schedule,rows,'latency_interleaved','C-ON',execute,lambda x:None)
   bad=copy.deepcopy(rows);bad[0]['split']='CONFIRM'
   with self.assertRaises(ValueError):runner.execute_schedule(schedule,bad,'latency_interleaved',None,execute,lambda x:None)
   reordered=copy.deepcopy(schedule);reordered['schedule'][0],reordered['schedule'][1]=reordered['schedule'][1],reordered['schedule'][0]
   for i,item in enumerate(reordered['schedule']):item['order']=i
   with self.assertRaisesRegex(ValueError,'seeded order'):runner.execute_schedule(reordered,rows,'latency_interleaved',None,execute,lambda x:None)
 def test_latency_role_never_enables_memory_probe(self):
  import runner,contract
  from unittest.mock import patch
  rows=[{'task':t,'qid':str(i),'group_id':t+str(i),'split':'DEV','image_start':2} for t in contract.TASKS for i in range(128)]
  schedule=contract.make_schedule(rows);seen=[]
  def fake_execute(wrapper,native,torch,row,mode,protocol,sync):
   seen.append(mode)
   if len(seen)==2:raise RuntimeError('end test')
   return {'mode':mode,'protocol':protocol}
  with patch.object(runner,'execute_one',side_effect=fake_execute),patch('memory_probe.memory_measure',side_effect=AssertionError('memory probe in latency')):
   with self.assertRaisesRegex(RuntimeError,'end test'):runner.run_model_jobs(schedule,rows,'latency_interleaved',None,None,None,torch,lambda x:None)
  self.assertEqual(len(seen),2)
 def test_timing_budget_binds_deadline_and_cleanup_grace(self):
  import hashlib,tempfile
  from timing_launch import validate_budget_window
  with tempfile.TemporaryDirectory() as d:
   source=Path(d)/'budget.json';source.write_text(json.dumps({'budget_start_unix':100,'budget_end_unix':10000}));digest=hashlib.sha256(source.read_bytes()).hexdigest()
   plan={'budget_source':str(source),'budget_source_sha256':digest,'budget_start_unix':100,'budget_end_unix':10000}
   validate_budget_window(plan,200,3600,3720,90)
   for now,inner,outer,grace in [(50,3600,3720,90),(7000,3600,3720,90),(200,3600,3650,90),(200,3600,3720,30)]:
    with self.assertRaises(ValueError):validate_budget_window(plan,now,inner,outer,grace)
   source.write_text('{}')
   with self.assertRaises(ValueError):validate_budget_window(plan,200,3600,3720,90)
 def test_launcher_exit_fail_closed(self):
  from timing_launch import exit_code
  good={'timed_out':False,'interrupted':None,'client_exited_while_container_running':False,'container_stopped_verified':True,'container_identity':{'container_id':'id','host_init':{'pid':1,'start_ticks':2,'boot_id':'boot'}},'gpu_pids_after':[],'container_exit_code':0,'returncode':0}
  self.assertEqual(exit_code(good),0)
  for key,value in [('timed_out',True),('interrupted',15),('client_exited_while_container_running',True),('container_stopped_verified',False),('container_identity',None),('gpu_pids_after',[2]),('container_exit_code',1),('returncode',None)]:
   bad=dict(good);bad[key]=value;self.assertNotEqual(exit_code(bad),0,key)
 def test_launcher_records_subprocess_creation_failure(self):
  import tempfile,timing_launch
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d,patch.object(timing_launch.subprocess,'Popen',side_effect=OSError('synthetic launch failure')):
   status=timing_launch.run_bounded(['docker'],'test',60,lambda:[],lambda pid:{'pid':pid},Path(d))
   result=json.loads((Path(d)/'exit.json').read_text())
   self.assertNotEqual(status,0);self.assertEqual(result['failure_type'],'OSError');self.assertIn('synthetic',result['failure']);self.assertFalse(result['container_stopped_verified'])
 def test_launcher_failed_inspect_records_cleanup_observations(self):
  import tempfile,timing_launch
  from unittest.mock import patch
  class Done:
   def poll(self):return 0
   def wait(self,timeout=None):return 0
  def command(cmd,**kwargs):
   return types.SimpleNamespace(returncode=1 if cmd[1] in ('inspect','stop') else 0,stdout='',stderr='synthetic failure')
  with tempfile.TemporaryDirectory() as d,patch.object(timing_launch.subprocess,'Popen',return_value=Done()),patch.object(timing_launch.subprocess,'run',side_effect=command):
   status=timing_launch.run_bounded(['docker'],'test',60,lambda:[],lambda pid:{'pid':pid},Path(d));result=json.loads((Path(d)/'exit.json').read_text())
   self.assertNotEqual(status,0);self.assertEqual(result['failure_type'],'RuntimeError');self.assertEqual(result['last_inspect']['returncode'],1)
   self.assertEqual([x['action'] for x in result['cleanup_attempts']],['stop','kill'])
 def test_prepared_real_worker_is_locked_before_cuda(self):
  cmd=[sys.executable,str(HERE/'real_worker.py'),'--role','latency_interleaved']
  for name in ('e1-acceptance','e1-started','e1-terminal','out'):cmd.extend(['--'+name,'/tmp/none'])
  proc=subprocess.run(cmd,text=True,capture_output=True)
  self.assertNotEqual(proc.returncode,0);self.assertIn('real timing locked',proc.stderr)
 def test_real_path_fails_before_torch_import_without_identity(self):
  cmd=[sys.executable,str(HERE/'runner.py'),'--role','latency_interleaved'];r=subprocess.run(cmd,input='',text=True,capture_output=True)
  self.assertNotEqual(r.returncode,0);self.assertIn('real timing locked',r.stderr)
if __name__=='__main__':unittest.main()
