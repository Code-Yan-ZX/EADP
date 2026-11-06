"""Exercise the actual 96-image worker main with CPU-only fake dependencies."""
import json,os,sys,tempfile,types,unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import torch
import worker
from trials import measurement_order
from case_contract import partition
from gates import SYNTHETIC

class FakeWrapper:
 def __init__(self,**kwargs):self.model=types.SimpleNamespace(dtype=torch.bfloat16,config=types.SimpleNamespace(_attn_implementation='flash_attention_2'))
 def _tokenize_instruction(self,*args,**kwargs):return torch.tensor([[1,2]])
class FakeCapture:
 def __init__(self,*args):self.state={}

def synthetic_rows():
 ks=(4,4,4,8,1,8);return [{'case':c,'k':k,'author_ids':list(range(k)),'fused_ids':list(range(k)),'graph_ids':list(range(k)),'fused_equal':True,'graph_equal':True} for c,k in zip(SYNTHETIC,ks)]+[{'case':'graph_changed_input','k':4,'author_ids':[1,0,2,3],'graph_ids':[1,0,2,3],'before_graph_ids':[0,1,2,3],'graph_equal':True,'different_from_original':True}]
class Tests(unittest.TestCase):
 def run_worker(self,fail_at=None):
  with tempfile.TemporaryDirectory() as tmp:
   base=Path(tmp);here=base/'e5/selector_real128';here.mkdir(parents=True);(base/'e4_c_microbench').mkdir();out=here/'run_001';out.mkdir();rows=[{'task':'D','qid':str(i),'group_id':'g'+str(i),'message_sha256':'m','image_sha256':'i','image_start':4,'message':[]} for i in range(96)];keys=('task','qid','group_id','message_sha256','image_sha256');pilot=[{k:r[k] for k in keys} for r in rows];(here/'sample_manifest_001.json').write_text(json.dumps({'pilot':pilot,'full_128':[{}]*128,'budgets':[128,256,512]}));(base/'e4_c_microbench/schedule001.json').write_text(json.dumps({'input_identities':pilot}));approval={'source_sha256':{},'prior_identity':{'e4':{}}};(out/'launch.json').write_text(json.dumps({'approval_bundle':approval,'worker_sha256':'hash'}));prepared={'plan':{'recipe':{}},'rows':rows,'refs':{},'weights_before':{}};native=types.SimpleNamespace();calls={'cases':0}
   def fake_module(path,name):
    if str(path).endswith('/capture.py'):return types.SimpleNamespace(CaptureInputs=FakeCapture,clone_tree=lambda t,x:x)
    if str(path).endswith('/light_adapter.py'):return types.SimpleNamespace(LightAdapter=object)
    if str(path).endswith('/kernel.py'):return types.ModuleType('kernel')
    if str(path).endswith('/graph.py'):return types.SimpleNamespace(GraphSelector=object)
    raise AssertionError(path)
   def fake_capture(t,w,n,row,author,adapter,Capture,clone,journal):
    cap=Capture();v=torch.ones(1);cap.state={'selector':{'args':(v,torch.ones(1),torch.ones(1),torch.ones(1)),'kwargs':{}}};journal({'status':'capture_generated','task':row['task'],'qid':row['qid'],'group_id':row['group_id'],'output_ids':[[1]]});journal({'status':'capture_cleanup','passed':True,'capture_completed':True});actual={'processor':{k:torch.ones(1) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},'visual':(v,[v,v,v]),'ordered':torch.arange(256)};return {},{'selector':v},actual,lambda:True
   def fake_author(t,n,args,k,meta,record,tensor_meta):
    calls['cases']+=1
    if calls['cases']==fail_at:raise RuntimeError('injected case failure')
    result={x:True for x in ('e4_common_metadata_equal','e4_order_equal','e4_spatial_equal','e4_pruned_hash_equal')};return result,torch.ones(1),torch.ones(1),torch.arange(k)[None,:],torch.ones(1)
   def fake_bench(t,n,kernel,graph_cls,graphs,sim,imp,k,expected,journal,allowed):
    ms=[]
    for rep in range(5):
     for name in measurement_order(rep):
      m={'repeat':rep,'implementation':name,'status':'measured','gpu_event_ms':1.,'host_wall_ms':1.,'equals_author':True,'ordered_ids':expected.tolist()};ms.append(m);journal({'status':'case_measurement','budget':k,'measurement':m})
    ids=expected.tolist();return {'author_equals_endpoint':True,'fused_equals_author':True,'graph_equals_author':True,'fused_ordered_ids':ids,'graph_ordered_ids':ids},ms,None
   def fake_archive(t,path,obj):path.write_bytes(b'fake');return 'hash'
   records={(r['task'],r['qid'],k):{} for r in rows for k in (128,256,512)};mod=types.ModuleType('vlmeval.vlm.qwen3_vl.model_fixed_res');mod.Qwen3VLChatCDPruner=types.SimpleNamespace(generate_inner_transformers_fixed=lambda *a,**k:None)
   packages={x:types.ModuleType(x) for x in ('vlmeval','vlmeval.vlm','vlmeval.vlm.qwen3_vl')};packages['vlmeval.vlm.qwen3_vl.model_fixed_res']=mod
   for name in ('vlmeval','vlmeval.vlm','vlmeval.vlm.qwen3_vl'):packages[name].__path__=[]
   patches=[patch.dict(sys.modules,packages),patch.object(worker,'HERE',here),patch.object(worker,'BASE',base),patch.dict(os.environ,{'TRITON_CACHE_DIR':str(here/'triton_cache')}),patch.object(worker,'admit',return_value=approval),patch.object(worker,'sources',return_value=[]),patch.object(worker,'sha',return_value='hash'),patch.object(worker,'load_manifest',return_value=({'budgets':[128,256,512]},pilot)),patch.object(worker,'prepare',return_value=prepared),patch.object(worker,'bind_rows',return_value=rows),patch.object(worker,'get_records',return_value=records),patch.object(worker,'load_runtime',return_value=(torch,native,FakeWrapper,{})),patch.object(worker,'module',side_effect=fake_module),patch.object(worker,'preflight_one'),patch.object(worker,'get_reference',return_value={}),patch.object(worker,'capture_one',side_effect=fake_capture),patch.object(worker,'validate_identity',return_value={'status':'passed'}),patch.object(worker,'compare_real',return_value={'sequence_equal':True,'mean_equal':True}),patch.object(worker,'tensor_meta',return_value={'fake':True}),patch.object(worker,'author_case',side_effect=fake_author),patch.object(worker,'bench',side_effect=fake_bench),patch.object(worker,'archive',side_effect=fake_archive),patch.object(worker,'synthetic',return_value=synthetic_rows()),patch.object(torch.cuda,'is_initialized',return_value=False),patch.object(torch.cuda,'get_device_name',return_value='NVIDIA A40'),patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'max_memory_allocated',return_value=0),patch.object(sys,'argv',['worker.py','--out',str(out)])]
   with ExitStack() as stack:
    for context in patches:stack.enter_context(context)
    if fail_at:
     with self.assertRaisesRegex(RuntimeError,'injected case failure'):worker.main()
     self.assertFalse((out/'worker_complete.json').exists());events=[json.loads(x) for x in (out/'journal.jsonl').read_text().splitlines()];self.assertTrue(any(x['status']=='capture_generated' for x in events));self.assertTrue(any(x['status']=='case_completed' for x in events));return
    worker.main();done=json.loads((out/'worker_complete.json').read_text());events=[json.loads(x) for x in (out/'journal.jsonl').read_text().splitlines()];self.assertEqual((done['images'],done['cases']),(96,288));self.assertEqual(len(partition(events,pilot)),96);self.assertEqual(sum(x['status']=='capture_generated' for x in events),96)
 def test_complete_real_main(self):self.run_worker()
 def test_midrun_failure_retains_exposure_without_complete(self):self.run_worker(fail_at=5)
if __name__=='__main__':unittest.main()
