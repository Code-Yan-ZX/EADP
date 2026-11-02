"""Exercise actual worker capture/serialization against saved fixed16 tensors."""
import ast,hashlib,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import common as c
from contracts import mask_span
from verify_generation import check_observation,check_anchor

class Tests(unittest.TestCase):
 def test_real_worker_capture_and_fixed16_independent_anchor(self):
  old=c.read(c.OLD/'run_plan.json');model=Path(old['recipe']['model_path']);config=c.read(model/'config.json')
  generation=c.read(model/'generation_config.json');eos=generation['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
  item=c.read(c.FIXED/'plan.json')['panel'][0];tree=ast.parse((c.HERE/'worker.py').read_text())
  main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
  wanted={'save_line','array_value','meta','collect_arrays','capture'}
  nodes=[n for n in main.body if isinstance(n,ast.FunctionDef) and n.name in wanted];assert len(nodes)==len(wanted)
  for arm in ('ds_off','ds_on'):
   record=c.read(c.FIXED/'execution'/f'00_{arm}.json');path=c.FIXED/'execution'/f'00_{arm}.npz'
   with np.load(path,allow_pickle=False) as z:
    def tensor(k):
     t=torch.from_numpy(z[k].copy());dtype=record['arrays'][k]['dtype']
     if dtype=='torch.bfloat16':t=t.view(torch.bfloat16)
     assert str(t.dtype)==dtype;return t
    def field(label):
     d={k:tensor(label+'_'+k) if label+'_'+k in z else None for k in ('inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks')}
     deep=[tensor(label+'_deepstack_'+str(j)) for j in range(3) if label+'_deepstack_'+str(j) in z]
     d['deepstack_visual_embeds']=deep or None;return d
    rec={k:record[k] for k in ('response','native_failure','selection_errors','vision_calls','pruner_calls','processor_calls','generate_calls','llm_calls','selection_calls','intervention_attempts','intervention_applied','intervention_error')}
    rec['errors']=[];rec['vision']=(tensor('visual_features'),[tensor('visual_deepstack_'+str(j)) for j in range(3)])
    rec['selection_results']=[dict(ordered_ids=tensor('ordered_ids'),quotas=tensor('quotas'))];rec['split_sizes']=[256]
    for k in ('pruner_input','pruner_output','original_token_embeddings','rope_deltas','output_ids'):rec[k]=tensor(k)
    rec['processor_inputs']={k:tensor('processor_'+k) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
    rec['generation_kwargs']={**record['generate_non_tensor_kwargs'],**{k:tensor('generate_'+k) for k in ('inputs_embeds','attention_mask')}}
    for label in ('original_prefill','proposed_prefill','final_prefill'):rec[label]=field(label)
    rec['continuation']=[{**{k:tensor('continuation_%04d_%s'%(i,k)) for k in ('position_ids','cache_position','rope_deltas')},**presence} for i,presence in enumerate(record['continuation_presence'])]
   wrapper=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(image_token_id=config['image_token_id'])),processor=SimpleNamespace(tokenizer=SimpleNamespace(batch_decode=lambda *a,**kw:[record['response']])))
   with tempfile.TemporaryDirectory() as d:
    here=Path(d);anchors=here/'execution'/'anchors';anchors.mkdir(parents=True)
    streams={arm:(io.StringIO(),io.StringIO())}
    def fake_run(torch_,wrapper_,native_,method_,item_,arm_,base_,preserve):preserve(item_,arm_,rec['response'],None);return rec
    scope=dict(torch=torch,np=np,c=c,hashlib=hashlib,json=json,mask_span=mask_span,streams=streams,wrapper=wrapper,native=None,Qwen3VLChatCDPruner=SimpleNamespace(generate_inner_transformers_fixed=None),run_arm=fake_run,baseline_capture=lambda *_:None,eos=eos,anchors=anchors)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(c.HERE/'worker.py'),'exec'),scope)
    scope['capture'](item,arm,None,0)
    pred=json.loads(streams[arm][0].getvalue());obs=json.loads(streams[arm][1].getvalue())
    check_observation(item,pred,obs,arm,config['image_token_id'],config['text_config']['vocab_size'])
    key='00_'+arm;record_path=anchors/(key+'.json');archive_path=anchors/(key+'.npz')
    expected=dict(records={key:c.sha(record_path)},archives={key:c.sha(archive_path)})
    with patch.object(c,'HERE',here):check_anchor(c,0,arm,pred,obs,expected)
if __name__=='__main__':unittest.main()
