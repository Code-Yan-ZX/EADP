"""Three controlled random-CPU tests of cached-vs-recomputed decoder prefixes."""
import argparse,json,sys,traceback
from pathlib import Path
import torch,transformers
from transformers import Qwen3VLTextConfig

D=Path(__file__).resolve().parent;B=D.parents[1];P=B/'e6/hilo_local_probe'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();sys.path.insert(0,str(P))
    import hilo_prune.models.qwen3_vl.modeling_hilo_qwen3_vl as mod
    assert Path(mod.__file__).resolve()==P/'hilo_prune/models/qwen3_vl/modeling_hilo_qwen3_vl.py' and not torch.cuda.is_initialized()
    torch.set_num_threads(1);torch.manual_seed(20261105)
    cfg=Qwen3VLTextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=4,head_dim=8,use_cache=True)
    cfg.rope_parameters={'rope_type':'default','rope_theta':5000000.0,'mrope_section':[2,1,1]};cfg._attn_implementation='eager';cfg.hilo_keep_ratio=.5;cfg.hilo_candidate_ratio=.75;cfg.hilo_prune_layer=2
    model=mod.Qwen3VLTextModel(cfg).eval();embeds=torch.randn(1,8,32);ids=torch.tensor([[1,9,9,9,9,9,9,2]])
    deep=[torch.randn(6,32) for _ in range(3)];patches=torch.randn(6,12);tokens=torch.randn(1,32,32);defaults=[l.self_attn.use_pre_prune_fusion for l in model.layers]
    result={'scope':'random 4-layer CPU model, eager, synthetic default positions','torch':torch.__version__,'transformers':transformers.__version__,
            'steps_planned':32,'atol':1e-5,'rtol':1e-5,'GT_opened':False,'model_weights_loaded':False,'gpu_used':False,'cases':[],
            'limitations':['Synthetic simple positions do not establish Qwen3 3-axis correctness.',
                           'Full-prefix recomputation changes PA-Fusion execution in comparison to cache decode, so a difference alone does not establish an incorrect author implementation.',
                           'Both paths use the same local repair, not the original author fixed environment.']}
    for name,selected,fusion in (('no_selection',False,True),('selection_fusion_disabled',True,False),('selection_fusion_enabled',True,True)):
        case={'name':name,'selected':selected,'fusion_enabled':fusion,'steps':[],'status':'failed'}
        try:
            for layer,default in zip(model.layers,defaults):layer.self_attn.use_pre_prune_fusion=default if fusion else False
            visual=patches if selected else None
            with torch.inference_mode():
                first=model(inputs_embeds=embeds.clone(),input_ids_=ids,image_token_id=9,image_tensor=visual,visual_pos_masks=ids.eq(9),deepstack_visual_embeds=deep,use_cache=True)
                cache=first.past_key_values;case['prefill_lengths']=[cache.get_seq_length(i) for i in range(4)]
                for i in range(32):
                    cached=model(inputs_embeds=tokens[:,i:i+1],past_key_values=cache,use_cache=True);cache=cached.past_key_values
                    full=torch.cat((embeds,tokens[:,:i+1]),dim=1);fullids=torch.cat((ids,torch.full((1,i+1),2,dtype=ids.dtype)),dim=1)
                    ref=model(inputs_embeds=full,input_ids_=fullids,image_token_id=9,image_tensor=visual,visual_pos_masks=fullids.eq(9),deepstack_visual_embeds=deep,use_cache=False)
                    x=cached.last_hidden_state[:,-1,:];y=ref.last_hidden_state[:,-1,:]
                    case['steps'].append({'step':i+1,'cache_lengths':[cache.get_seq_length(j) for j in range(4)],'max_abs_error':float((x-y).abs().max()),'allclose':bool(torch.allclose(x,y,atol=1e-5,rtol=1e-5))})
            case['status']='completed';case['all_steps_match']=all(x['allclose'] for x in case['steps'])
        except Exception as e:case['failure']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
        result['cases'].append(case)
    result['cuda_initialized']=torch.cuda.is_initialized()
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
