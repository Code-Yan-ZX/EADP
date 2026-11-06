"""Post-hoc synthetic test: restrict recomputation fusion to original prompt."""
import argparse,contextlib,json,sys,traceback
from pathlib import Path
import torch,transformers
from transformers import Qwen3VLTextConfig
D=Path(__file__).resolve().parent;B=D.parents[1];P=B/'e6/hilo_local_probe'

@contextlib.contextmanager
def prompt_only_fusion(model,prompt_length):
    """Diagnostic reference only; never changes the source model package."""
    saved=[];counts={'calls':0,'suppressed_positions':0}
    try:
        for layer in model.layers:
            module=layer.self_attn;old=module._compute_selective_boost;had='_compute_selective_boost' in module.__dict__;raw=module.__dict__.get('_compute_selective_boost')
            def wrapped(*args,_old=old,**kwargs):
                out=_old(*args,**kwargs)
                if out is not None and out.shape[1]>prompt_length:
                    out=out.clone();counts['calls']+=1;counts['suppressed_positions']+=out.shape[1]-prompt_length;out[:,prompt_length:]=0
                return out
            module._compute_selective_boost=wrapped;saved.append((module,had,raw))
        yield counts
    finally:
        for module,had,raw in saved:
            if had:module._compute_selective_boost=raw
            else:delattr(module,'_compute_selective_boost')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();sys.path.insert(0,str(P))
    import hilo_prune.models.qwen3_vl.modeling_hilo_qwen3_vl as mod
    assert Path(mod.__file__).resolve()==P/'hilo_prune/models/qwen3_vl/modeling_hilo_qwen3_vl.py' and not torch.cuda.is_initialized()
    torch.set_num_threads(1);torch.manual_seed(20261105)
    cfg=Qwen3VLTextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=4,head_dim=8,use_cache=True)
    cfg.rope_parameters={'rope_type':'default','rope_theta':5000000.0,'mrope_section':[2,1,1]};cfg._attn_implementation='eager';cfg.hilo_keep_ratio=.5;cfg.hilo_candidate_ratio=.75;cfg.hilo_prune_layer=2
    model=mod.Qwen3VLTextModel(cfg).eval();embeds=torch.randn(1,8,32);ids=torch.tensor([[1,9,9,9,9,9,9,2]])
    deep=[torch.randn(6,32) for _ in range(3)];patches=torch.randn(6,12);tokens=torch.randn(1,32,32)
    result={'status':'failed','scope':'post-hoc random CPU synthetic model; fused outputs on generated positions suppressed in recomputation only',
            'torch':torch.__version__,'transformers':transformers.__version__,'steps_planned':32,'atol':1e-5,'rtol':1e-5,'steps':[],
            'GT_opened':False,'model_weights_loaded':False,'gpu_used':False,
            'limitations':['This constructs a diagnostic reference to match prefill-only fusion; it does not establish author intent.',
                           'Random tiny eager model with simple default positions, not real Qwen3 3-axis/cache identity.',
                           'Post-hoc test after seeing v2 discrepancies; no quality or latency conclusion.']}
    try:
        with torch.inference_mode():
            first=model(inputs_embeds=embeds.clone(),input_ids_=ids,image_token_id=9,image_tensor=patches,visual_pos_masks=ids.eq(9),deepstack_visual_embeds=deep,use_cache=True)
            cache=first.past_key_values;result['prefill_lengths']=[cache.get_seq_length(i) for i in range(4)]
            for i in range(32):
                cached=model(inputs_embeds=tokens[:,i:i+1],past_key_values=cache,use_cache=True);cache=cached.past_key_values
                full=torch.cat((embeds,tokens[:,:i+1]),dim=1);fullids=torch.cat((ids,torch.full((1,i+1),2,dtype=ids.dtype)),dim=1)
                with prompt_only_fusion(model,8) as counts:
                    ref=model(inputs_embeds=full,input_ids_=fullids,image_token_id=9,image_tensor=patches,visual_pos_masks=fullids.eq(9),deepstack_visual_embeds=deep,use_cache=False)
                x=cached.last_hidden_state[:,-1,:];y=ref.last_hidden_state[:,-1,:]
                result['steps'].append({'step':i+1,'cache_lengths':[cache.get_seq_length(j) for j in range(4)],'max_abs_error':float((x-y).abs().max()),'allclose':bool(torch.allclose(x,y,atol=1e-5,rtol=1e-5)),**counts})
        result['status']='completed';result['all_steps_match']=all(x['allclose'] for x in result['steps'])
    except Exception as e:result['failure']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
    result['cuda_initialized']=torch.cuda.is_initialized()
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
