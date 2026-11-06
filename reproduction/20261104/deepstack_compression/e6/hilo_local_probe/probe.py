"""Tiny random CPU model only; compare fixed upstream and explicit local repair."""
import argparse,hashlib,json,sys,traceback
from pathlib import Path
import torch,transformers
from transformers import Qwen3VLTextConfig

D=Path(__file__).resolve().parent;B=D.parents[1]
P={'upstream':B/'e5/Hi-Lo_Prune','local':D}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=P,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    sys.path.insert(0,str(P[a.variant]));import hilo_prune.models.qwen3_vl.modeling_hilo_qwen3_vl as mod
    assert Path(mod.__file__).resolve()==P[a.variant]/'hilo_prune/models/qwen3_vl/modeling_hilo_qwen3_vl.py'
    assert not torch.cuda.is_initialized();torch.set_num_threads(1);torch.manual_seed(0)
    cfg=Qwen3VLTextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=4,head_dim=8,use_cache=False)
    cfg.rope_parameters={'rope_type':'default','rope_theta':5000000.0,'mrope_section':[2,1,1]};cfg._attn_implementation='eager'
    cfg.hilo_keep_ratio=.5;cfg.hilo_candidate_ratio=.75;cfg.hilo_prune_layer=2
    model=mod.Qwen3VLTextModel(cfg).eval();embeds=torch.randn(1,8,32);ids=torch.tensor([[1,9,9,9,9,9,9,2]])
    mask=ids.eq(9);deep=[torch.randn(6,32) for _ in range(3)];patches=torch.randn(6,12)
    records=[]
    for selected,use_cache in ((False,False),(True,False),(True,True)):
        result={'selection':selected,'use_cache':use_cache}
        try:
            with torch.inference_mode():
                output=model(inputs_embeds=embeds.clone(),input_ids_=ids,image_token_id=9,image_tensor=patches if selected else None,
                             visual_pos_masks=mask,deepstack_visual_embeds=deep,use_cache=use_cache)
            data=output.last_hidden_state.detach().contiguous().view(torch.uint8).numpy().tobytes()
            result.update(status='passed',shape=list(output.last_hidden_state.shape),sha256=hashlib.sha256(data).hexdigest())
            if use_cache:
                cache=output.past_key_values;result['prefill_layer_lengths']=[cache.get_seq_length(i) for i in range(4)]
                try:
                    with torch.inference_mode():
                        nxt=model(inputs_embeds=torch.zeros((1,1,32)),past_key_values=cache,use_cache=True)
                    result['decode']={'status':'passed','shape':list(nxt.last_hidden_state.shape),'layer_lengths':[nxt.past_key_values.get_seq_length(i) for i in range(4)]}
                except Exception as exc:
                    result['decode']={'status':'failed','error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
        except Exception as exc:
            result.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        records.append(result)
    report={'variant':a.variant,'module':str(Path(mod.__file__).resolve()),'torch':torch.__version__,'transformers':transformers.__version__,
            'scope':'random tiny CPU model, not author fixed environment or full Qwen3 inference','cases':records,
            'GT_opened':False,'model_weights_loaded':False,'gpu_used':False,'cuda_initialized':torch.cuda.is_initialized()}
    with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
if __name__=='__main__':main()
