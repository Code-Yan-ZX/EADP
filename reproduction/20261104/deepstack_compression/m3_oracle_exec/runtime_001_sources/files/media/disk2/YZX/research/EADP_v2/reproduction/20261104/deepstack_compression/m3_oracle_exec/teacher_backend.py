"""Actual four-stream P3 teacher computations for a pre-verified live model.

This module does not open datasets, initialize a model, generate answers or
admit a GPU run. Its caller must bind the input/visual/pruner identity and
run-time supervisor before it can be used on real data.
"""
import math
import torch
import baseline

def utilities(teacher,saliency,model,ids,start,streams,positions,delta,target):
    if torch.is_inference_mode_enabled() or len(streams)!=4:raise ValueError('invalid full-stream autograd context')
    loss,leaves,grads=teacher.loss_and_gradients(model,ids,start,streams[0],streams[1:],positions,delta,target,expected_visual_count=1024)
    if len(leaves)!=4 or len(grads)!=4 or any(not torch.count_nonzero(g) for g in grads):raise ValueError('missing/nonzero gradient')
    evidence={'features':[baseline.meta(x) for x in leaves],'gradients':[baseline.meta(x) for x in grads],
              'gradient_nonzero_counts':[int(torch.count_nonzero(x).item()) for x in grads]}
    return float(loss.item()),saliency.utilities(leaves,grads),evidence

def scorer(teacher,contract,model,ids,start,streams,positions,delta,target):
    if len(streams)!=4:raise ValueError('four full streams required')
    def evaluate(spatial_ids):
        compact,pos,adjusted,retained=contract.compact_prompt(ids,positions,start,1024,spatial_ids,delta)
        index=torch.tensor(spatial_ids,device=streams[0].device,dtype=torch.long)
        selected=[s.index_select(0,index) for s in streams]
        result=float(teacher.loss_only(model,compact,start,selected[0],selected[1:],pos,adjusted,target,expected_visual_count=256).item())
        if not math.isfinite(result):raise ValueError('nonfinite loss')
        return result
    return evaluate
