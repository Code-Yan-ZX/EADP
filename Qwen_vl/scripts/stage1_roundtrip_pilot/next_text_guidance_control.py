"""Opt-in correctness controls for released EADP CLIP text guidance.

These are separate experimental protocols, not a claim to recover the authors'
unpublished runtime. Production sources and existing results stay untouched.
No alpha, beta, q, temperature, visual budget, or selection rule is tuned.
"""
from __future__ import annotations

import types
import torch


def bounded_clip_chunks(ids, attention_mask, bos_id, eos_id, max_length=77):
    """Preserve every content token; give every chunk its own BOS and EOS.

    Returns the flattened [1, M*77] form consumed by the released forward.
    This evaluator accepts one prompt, matching its single-image batch policy.
    """
    if ids.ndim != 2 or ids.shape[0] != 1 or attention_mask.shape != ids.shape:
        raise ValueError('Expected a single prompt with aligned attention mask')
    if max_length < 3:
        raise ValueError('CLIP context must hold BOS, content and EOS')
    tokens = ids[0][attention_mask[0].bool()]
    if tokens.numel() < 2 or int(tokens[0]) != bos_id or int(tokens[-1]) != eos_id:
        raise ValueError('Expected original tokenizer BOS/content/EOS sequence')
    content = tokens[1:-1]
    parts = list(content.split(max_length-2)) if content.numel() else [content]
    chunks = ids.new_zeros((len(parts),max_length))
    masks = attention_mask.new_zeros(chunks.shape)
    for i, part in enumerate(parts):
        chunks[i,0] = bos_id
        chunks[i,1:1+len(part)] = part
        chunks[i,1+len(part)] = eos_id
        masks[i,:2+len(part)] = 1
    return dict(input_ids=chunks.reshape(1,-1), attention_mask=masks.reshape(1,-1))


class BoundedCLIPTokenizer:
    def __init__(self, original, max_length):
        self.original = original
        self.max_length = max_length

    def __getattr__(self, name):
        return getattr(self.original,name)

    def __call__(self, *args, **kwargs):
        raw = self.original(*args,**kwargs)
        from transformers import BatchEncoding
        return BatchEncoding(bounded_clip_chunks(raw.input_ids, raw.attention_mask,
                             self.bos_token_id,self.eos_token_id,self.max_length))


def valid_entropy_filter(local, valid, T=100., entropy_keep_ratio=.2):
    """Use the original entropy and q rule on valid content positions only.

    Ragged selections are padded with zero similarity / infinite entropy,
    which gives exactly zero aggregation weight. An empty segment is handled
    by the installer's explicit global-guidance fallback.
    """
    if valid.shape != local.shape[2:]:
        raise ValueError('Content mask does not align with CLIP segments')
    if not 0 < entropy_keep_ratio <= 1:
        raise ValueError('q must be in (0, 1]')
    probs = torch.softmax(local*T,dim=1)
    entropy = -(probs*torch.log(probs+1e-12)).sum(dim=1)
    entropy = entropy.masked_fill(~valid.unsqueeze(0),float('inf'))
    counts = valid.sum(-1)
    quotas = (counts*entropy_keep_ratio).long().clamp(min=1)
    max_keep = int(quotas.max())
    values, indices = entropy.topk(max_keep,dim=-1,largest=False)
    active = (torch.arange(max_keep,device=local.device)[None,:] < quotas[:,None]) & (counts[:,None] > 0)
    values = values.masked_fill(~active.unsqueeze(0),float('inf'))
    gathered = local.gather(-1,indices.unsqueeze(1).expand(-1,local.shape[1],-1,-1))
    gathered = gathered.masked_fill(~active[None,None,:,:],0.)
    empty = counts == 0
    # Avoid softmax(-inf, ... -inf); aggregation is replaced by global below.
    values = torch.where(empty[None,:,None],torch.zeros_like(values),values)
    return gathered, values


def install(model, *, mask_content=False, bounded_chunks=False):
    """Install on one loaded EADP model and return a complete restore callback.

    The hook records text metadata in its producing stream. The original
    forward synchronizes both streams before any pruning consumes this mask.
    Use the independently validated caller-stream wait repair as well.
    """
    tower = model.get_vision_tower()
    tokenizer = tower.text_tokenizer
    previous = {name:(name in model.__dict__,getattr(model,name)) for name in
                ['entrpy_filter','local_aggregation','sim_cross']}
    state = {}
    if bounded_chunks:
        tower.text_tokenizer = BoundedCLIPTokenizer(tokenizer,tower.max_position_embeddings)
    hook = None
    if mask_content:
        def capture(module,args,kwargs,output):
            ids = kwargs['input_ids']
            valid = kwargs['attention_mask'].bool()
            state['valid'] = valid & (ids != tokenizer.bos_token_id) & (ids != tokenizer.eos_token_id)
        hook = tower.text_tower.register_forward_hook(capture,with_kwargs=True)

        def cross(self,*args,**kwargs):
            returned = previous['sim_cross'][1](*args,**kwargs)
            state['global'] = returned[0]
            return returned

        def filtering(self,local_sim_all,T=100.,entropy_keep_ratio=.5):
            return valid_entropy_filter(local_sim_all,state['valid'],T,entropy_keep_ratio)

        def aggregate(self,*args,**kwargs):
            local = previous['local_aggregation'][1](*args,**kwargs)
            empty = ~state['valid'].any(-1)
            return torch.where(empty[None,None,:],state['global'],local)

        model.sim_cross = types.MethodType(cross,model)
        model.entrpy_filter = types.MethodType(filtering,model)
        model.local_aggregation = types.MethodType(aggregate,model)

    def restore():
        if hook is not None:
            hook.remove()
        tower.text_tokenizer = tokenizer
        for name,(had_local,original) in previous.items():
            if had_local:
                setattr(model,name,original)
            elif name in model.__dict__:
                delattr(model,name)
    return restore
