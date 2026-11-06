"""Evidence-only wrapper for the explicit user-text selection adapter."""
import hashlib
from contextlib import contextmanager
from types import MethodType
import torch
from selector import user_text_selector


def digest(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


@contextmanager
def traced_selector(model,expected_ids,positions):
    trace={'calls':0,'input_ids_sha256':digest(expected_ids),'positions_sha256':digest(positions),
           'selected_ids_sha256':digest(expected_ids[0,positions]),'returned_embedding_sha256':None,
           'returned_exact_selected_embeddings':None}
    with user_text_selector(model,expected_ids,positions):
        original=model._extract_query_text_embeds
        def counted(_self,embeds,input_ids,mode=None):
            out=original(embeds,input_ids,mode)
            expected=embeds[0,positions]
            trace['calls']+=1
            trace['returned_exact_selected_embeddings']=bool(torch.equal(out,expected))
            trace['returned_embedding_sha256']=digest(out)
            return out
        model._extract_query_text_embeds=MethodType(counted,model)
        try:yield trace
        finally:model._extract_query_text_embeds=original
