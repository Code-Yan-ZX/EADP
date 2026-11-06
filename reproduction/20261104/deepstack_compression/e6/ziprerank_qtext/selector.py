"""Explicit user-text adapter for ZipRerank, preserving the original prompt.

This is a local adaptation: it selects all tokens wholly inside the original
user text message (including any answer instructions/options) for QI-EI.  It
does not change the model input, reorder the prompt, or use reference answers.
"""
from contextlib import contextmanager
from types import MethodType


def locate_user_text(processor, messages, user_text, actual_ids, image_token_id):
    """Validate complete image expansion and return exact text token positions."""
    import torch
    if actual_ids.ndim != 2 or actual_ids.shape[0] != 1:
        raise ValueError('expected batch size one')
    rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    if not user_text or rendered.count(user_text)!=1: raise ValueError('ambiguous user text')
    start=rendered.index(user_text); end=start+len(user_text)
    encoded=processor.tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded['input_ids']; offsets=encoded['offset_mapping']
    image=[i for i,x in enumerate(ids) if x==image_token_id]
    if len(image)!=1: raise ValueError('expected one image placeholder')
    actual=actual_ids[0].tolist(); visual=[i for i,x in enumerate(actual) if x==image_token_id]
    if len(visual)!=1024 or visual!=list(range(visual[0],visual[0]+1024)):
        raise ValueError('expected contiguous 1024 visual tokens')
    expanded=ids[:image[0]]+[image_token_id]*1024+ids[image[0]+1:]
    if actual!=expanded: raise ValueError('full tokenizer/processor IDs differ')
    selected=[i for i,(a,b) in enumerate(offsets) if b>a and a<end and b>start]
    if (not selected or selected!=list(range(selected[0],selected[-1]+1)) or selected[0]<=image[0]
            or any(offsets[i][0]<start or offsets[i][1]>end for i in selected)):
        raise ValueError('user text has invalid token boundaries')
    result=torch.tensor([i+1023 for i in selected],dtype=torch.long,device=actual_ids.device)
    if any(actual[int(i)]!=ids[j] for i,j in zip(result.tolist(),selected)):
        raise ValueError('selected token ID mismatch')
    return result


@contextmanager
def user_text_selector(model, expected_ids, positions):
    """Temporarily change only QI text embedding extraction for one input."""
    import torch
    if expected_ids.ndim!=2 or expected_ids.shape[0]!=1 or positions.ndim!=1 or not positions.numel():
        raise ValueError('invalid selector inputs')
    if positions.min().item()<0 or positions.max().item()>=expected_ids.shape[1]:
        raise ValueError('selector positions out of range')
    if positions.tolist()!=sorted(set(positions.tolist())):
        raise ValueError('selector positions not unique and ordered')
    original_instance=model.__dict__.get('_extract_query_text_embeds',None)
    had_instance='_extract_query_text_embeds' in model.__dict__
    expected=expected_ids.detach().clone(); selected=positions.detach().clone()
    def extract(_self, embeds, input_ids, mode=None):
        if input_ids is None or not torch.equal(input_ids,expected) or embeds.ndim!=3 or embeds.shape[:2]!=expected.shape:
            raise ValueError('selector input changed')
        return embeds[0,selected]
    model._extract_query_text_embeds=MethodType(extract,model)
    try:
        yield
    finally:
        if had_instance: model._extract_query_text_embeds=original_instance
        else: delattr(model,'_extract_query_text_embeds')
