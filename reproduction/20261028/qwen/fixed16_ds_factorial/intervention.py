"""One prefill-only DeepStack toggle; preserve all other native arguments."""
ARMS=('plain','eadp1024','plain_ds_off','eadp1024_ds_on')

def compatible(mask,deep,embeds):
    if mask is None or deep is None or len(deep)!=3:return 'missing mask or three native DeepStack tensors'
    if tuple(mask.shape)!=tuple(embeds.shape[:2]):return 'mask shape mismatch'
    if str(mask.dtype)!='torch.bool' or mask.device!=embeds.device:return 'mask dtype/device mismatch'
    # This synchronization observes real GPU mask values before the prefill.
    # It is an explicit safety check, not part of author inference or timing.
    count=int(mask.sum().item())
    if any(len(x.shape)!=2 or x.shape[0]!=count or x.shape[1]!=embeds.shape[2] for x in deep):return 'mask cardinality/DeepStack shape mismatch'
    if any(x.device!=embeds.device or x.dtype!=embeds.dtype for x in deep):return 'DeepStack dtype/device mismatch'
    return None

def change(arm, kwargs, image_mask, deepstack):
    """Return (kwargs, applied, error) without mutating original or raising.

    If actual metadata/cardinality do not match, run original prefill unchanged;
    save its output, then mark the proposed arm unmeasured and stop.
    """
    if arm not in ARMS:return kwargs,False,'unknown arm'
    if arm in ('plain','eadp1024'):return kwargs,False,None
    try:
        mask=kwargs.get('visual_pos_masks');deep=kwargs.get('deepstack_visual_embeds')
        embeds=kwargs.get('inputs_embeds')
        if embeds is None or len(embeds.shape)!=3:return kwargs,False,'missing prefill embeddings'
        if arm=='plain_ds_off':
            error=compatible(mask,deep,embeds)
            if error:return kwargs,False,'plain native '+error
            result=dict(kwargs);result['visual_pos_masks']=None;result['deepstack_visual_embeds']=None
        else:
            if mask is not None or deep is not None:return kwargs,False,'EADP unexpectedly already carries DeepStack'
            error=compatible(image_mask,deepstack,embeds)
            if error:return kwargs,False,'EADP candidate '+error
            result=dict(kwargs);result['visual_pos_masks']=image_mask;result['deepstack_visual_embeds']=deepstack
        return result,True,None
    except Exception as error:return kwargs,False,'intervention prerequisites: '+repr(error)
