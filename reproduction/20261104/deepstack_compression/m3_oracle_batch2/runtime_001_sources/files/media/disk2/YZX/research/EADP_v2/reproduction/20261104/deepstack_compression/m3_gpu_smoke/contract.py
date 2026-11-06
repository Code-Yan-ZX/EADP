"""No-GT identity and resource contracts for one real gradient smoke."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def tensor_meta(value):
    import torch
    t=value.detach().contiguous().cpu()
    return {'dtype':str(t.dtype),'shape':list(t.shape),'sha256':hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()}

def anchor(sample_path, records_path, task, qid, group):
    sample=json.loads(Path(sample_path).read_text())
    if sample.get('GT_opened') is not False or sample.get('predictions_opened') is not False:
        raise ValueError('sample provenance opened labels or predictions')
    rows=[x for x in sample['examples'] if x['task']==task and str(x['qid'])==qid and x['group_id']==group]
    if len(rows)!=1:raise ValueError('fixed M3 sample missing or duplicated')
    with Path(records_path).open() as f:
        refs=[r for line in f if (r:=json.loads(line)).get('task')==task and str(r.get('qid'))==qid
              and r.get('group_id')==group and r.get('path')=='stock_native' and r.get('forced') is False]
    if len(refs)!=1:raise ValueError('native identity anchor missing or duplicated')
    row,ref=rows[0],refs[0]
    for key in ('image_sha256','message_sha256'):
        if row[key]!=ref[key]:raise ValueError('M3/native input identity mismatch: '+key)
    array=ref['arrays']
    if array.get('kind')!='full_arrays' or sha(array['path'])!=array.get('sha256'):
        raise ValueError('native raw anchor array changed')
    return row,ref

def check_targets(targets, vocab_size, special_ids):
    if (targets!=[100,200,300] or any(type(x) is not int or x<0 or x>=vocab_size or x in special_ids for x in targets)):
        raise ValueError('synthetic target IDs invalid or modified')

def check_prefix_positions(positions, prompt_length, rope_delta, forced_ref, prefix_length):
    if (tuple(positions.shape)!=(3,1,prompt_length+prefix_length) or len(forced_ref.get('decode_trace',[]))<prefix_length+1):
        raise ValueError('teacher prefix position geometry invalid')
    for i in range(prefix_length):
        expected=prompt_length+rope_delta+i
        actual=positions[:,:,prompt_length+i].detach().cpu().tolist()
        native=forced_ref['decode_trace'][i+1]['position_ids']
        if actual!=[[expected],[expected],[expected]] or native!=[[[expected]],[[expected]],[[expected]]]:
            raise ValueError('teacher prefix differs from actual native cache decode')

def check_actual_inputs(torch,inputs,main,streams,positions,delta,ref):
    if (inputs['input_ids'].shape[0]!=1 or len(streams)!=3 or main.shape!=(1024,4096) or
        any(x.shape!=main.shape or x.dtype!=main.dtype or x.device!=main.device for x in streams)):
        raise ValueError('wrong full visual identity geometry')
    for key in ('input_ids','attention_mask','pixel_values','image_grid_thw'):
        if tensor_meta(inputs[key])!=ref['processor'][key]:raise ValueError('processor mismatch: '+key)
    if tensor_meta(main)!=ref['main_visual'] or [tensor_meta(x) for x in streams]!=ref['full_deepstack']:
        raise ValueError('actual main/DeepStack identity mismatch')
    if positions.detach().cpu().tolist()!=ref['prefill_position_ids'] or delta.detach().cpu().tolist()!=ref['rope_delta']:
        raise ValueError('actual native P3/rope delta mismatch')
    mask=inputs['attention_mask']
    if not torch.equal(mask,torch.ones_like(mask)):raise ValueError('one unpadded prompt required')

def compare_logits(actual, native, max_abs):
    import torch
    if actual.shape!=native.shape or not torch.isfinite(actual).all() or not torch.isfinite(native).all():
        raise ValueError('invalid native comparison logits')
    diff=(actual.float()-native.float()).abs()
    maximum=float(diff.max().item());equal_argmax=int(actual.argmax().item())==int(native.argmax().item())
    if maximum>max_abs or not equal_argmax:raise ValueError('native first logits outside frozen tolerance')
    return {'max_abs':maximum,'same_argmax':equal_argmax,'dtype':str(actual.dtype),'shape':list(actual.shape)}
