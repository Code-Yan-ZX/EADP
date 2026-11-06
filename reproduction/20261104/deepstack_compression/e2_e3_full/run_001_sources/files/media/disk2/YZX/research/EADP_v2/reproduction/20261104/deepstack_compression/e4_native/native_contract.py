"""K1024 DS-on versus unmodified stock native identity, no scoring."""
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
E4=HERE.parent/'e4'
sys.path.insert(0,str(E4))
from smoke_contract import select_smoke

TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')

def select_anchors(e1):
    rows=select_smoke(e1);out=[]
    for task in TASKS:
        selected=[r for r in rows if r['task']==task]
        if not selected:raise ValueError('task has no predeclared smoke anchor')
        out.append(selected[0])
    if len({r['group_id'] for r in out})!=3:raise ValueError('anchors are not distinct groups')
    return out


def compare_pair(native,budget):
    if native.get('path')!='stock_native' or budget.get('budget')!=1024 or budget.get('ds_on') is not True:
        raise ValueError('incorrect comparison path')
    if budget.get('selection_mode')!='keep_all' or budget.get('greedy_calls')!=0:
        raise ValueError('K1024 is not author keep-all')
    if native.get('forced') is not budget.get('forced') or type(native.get('forced')) is not bool:
        raise ValueError('generation mode mismatch')
    mapping={'task':'task','qid':'qid','group_id':'group_id','image_sha256':'image_sha256','message_sha256':'message_sha256',
             'processor':'processor','main_visual':'main_visual','full_deepstack':'full_deepstack',
             'prefill_embeds':'prefill_embeds','prefill_mask':'prefill_mask','prefill_position_ids':'prefill_position_ids',
             'rope_delta':'adjusted_delta','injected_mask':'injected_mask','injected_deepstack':'injected_deepstack',
             'prefill_logits':'prefill_logits','output_ids':'output_ids','answer':'answer','eos_ids':'eos_ids',
             'eos':'eos','truncated':'truncated','max_new_tokens':'max_new_tokens','generation_scalars':'generation_scalars',
             'decode_trace':'decode_trace','cache_fingerprints':'cache_fingerprints'}
    for left,right in mapping.items():
        if left not in native or right not in budget or native[left]!=budget[right]:raise ValueError('stock/budget mismatch: '+left)
    if budget['spatial_ids']!=list(range(1024)) or budget['ordered_ids']!=list(range(1024)):
        raise ValueError('K1024 missing complete spatial identity')
    if len(native['full_deepstack'])!=3 or len(native['injected_deepstack'])!=3:
        raise ValueError('missing native DS streams')
    wanted=33 if native['forced'] else 1
    if len(native['cache_fingerprints'])!=wanted:raise ValueError('incomplete actual KV fingerprints')
    return True
