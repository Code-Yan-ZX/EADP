"""Real-model natural generation with durable immediate token exposure callback.

Requires caller to have separately admitted a frozen DEV row/model. This file
never loads a model, opens labels, starts GPU work, or grants execution.
"""
from contextlib import ExitStack
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from selection_adapter import SelectedAdapter
from baseline import meta
from decode_observer import DecodeObserver,validate_trace
if Path(sys.modules['decode_observer'].__file__).resolve()!=(HERE.parent/'e4/decode_observer.py').resolve():
    raise ValueError('conflicting decode observer module')

def generate(wrapper,pruner_module,row,spatial,greedy_order,on_generated,author_generate,torch,expected_scalars,expected_eos,expected_full):
    if row.get('split')!='DEV' or wrapper.visual_token_num!=256 or wrapper.pruner.visual_token_num!=256:
        raise ValueError('expected admitted DEV K256 model')
    seen={};original=wrapper.model.generate
    def capture(*args,**kwargs):
        if seen:raise ValueError('multiple model.generate calls')
        names=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')
        scalars={k:kwargs.get(k) for k in names}
        if scalars!=expected_scalars or scalars['max_new_tokens']!=2048 or scalars['do_sample'] not in (False,None):
            raise ValueError('generation recipe mismatch')
        eos=kwargs.get('eos_token_id',wrapper.model.generation_config.eos_token_id)
        eos=[eos] if isinstance(eos,int) else list(eos or [])
        if eos!=expected_eos:raise ValueError('generation EOS recipe mismatch')
        output=original(*args,**kwargs);ids=output.detach().cpu().tolist()[0]
        # This callback fsyncs before author decoding, hook validation or any
        # other operation that could fail after a real generation.
        on_generated({'token_ids':ids,'eos_ids':eos,'generation_scalars':scalars})
        seen.update(token_ids=ids,eos_ids=eos,generation_scalars=scalars)
        return output
    with SelectedAdapter(torch,wrapper,pruner_module,spatial,greedy_order,expected_full) as adapter, DecodeObserver(wrapper) as observer:
        wrapper.model.generate=capture
        try:answer=author_generate(wrapper,row)
        finally:wrapper.model.generate=original
        adapter.verify_calls()
        if not seen or not isinstance(answer,str):raise ValueError('generation not completed')
        ids=seen['token_ids'];eos=bool(ids and ids[-1] in seen['eos_ids']);truncated=not eos and len(ids)>=2048
        if not eos and not truncated:raise ValueError('unexpected generation termination')
        state=adapter.state
        validate_trace(observer.trace,ids,len(state['retained']),int(state['adjusted_delta'].item()))
        evidence={'processor':{k:meta(v) for k,v in state['inputs'].items() if k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},
                  'full_streams':[meta(state['visual'])]+[meta(x) for x in state['deepstack']],
                  'selected_ds':[meta(x) for x in state['actual_deepstack']], 'spatial_ids':state['spatial'].detach().cpu().tolist(),
                  'retained_ids':state['retained'].detach().cpu().tolist(),'prefill_embeds':meta(state['prefill_embeds']),
                  'prefill_mask':meta(state['prefill_mask']),'position_ids':state['prefill_position_ids'].detach().cpu().tolist(),
                  'rope_delta':state['adjusted_delta'].detach().cpu().tolist(),'ds_mask':state['actual_ds_mask'].detach().cpu().tolist(),
                  'decode_trace':observer.trace}
        return {'token_ids':ids,'eos':eos,'truncated':truncated,'answer':answer,
                'eos_ids':seen['eos_ids'],'generation_scalars':seen['generation_scalars'],
                'actual_selection_verified':state['actual_injection_verified'],'evidence':evidence}
