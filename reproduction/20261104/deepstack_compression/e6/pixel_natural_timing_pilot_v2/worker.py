"""One-model / one-arm / one-anchor natural TTFT worker candidate.

Real GPU entry is deliberately locked until controller, no-GPU preflight and
independent source/endpoint reviews have been implemented and passed.
"""
import argparse,contextlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
E4=BASE/'e4_c_latency'
sys.path.insert(0,str(E4));sys.path.insert(0,str(HERE))

def _load_bootstrap():
 spec=importlib.util.spec_from_file_location('_pixel_pilot_bootstrap_v2',HERE/'bootstrap.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.pin_all();return module
_bootstrap=_load_bootstrap()
from pilot_contract import RECIPE,validate_generation,verify_expected
from observer import count_calls
from timer import measure

@contextlib.contextmanager
def replaced(obj,name,fn):
    had=name in getattr(obj,'__dict__',{});old=getattr(obj,name);stored=obj.__dict__.get(name) if had else None
    setattr(obj,name,fn(old))
    try:yield
    finally:
        if had:setattr(obj,name,stored)
        else:delattr(obj,name)

def durable_raw(original,emit):
    def generate(*args,**kw):
        result=original(*args,**kw)
        emit(result.detach().cpu().tolist())
        return result
    return generate

def append(path,record):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())

def execute_pixel(wrapper,patch,torch,row,synchronize,expected,emit_raw,observe_count=True,validate_input=None):
    """Run the author message path; measure from image read to first chosen token."""
    from transformers.generation.stopping_criteria import StoppingCriteriaList
    captured={}
    def processor(original):
        def call(*args,**kw):
            inputs=original(*args,**kw);captured['inputs']=inputs;captured['prompt_length']=int(inputs['input_ids'].shape[1]);return inputs
        return call
    def invoke(boundary):
        def generate(original):
            def call(*args,**kw):
                expected_scalar={k:v for k,v in RECIPE.items() if k not in ('num_beams','use_cache')}
                if {k:kw.get(k) for k in expected_scalar}!=expected_scalar:raise ValueError('actual recipe mismatch')
                for key in ('num_beams','use_cache'):
                    if key in kw and kw[key]!=RECIPE[key]:raise ValueError('conflicting generation setting')
                    kw[key]=RECIPE[key]
                if kw.get('stopping_criteria') is not None:raise ValueError('unexpected stopping criterion')
                kw['stopping_criteria']=StoppingCriteriaList([boundary])
                eos=kw.get('eos_token_id',wrapper.model.generation_config.eos_token_id)
                captured['eos_ids']=[eos] if type(eos) is int else list(eos or [])
                out=original(*args,**kw)
                captured['full']=out.detach().cpu().tolist()
                emit_raw(captured['full'])
                return out
            return call
        with replaced(wrapper.model,'generate',generate):
            return wrapper.generate_inner_transformers_fixed(row['message'],dataset=row['task'])
    observer=count_calls(patch,'compute_merged_keep_indices','pixel',1) if observe_count else contextlib.nullcontext([])
    with observer as k:
        with replaced(wrapper,'_processor_inputs',processor):
            with torch.inference_mode(): answer,times=measure(invoke,synchronize)
    if len(captured.get('full',[]))!=1 or 'prompt_length' not in captured:raise ValueError('missing generated sequence')
    seq=captured['prompt_length'];full=captured['full'][0]
    if len(full)<seq:raise ValueError('output shorter than input')
    if captured['inputs']['input_ids'].detach().cpu().tolist()[0]!=full[:seq]:raise ValueError('generated input prefix mismatch')
    if validate_input is not None:validate_input(captured['inputs'])
    tokens=full[seq:]
    status=validate_generation(tokens,captured['eos_ids'],times['first_criteria_length'],seq,True)
    verify_expected(tokens,expected)
    return {'answer':answer,'tokens':tokens,'eos_ids':captured['eos_ids'],'dynamic_k':k[0] if k else None,
            'selection_calls':len(k) if observe_count else None,'count_observer_enabled':observe_count,'recipe':RECIPE,'boundaries':times,**status}

def execute_eadp(wrapper,native,torch,row,synchronize,expected,emit_raw,observe_count=True,validate_input=None):
    from runner import execute_one
    captured={}
    def wrap(original):
        def call(*args,**kw):
            scalar={k:v for k,v in RECIPE.items() if k not in ('num_beams','use_cache')}
            if {k:kw.get(k) for k in scalar}!=scalar:raise ValueError('actual EADP generation recipe mismatch')
            for key in ('num_beams','use_cache'):
                if key in kw and kw[key]!=RECIPE[key]:raise ValueError('conflicting EADP setting')
                kw[key]=RECIPE[key]
            return durable_raw(original,emit_raw)(*args,**kw)
        return call
    def processor(original):
        def call(*args,**kw):
            inputs=original(*args,**kw);captured['inputs']=inputs;return inputs
        return call
    observer=count_calls(native,'_greed_select_impl','eadp',1) if observe_count else contextlib.nullcontext([])
    with observer as k:
        with replaced(wrapper,'_processor_inputs',processor):
            with replaced(wrapper.model,'generate',wrap):
                with torch.inference_mode():value=execute_one(wrapper,native,torch,row,'C-ON','natural',synchronize)
    if validate_input is not None:
        if 'inputs' not in captured:raise ValueError('missing actual processor input')
        validate_input(captured['inputs'])
    if observe_count and k!=[256]:raise ValueError('actual EADP K mismatch')
    tokens=value['output_ids'];verify_expected(tokens,expected)
    status=validate_generation(tokens,value['eos_ids'],value['boundaries']['first_criteria_length'],0,False)
    return {'answer':value['answer'],'tokens':tokens,'eos_ids':value['eos_ids'],'dynamic_k':k[0] if k else None,
            'selection_calls':len(k) if observe_count else None,'count_observer_enabled':observe_count,'recipe':RECIPE,'boundaries':value['boundaries'],**status}

def memory_snapshot(cuda):
    return {'allocated_bytes':int(cuda.memory_allocated()),'reserved_bytes':int(cuda.memory_reserved())}

def run_two(job,run,cuda,journal):
    """One process/job; raw output is persisted after each generate returns.

    Allocator peaks cover the entire generation including decoder and observer,
    not only TTFT; they are not process NVML or device-wide peak measurements.
    """
    for phase in ('warmup','measured'):
        append(journal,{'event':'started','phase':phase,'job':job,'pid':os.getpid()})
        cuda.synchronize();before=memory_snapshot(cuda);cuda.reset_peak_memory_stats()
        try:
            result=run(phase,lambda raw:append(journal,{'event':'raw_generation','phase':phase,'job':job,'ids':raw}))
            cuda.synchronize()
            memory={'baseline':before,'end':memory_snapshot(cuda),'max_allocated_bytes':int(cuda.max_memory_allocated()),
                    'max_reserved_bytes':int(cuda.max_memory_reserved()),'scope':'PyTorch allocator, entire generation; not NVML or total-device peak'}
            append(journal,{'event':'completed','phase':phase,'job':job,'result':result,'memory':memory,
                 'count_observer':'disabled in warmup; measured uses Python selector-return wrapper, overhead included, no added explicit CUDA sync',
                 'raw_fsync':'after generate return; after TTFT boundary; included in generation-end time'})
        except BaseException as exc:
            append(journal,{'event':'failed','phase':phase,'job':job,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
            raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--job');parser.add_argument('--out',required=True);args=parser.parse_args()
    if not args.preflight_only:
        # Must fail before torch import, output creation, model load or GPU query.
        raise ValueError('GPU locked: complete bounded controller, reference admission, endpoint identity, actual no-GPU preflight and independent review required')
    raise ValueError('CPU preflight locked until frozen dependency inventory and runtime checks are installed')
if __name__=='__main__':main()
