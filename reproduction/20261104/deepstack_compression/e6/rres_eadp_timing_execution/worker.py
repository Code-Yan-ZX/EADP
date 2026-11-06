"""Isolated model process candidate; public GPU entry is deliberately locked."""
import inspect
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = ROOT.parents[2]
TIMING = ROOT / 'e6/rres_eadp_timing_pilot_v4'


def clock_info():
    return {'time_namespace': os.readlink('/proc/self/ns/time'),
            'monotonic_clock': time.get_clock_info('monotonic').implementation,
            'perf_counter_clock': time.get_clock_info('perf_counter').implementation}


def load_dependencies():
    """Actual import path shared by a future no-device probe and GPU worker."""
    import torch
    sys.path.insert(0, str(TIMING))
    import module_load
    import process_core
    author = PROJECT / '_reproduction/author_eadp/Qwen_vl'
    sys.path[:0] = [str(author), str(author / 'VLMEvalKit'), str(ROOT / 'e6/rres_k256_pilot_v2')]
    import vlmeval.vlm.qwen3_vl.model_fixed_res as fixed
    import resolution
    from model import pruner
    from transformers import StoppingCriteriaList
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes, Qwen3VLChatEADP, Qwen3VLChatCDPruner
    if Path(inspect.getfile(Qwen3VLChatFixedRes)).resolve() != (author / 'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve():
        raise ValueError('author class location')
    adapter = module_load.load('_paired_timing_execution_adapter', ROOT / 'e4_c_budget_pilot_v2/adapter.py')
    return {'torch': torch, 'fixed': fixed, 'resolution': resolution, 'pruner': pruner,
            'criteria_type': StoppingCriteriaList, 'fixed_class': Qwen3VLChatFixedRes,
            'eadp_class': Qwen3VLChatEADP, 'cd_class': Qwen3VLChatCDPruner,
            'adapter': adapter, 'process_core': process_core}


def execute_after_admission(job, prepared, out):
    """Internal candidate, not an admitted GPU entry.

    Loads one model for its complete fixed process block. The two explicit
    synchronizations bracket model construction; neither is inside any request
    timing interval. Request-local adapter setup/cleanup remain outside the
    existing T0-to-text-return interval, as documented in the timing plan.
    """
    deps = load_dependencies()
    torch = deps['torch']
    recipe = json.loads((PROJECT / 'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json').read_text())['recipe']
    torch.cuda.synchronize()
    load_start_ns = time.monotonic_ns()
    if job['arm'] == 'rres':
        recipe = {k: v for k, v in recipe.items() if k not in ('visual_token_num', 'alpha', 'beta')}
        model = deps['fixed_class'](**recipe)
        def author_generate(wrapper, row):
            with torch.no_grad():
                return wrapper.generate_inner(row['message'], dataset=row['task'])
    elif job['arm'] == 'eadp':
        model = deps['eadp_class'](**recipe)
        def author_generate(wrapper, row):
            with torch.no_grad():
                return deps['cd_class'].generate_inner_transformers_fixed(wrapper, row['message'], dataset=row['task'])
    else:
        raise ValueError('arm')
    torch.cuda.synchronize()
    load_end_ns = time.monotonic_ns()
    if model.model.dtype != torch.bfloat16 or model.model.config._attn_implementation != 'flash_attention_2':
        raise ValueError('model backend')
    result = deps['process_core'].run_process(job, prepared['rows'], prepared['references'], torch=torch, wrapper=model,
        module=deps['fixed'], pruner=deps['pruner'], adapter_type=deps['adapter'].BudgetLightAdapter, criteria_type=deps['criteria_type'],
        fixed_resolution=deps['resolution'].fixed_resolution, author_generate=author_generate, out=out)
    return {**result, 'order': job['order'], 'model_load_start_monotonic_ns': load_start_ns,
            'model_load_end_monotonic_ns': load_end_ns, 'model_load_excluded': True, **clock_info()}


def main():
    raise RuntimeError('GPU timing worker locked pending full admission and independent review')


if __name__ == '__main__':
    main()
