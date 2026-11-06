"""Clean-process no-device import-path probe; no model construction."""
import inspect
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TIMING = ROOT / 'e6/rres_eadp_timing_pilot_v4'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('no-device environment required')
    sys.path.insert(0, str(TIMING))
    import reference_inputs
    prepared = reference_inputs.load()
    sys.path.insert(0, str(HERE))
    import worker
    deps = worker.load_dependencies()
    torch = deps['torch']
    if torch.cuda.is_initialized() or len(prepared['rows']) != 62 or len(prepared['references']) != 124:
        raise ValueError('no-device import identity')
    if Path(deps['process_core'].__file__).resolve() != TIMING / 'process_core.py' or Path(deps['process_core'].runner.__file__).resolve() != TIMING / 'paired_runner.py':
        raise ValueError('wrong actual process runner')
    if Path(inspect.getfile(deps['adapter'].BudgetLightAdapter)).resolve() != ROOT / 'e4_c_budget_pilot_v2/adapter.py':
        raise ValueError('adapter class path')
    if Path(inspect.getfile(deps['fixed_class'])).resolve() != PROJECT / '_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py':
        raise ValueError('model class path')
    print(json.dumps({'status':'passed', 'inputs':len(prepared['rows']), 'references':len(prepared['references']), 'cuda_initialized':False, 'model_loaded':False, 'GT_opened':False, 'existing_predictions_opened':True}))


PROJECT = ROOT.parents[2]
if __name__ == '__main__': main()
