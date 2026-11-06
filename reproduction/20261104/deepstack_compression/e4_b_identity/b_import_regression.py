"""Model-free B import chain check; safe without GPU devices or datasets."""
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
HERE=Path(__file__).resolve().parent;E4=HERE.parent/'e4'
worker=runpy.run_path(str(HERE/'b_worker.py'),run_name='b_import_regression')
local=worker['local'];plan=json.loads(worker['PLAN'].read_text())
sys.path[:0]=[str(HERE),str(E4),plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
import torch
arm=local('arm');contract=local('pair_contract');verify=local('verify_records');gate=local('b_gate')
analysis=gate._analysis()
from model import pruner
from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
expected={'arm':HERE/'arm.py','pair_contract':HERE/'pair_contract.py','verify_records':HERE/'verify_records.py',
          'b_gate':HERE/'b_gate.py','pruner':Path(plan['author_root'])/'model/pruner.py',
          'wrapper':Path(plan['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py',
          'run_cpu_checks':E4/'run_cpu_checks.py','score_e4':HERE.parent/'e4_analysis/score_e4.py'}
actual={'arm':Path(arm.__file__),'pair_contract':Path(contract.__file__),'verify_records':Path(verify.__file__),
        'b_gate':Path(gate.__file__),'pruner':Path(pruner.__file__),'wrapper':Path(inspect.getfile(Qwen3VLChatEADP)),
        'run_cpu_checks':Path(sys.modules['run_cpu_checks'].__file__),'score_e4':Path(analysis.__file__)}
if any(actual[k].resolve()!=v.resolve() for k,v in expected.items()) or Path(inspect.getfile(Qwen3VLChatCDPruner)).resolve()!=expected['wrapper'].resolve():
    raise ValueError('B import path mismatch')
versions={n:importlib.metadata.version(n) for n in plan['expected_versions']}
if versions!=plan['expected_versions'] or torch.cuda.is_initialized():raise ValueError('runtime version or CUDA state mismatch')
print(json.dumps({'status':'passed','cwd':os.getcwd(),'pythonpath':os.environ.get('PYTHONPATH'),
                  'resolved':{k:str(v.resolve()) for k,v in actual.items()},'versions':versions,
                  'cuda_initialized':False,'model_loaded':False,'predictions':False}),flush=True)
