#!/usr/bin/env python3
"""Native author entry, identical three waits, and read-only loaded-model identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--observer-output', required=True, type=Path)
observer, native_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + native_args

from wait_only_patch import install
proof = install()
(observer.observer_output / 'runtime_patch_proof.json').write_text(json.dumps(proof, indent=2) + '\n')

import llava.model.builder as builder
original_load = builder.load_pretrained_model


def observed_load(*args, **kwargs):
    assert kwargs['visual_token_num'] == 32 and kwargs['alpha'] == .5 and kwargs['beta'] == 1.
    result = original_load(*args, **kwargs)
    tokenizer, model, processor, context_len = result
    assert model.visual_token_num == 32 and model.alpha == .5 and model.beta == 1.
    tower = model.get_vision_tower()
    modules = {name: sys.modules[name].__file__ for name in [
        'llava.model.builder', 'llava.model.llava_arch', 'llava.model.language_model.llava_llama',
        'llava.model.multimodal_encoder.clip_encoder']}
    dtypes = {name: sorted({str(p.dtype) for p in value.parameters()}) for name, value in {
        'model': model, 'vision': tower.vision_tower, 'text': tower.text_tower,
        'visual_projection': tower.vision_tower.visual_projection}.items()}
    report = {'observer': 'After native model load only; native generate and per-question loop are unwrapped.',
        'worker_pid': os.getpid(), 'native_argv': sys.argv, 'actual_load_args': list(args),
        'actual_load_kwargs': kwargs, 'model_class': type(model).__module__ + '.' + type(model).__name__,
        'actual_alpha': model.alpha, 'actual_beta': model.beta, 'actual_visual_token_num': model.visual_token_num,
        'context_len': context_len, 'loaded_module_paths': modules,
        'loaded_source_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in modules.items()},
        'parameter_dtypes': dtypes, 'attention_backend': getattr(model.config, '_attn_implementation', None),
        'torch': __import__('torch').__version__, 'transformers': __import__('transformers').__version__,
        'tokenizer_class': type(tokenizer).__name__, 'processor_class': type(processor).__name__}
    (observer.observer_output / 'actual_loaded_runtime.json').write_text(json.dumps(report, indent=2) + '\n')
    print('[actual_loaded_runtime] ' + json.dumps(report), flush=True)
    return result


builder.load_pretrained_model = observed_load
runpy.run_module('llava.eval.model_vqa_loader', run_name='__main__')
