#!/usr/bin/env python3
"""Three-node wait patch, followed by the otherwise unmodified author entry."""
import json
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
from wait_only_patch import install

OUT = Path(__file__).resolve().parent / 'run02_source_wait_only_k32'
proof = install()
(OUT / 'runtime_patch_proof.json').write_text(json.dumps(proof, indent=2) + '\n')
print('[declared_runtime_wait_only_patch] ' + json.dumps(proof), flush=True)
runpy.run_module('llava.eval.model_vqa_loader', run_name='__main__')
