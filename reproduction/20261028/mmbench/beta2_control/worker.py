#!/usr/bin/env python3
"""Same frozen three-wait function and native observer; beta is native argv only."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
plan = json.loads((OUT / 'protocol.json').read_text())
for item in plan['source_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
spec = importlib.util.spec_from_file_location('unchanged_three_wait_builder', BASE / 'streamwait_control/patch_only.py')
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)
clip = importlib.import_module('llava.model.multimodal_encoder.clip_encoder')
assert Path(clip.__file__).resolve() == patcher.CLIP.resolve()
module, validation = patcher.build_patch()
assert validation == plan['runtime_patch']
namespace = dict(clip.__dict__)
exec(compile(module, validation['source_file'] + ':in_memory_mmb_streamwait', 'exec'), namespace)
clip.CLIPVisionTower.forward = namespace['forward']
validation = dict(validation, applied_in_this_process=True,
                  author_source_after_patch_sha256=hashlib.sha256(Path(validation['source_file']).read_bytes()).hexdigest())
assert validation['author_source_after_patch_sha256'] == validation['source_sha256']
(OUT / 'runtime_patch_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
sys.argv[0] = str(BASE / 'observe_native.py')
runpy.run_path(sys.argv[0], run_name='__main__')
