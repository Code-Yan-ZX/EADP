"""Install the same three waits, then enter the native FULL observer."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
plan=json.loads((OUT/'protocol.json').read_text())
for item in plan['source_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
spec=importlib.util.spec_from_file_location('frozen_full_wait_patch',BASE/'streamwait_control/patch_only.py')
patcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(patcher)
clip=importlib.import_module('llava.model.multimodal_encoder.clip_encoder')
assert Path(clip.__file__).resolve()==patcher.CLIP.resolve()
module,proof=patcher.build_patch()
assert proof==plan['runtime_patch']
namespace=dict(clip.__dict__)
exec(compile(module,proof['source_file']+':same_three_waits_FULL','exec'),namespace)
clip.CLIPVisionTower.forward=namespace['forward']
proof=dict(proof,applied_in_this_process=True,author_source_after_patch_sha256=hashlib.sha256(patcher.CLIP.read_bytes()).hexdigest())
assert proof['author_source_after_patch_sha256']==proof['source_sha256']
(OUT/'runtime_patch_validation.json').write_text(json.dumps(proof,indent=2)+'\n')
sys.argv[0]=str(OUT/'observe_full.py')
runpy.run_path(sys.argv[0],run_name='__main__')
