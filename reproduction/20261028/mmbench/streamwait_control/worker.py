#!/usr/bin/env python3
"""Apply the frozen three-statement AST control in memory, then the same observer."""
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import sys

from patch_only import OUT, BASE, build_patch

plan = json.loads((OUT / "protocol.json").read_text())
for item in plan["source_files"]:
    assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"], item["path"]
clip = importlib.import_module("llava.model.multimodal_encoder.clip_encoder")
module, validation = build_patch()
assert validation == plan["runtime_patch"]
namespace = dict(clip.__dict__)
exec(compile(module, validation["source_file"] + ":in_memory_mmb_streamwait", "exec"), namespace)
clip.CLIPVisionTower.forward = namespace["forward"]
assert clip.CLIPVisionTower.forward is namespace["forward"]
validation = dict(validation, applied_in_this_process=True,
                  author_source_after_patch_sha256=hashlib.sha256(Path(validation["source_file"]).read_bytes()).hexdigest())
assert validation["author_source_after_patch_sha256"] == validation["source_sha256"]
(OUT / "runtime_patch_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
# Same observer and native argv as the as-is run, with only output locations changed.
observer = BASE / "observe_native.py"
sys.argv[0] = str(observer)
runpy.run_path(str(observer), run_name="__main__")
