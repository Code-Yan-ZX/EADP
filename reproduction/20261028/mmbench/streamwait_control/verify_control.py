#!/usr/bin/env python3
"""Reuse the complete as-is CPU verifier with only its output directory rebound."""
import ast
import copy
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
verifier = BASE / "verify_completed.py"
original = ast.parse(verifier.read_text())
adapted = copy.deepcopy(original)
assignments = [node for node in adapted.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "OUT" for target in node.targets)]
assert len(assignments) == 1
old_value = copy.deepcopy(assignments[0].value)
assignments[0].value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(str(OUT))], keywords=[])
restored = copy.deepcopy(adapted)
restored_assignment = next(node for node in restored.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "OUT" for target in node.targets))
restored_assignment.value = old_value
assert ast.dump(restored) == ast.dump(original)
adapted = ast.fix_missing_locations(adapted)
exec(compile(adapted, str(verifier) + ":control_directory_only", "exec"), {"__file__": str(verifier), "__name__": "__main__"})
patch = json.loads((OUT / "runtime_patch_validation.json").read_text())
assert patch["applied_in_this_process"] and patch["restored_ast_identical"] and patch["exact_added_statement_count"] == 3
assert patch["author_source_after_patch_sha256"] == patch["source_sha256"]
report = json.loads((OUT / "independent_verification.json").read_text())
report["controlled_stream_patch"] = patch
report["verifier_adapter"] = {"path": str(Path(__file__).resolve()),
                              "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              "only_change": "CPU verifier OUT directory rebound; all checks and metric math retained"}
(OUT / "independent_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
