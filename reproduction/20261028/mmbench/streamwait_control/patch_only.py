"""Build the sole controlled change: caller capture and two CUDA stream waits."""
import ast
import copy
import hashlib
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
AUTHOR = BASE.parents[2] / "_reproduction/author_eadp"
CLIP = AUTHOR / "LLaVA/llava/model/multimodal_encoder/clip_encoder.py"


def build_patch():
    source = ast.parse(CLIP.read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "CLIPVisionTower")
    original = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    patched = copy.deepcopy(original)
    branch = next(node for node in patched.body if isinstance(node, ast.If))
    positions = [i + 1 for i, node in enumerate(branch.orelse) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "text_stream" for target in node.targets)]
    assert len(positions) == 1
    position = positions[0]
    additions = ast.parse("caller_stream = torch.cuda.current_stream(device=self.device)\n"
                          "image_stream.wait_stream(caller_stream)\n"
                          "text_stream.wait_stream(caller_stream)\n").body
    branch.orelse[position:position] = additions
    restored = copy.deepcopy(patched)
    restored_branch = next(node for node in restored.body if isinstance(node, ast.If))
    removed = restored_branch.orelse[position:position + 3]
    del restored_branch.orelse[position:position + 3]
    assert ast.dump(restored) == ast.dump(original)
    assert [ast.dump(node) for node in removed] == [ast.dump(node) for node in additions]
    module = ast.fix_missing_locations(ast.Module(body=[patched], type_ignores=[]))
    compile(module, str(CLIP) + ":in_memory_mmb_streamwait", "exec")
    digest = lambda node: hashlib.sha256(ast.dump(node).encode()).hexdigest()
    validation = {"source_file": str(CLIP), "source_sha256": hashlib.sha256(CLIP.read_bytes()).hexdigest(),
                  "original_forward_ast_sha256": digest(original), "patched_forward_ast_sha256": digest(patched),
                  "restored_ast_identical": True, "exact_added_statement_count": 3,
                  "added_statements": [ast.unparse(node) for node in additions],
                  "patched_forward_source": ast.unparse(patched), "production_source_modified": False,
                  "preserved": ["All original mathematical operations", "Original no_grad decorator",
                                "Original final torch.cuda.synchronize", "List/attention branches",
                                "Precision, allocation and generation configuration"],
                  "record_stream_added": False, "GPU_work_during_preparation": False}
    return module, validation
