#!/usr/bin/env python3
"""Declare and verify the exact three synchronization nodes, without disk edits."""
import ast
import copy
import hashlib
import importlib
from pathlib import Path

SOURCE = Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/LLaVA/llava/model/multimodal_encoder/clip_encoder.py')
PATCH_TEXT = '''caller_stream = torch.cuda.current_stream(device=self.device)
image_stream.wait_stream(caller_stream)
text_stream.wait_stream(caller_stream)
'''


def dump(node):
    return ast.dump(node, include_attributes=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def function_ast():
    module = ast.parse(SOURCE.read_text())
    cls = next(x for x in module.body if isinstance(x, ast.ClassDef) and x.name == 'CLIPVisionTower')
    return next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'forward')


def build():
    original = function_ast()
    patched = copy.deepcopy(original)
    inserted = ast.parse(PATCH_TEXT).body
    insertion_count = 0
    for node in list(ast.walk(patched)):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for index, statement in enumerate(value[:]):
                if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == 'text_stream'):
                    value[index + 1:index + 1] = copy.deepcopy(inserted)
                    insertion_count += 1
    assert insertion_count == 1
    restored = copy.deepcopy(patched)
    removal_count = 0
    for node in list(ast.walk(restored)):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for index in range(max(0, len(value) - 2)):
                if all(isinstance(value[index + j], ast.AST) and dump(value[index + j]) == dump(inserted[j]) for j in range(3)):
                    del value[index:index + 3]
                    removal_count += 1
                    break
    assert removal_count == 1 and dump(restored) == dump(original)
    proof = {'production_source': str(SOURCE), 'production_source_sha256': sha(SOURCE.read_bytes()),
        'original_forward_ast_sha256': sha(dump(original).encode()), 'patched_forward_ast_sha256': sha(dump(patched).encode()),
        'restored_forward_ast_sha256': sha(dump(restored).encode()), 'restoration_ast_identical': True,
        'inserted_nodes': PATCH_TEXT.strip().splitlines(), 'insertion_count': insertion_count,
        'source_files_modified': False, 'record_stream_added': False,
        'unchanged': ['all original arithmetic', 'no_grad decorator', 'trailing torch.cuda.synchronize()',
                      'attention/list branches', 'allocation/precision', 'text tokenization', 'model parameters', 'selection']}
    return patched, proof


def install():
    patched_ast, proof = build()
    module = importlib.import_module('llava.model.multimodal_encoder.clip_encoder')
    assert Path(module.__file__).resolve() == SOURCE.resolve()
    original = module.CLIPVisionTower.forward
    function_module = ast.Module(body=[patched_ast], type_ignores=[])
    ast.fix_missing_locations(function_module)
    namespace = {}
    exec(compile(function_module, str(SOURCE), 'exec'), module.__dict__, namespace)
    replacement = namespace['forward']
    replacement.__qualname__ = original.__qualname__
    replacement.__module__ = original.__module__
    module.CLIPVisionTower.forward = replacement
    proof['actual_imported_module'] = module.__file__
    proof['runtime_patch_installed'] = module.CLIPVisionTower.forward is replacement
    return proof
