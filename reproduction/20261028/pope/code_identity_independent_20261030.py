"""CPU-only independent check of the POPE entry's runtime code identity."""
import ast
import gc
import hashlib
import json
import marshal
from pathlib import Path
import runpy
import sys
import types

ROOT = Path('/media/disk2/YZX/research/EADP_v2')
POPE = ROOT / 'reproduction/20261028/pope'
ENTRY = ROOT / '_reproduction/author_eadp/LLaVA/llava/eval/model_vqa_loader.py'
EXPECTED_PYTHON = Path('/home/dell/miniconda3/envs/llava_pruner/bin/python')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def extract(module):
    return next(c for c in module.co_consts if isinstance(c, types.CodeType) and c.co_name == 'eval_model')


def main():
    assert Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve()
    original = ENTRY.read_bytes()
    old_plan = json.loads((POPE / 'full_alpha1_beta2_single_control/plan.json').read_text())
    assert digest(original) == old_plan['source_sha256'][str(ENTRY)]
    module = compile(original, str(ENTRY), 'exec', dont_inherit=True, optimize=0)
    expected = extract(module)
    with_parent = digest(marshal.dumps(expected))
    with_parent_v2 = digest(marshal.dumps(expected, 2))
    del module
    gc.collect()
    without_parent = digest(marshal.dumps(expected))
    without_parent_v2 = digest(marshal.dumps(expected, 2))
    assert with_parent != without_parent and with_parent_v2 == without_parent_v2

    sys.path.insert(0, str(ROOT / '_reproduction/author_eadp/LLaVA'))
    sys.path.insert(0, str(POPE))
    from wait_only_patch import install
    install()
    import torch
    assert not torch.cuda.is_initialized()
    namespace = runpy.run_path(str(ENTRY), run_name='identity_cpu_independent')
    actual = namespace['eval_model'].__code__
    assert ENTRY.read_bytes() == original
    assert actual == expected and actual.co_filename == str(ENTRY)
    tree = ast.parse(original)
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'eval_model')
    function.body.insert(0, ast.parse('raise RuntimeError("changed")').body[0])
    ast.fix_missing_locations(tree)
    altered = extract(compile(tree, str(ENTRY), 'exec', dont_inherit=True, optimize=0))
    assert altered != actual
    renamed = actual.replace(co_filename='/different/path.py')
    assert renamed == actual and renamed.co_filename != str(ENTRY)
    changed_constant = actual.replace(co_consts=actual.co_consts + ('injected_constant',))
    assert changed_constant != actual
    assert not torch.cuda.is_initialized()
    result = dict(python=sys.version, source_sha256=digest(original),
        plan_recorded_marshal_sha256=old_plan['eval_model_code_sha256'],
        marshal_v4_with_parent=with_parent, marshal_v4_without_parent=without_parent,
        marshal_v2_with_parent=with_parent_v2, marshal_v2_without_parent=without_parent_v2,
        same_object_marshal_v4_depends_on_parent_lifetime=True,
        actual_code_equals_independently_compiled_source=actual == expected,
        filename_matches=actual.co_filename == str(ENTRY),
        injected_raise_rejected=altered != actual, changed_constant_rejected=changed_constant != actual,
        different_filename_requires_separate_check=renamed == actual and renamed.co_filename != str(ENTRY),
        torch_cuda_initialized=torch.cuda.is_initialized(), model_loaded=False,
        ground_truth_read=False, scoring_started=False,
        limits='Code equality and filename do not alone prove defaults, globals, dependencies or runtime inputs; retain other guards. The historical failed worker did not record its actual marshal hash.')
    target = POPE / 'code_identity_independent_20261030.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
