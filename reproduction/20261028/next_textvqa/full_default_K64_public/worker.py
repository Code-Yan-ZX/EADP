"""Isolated K64 adapter of the frozen K128 native worker; never schedules GPUs."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
NEXT = OUT.parent
BASE = NEXT / 'repro_full_default.py'
BASE_SHA = '466e3c7be72cf5db3a12409e50fcb47286f6d054af4fa6f6f0fad351a6d4e94d'
PLAN = OUT / 'plan.json'
RUN = OUT / 'execution'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def adapted_module():
    assert sha(BASE) == BASE_SHA
    original = ast.parse(BASE.read_text())
    original.body = [n for n in original.body if not (isinstance(n, ast.If) and '__name__' in ast.unparse(n.test))]
    tree = copy.deepcopy(original)
    originals = {}
    changes = {'HERE': str(NEXT), 'PLAN': str(PLAN), 'RUN': str(RUN)}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            if name in changes:
                originals[name] = copy.deepcopy(n.value)
                n.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[ast.Constant(changes[name])], keywords=[])
            elif name == 'PARAMETERS':
                originals[name] = copy.deepcopy(n.value)
                for k in n.value.keywords:
                    if k.arg == 'visual_token_num':
                        assert k.value.value == 128
                        k.value.value = 64
                    elif k.arg == 'budget_nominal':
                        assert k.value.value == 640
                        k.value.value = 320
    assert set(originals) == {'HERE', 'PLAN', 'RUN', 'PARAMETERS'}
    verify = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify')
    changed = 0
    for n in ast.walk(verify):
        if isinstance(n, ast.Compare) and ast.unparse(n) == "actual['K'] == 128":
            n.comparators[0].value = 64
            changed += 1
        if isinstance(n, ast.keyword) and n.arg in ('budget_cli', 'budget_nominal'):
            expected = 128 if n.arg == 'budget_cli' else 640
            assert n.value.value == expected
            n.value.value = 64 if n.arg == 'budget_cli' else 320
            changed += 1
    assert changed == 3
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    loader = next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == 'loader')
    at = next(i for i, n in enumerate(loader.body) if isinstance(n, ast.Assign) and ast.unparse(n) == 'generate = model.generate')
    hook = ast.parse('install_observer(model, observed, RUN)').body[0]
    loader.body.insert(at, hook)

    restored = copy.deepcopy(tree)
    for n in restored.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in originals:
            n.value = copy.deepcopy(originals[n.targets[0].id])
    verifier = next(n for n in restored.body if isinstance(n, ast.FunctionDef) and n.name == 'verify')
    for n in ast.walk(verifier):
        if isinstance(n, ast.Compare) and ast.unparse(n) == "actual['K'] == 64":
            n.comparators[0].value = 128
        if isinstance(n, ast.keyword) and n.arg in ('budget_cli', 'budget_nominal'):
            n.value.value = 128 if n.arg == 'budget_cli' else 640
    run_restored = next(n for n in restored.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    loader_restored = next(n for n in run_restored.body if isinstance(n, ast.FunctionDef) and n.name == 'loader')
    assert ast.dump(loader_restored.body.pop(at)) == ast.dump(hook)
    assert ast.dump(restored) == ast.dump(original), 'Every generation/scoring operation must restore exactly'
    ast.fix_missing_locations(tree)
    from observe import install_observer
    namespace = dict(__name__='isolated_K64_worker', __file__=str(Path(__file__).resolve()), install_observer=install_observer)
    exec(compile(tree, str(BASE) + '[only K64 expectations and isolated observer]', 'exec'), namespace)
    import types
    proof = dict(base_path=str(BASE), base_sha256=BASE_SHA, restored_AST_identical=True,
                 numerical_source_changed=False, parameter_changes={'visual_token_num': [128, 64], 'budget_nominal': [640, 320]},
                 other_changes=['HERE/PLAN/RUN isolation', 'Verifier K/budget expectations', 'Observation-only install before native generate'])
    return types.SimpleNamespace(**namespace), proof


if __name__ == '__main__':
    # Only the allocated controller may fork this single attempt.
    assert len(sys.argv) == 2
    note = json.loads(Path(sys.argv[1]).read_text())
    claim = json.loads((OUT / 'launch.claim.json').read_text())
    assert note['authorization'] == 'explicit_root_gpu_allocation'
    assert note['run_id'] == 'NeXT_public_default_K64_alpha05_beta1_wait'
    assert note['plan_sha256'] == sha(PLAN) == claim['plan_sha256']
    assert sha(sys.argv[1]) == claim['allocation_note_sha256']
    assert os.getppid() == claim['controller']['pid']
    full, proof = adapted_module()
    full.run(True)
