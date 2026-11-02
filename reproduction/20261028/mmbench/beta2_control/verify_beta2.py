#!/usr/bin/env python3
"""Unchanged full CPU verifier, with only output path and expected beta rebound."""
import ast
import copy
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def adapted_verifier():
    path = BASE / 'verify_completed.py'
    original = ast.parse(path.read_text())
    tree = copy.deepcopy(original)
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'OUT' for t in n.targets)]
    assert len(assignments) == 1
    old_out = copy.deepcopy(assignments[0].value)
    assignments[0].value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[ast.Constant(str(OUT))], keywords=[])
    comparisons = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)
                   and ast.unparse(n.left) == "loaded['beta']"]
    assert len(comparisons) == 1 and ast.unparse(comparisons[0]) == "loaded['beta'] == 1"
    comparisons[0].comparators[0] = ast.Constant(2)
    restored = copy.deepcopy(tree)
    target = next(n for n in restored.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'OUT' for t in n.targets))
    target.value = old_out
    target_beta = next(n for n in ast.walk(restored) if isinstance(n, ast.Compare)
                       and ast.unparse(n.left) == "loaded['beta']")
    target_beta.comparators[0] = ast.Constant(1)
    assert ast.dump(restored) == ast.dump(original)
    tree = ast.fix_missing_locations(tree)
    compile(tree, str(path) + ':only_OUT_and_expected_beta', 'exec')
    return path, tree


def main():
    path, tree = adapted_verifier()
    exec(compile(tree, str(path) + ':only_OUT_and_expected_beta', 'exec'), {'__file__': str(path), '__name__': '__main__'})
    patch = json.loads((OUT / 'runtime_patch_validation.json').read_text())
    assert patch['applied_in_this_process'] and patch['restored_ast_identical'] and patch['exact_added_statement_count'] == 3
    assert patch['author_source_after_patch_sha256'] == patch['source_sha256']
    report = json.loads((OUT / 'independent_verification.json').read_text())
    report['controlled_stream_patch'] = patch
    report['verifier_adapter'] = {'path': str(Path(__file__).resolve()),
                                  'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  'only_changes': ['OUT directory', 'Expected loaded beta from1 to2'],
                                  'restored_original_AST_identical': True, 'metric_math_changed': False}
    baseline_path = BASE / 'streamwait_control/independent_verification.json'
    baseline = json.loads(baseline_path.read_text())
    assert baseline['success'] and baseline['n_rows'] == 4377 and baseline['n_circular_groups'] == 1176
    report['comparison_boundary'] = ('Primary comparison is oldEN4377 K128 alpha0.5 beta1/wait versus beta2/wait, with identical inputs and three waits. '
                                     'Historical V11 beta2 remains a separate reference; it does not by itself isolate every environment difference.')
    report['current_beta'] = 2
    report['beta1_same_input_wait_baseline'] = {'path': str(baseline_path),
                                               'sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                                               'value_percent': baseline['circular_accuracy_percent'],
                                               'current_minus_baseline_percentage_points': report['circular_accuracy_percent'] - baseline['circular_accuracy_percent']}
    (OUT / 'independent_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
