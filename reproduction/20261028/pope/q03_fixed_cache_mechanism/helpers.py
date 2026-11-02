"""Fresh-author pruning ASTs; change only the entropy-filter call .2 to .3."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE.parents[3] / '_reproduction/author_eadp/LLaVA/llava/model/llava_arch.py'
ORIGINAL_HELPER = HERE.parent / 'gpu_formula_facility_panel/formula_helpers.py'
ORIGINAL_HELPER_SHA = 'ce0c928ee20c035739eb2376e8e057cc46b8ea1bf79a9334feb39bf6c02d2e6a'


def digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def old_helpers():
    assert hashlib.sha256(ORIGINAL_HELPER.read_bytes()).hexdigest() == ORIGINAL_HELPER_SHA
    spec = importlib.util.spec_from_file_location('frozen_POPE_formula_AST_helpers', ORIGINAL_HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def nodes():
    original = old_helpers()
    methods, source, _, traced, old_proof = original.source_parts(ARCH)
    changed = copy.deepcopy(source)
    calls = [n for n in ast.walk(changed) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == 'entrpy_filter']
    assert len(calls) == 1
    keyword = next(n for n in calls[0].keywords if n.arg == 'entropy_keep_ratio')
    assert isinstance(keyword.value, ast.Constant) and keyword.value.value == .2
    keyword.value.value = .3
    inverse = copy.deepcopy(changed)
    call = next(n for n in ast.walk(inverse) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == 'entrpy_filter')
    next(n for n in call.keywords if n.arg == 'entropy_keep_ratio').value.value = .2
    assert digest(inverse) == digest(source) == old_proof['native_importance_ast_sha256']
    # The final effective mask uses the same author scatter/clamp program. No
    # alternative selection/mask implementation is introduced by this replay.
    encode = methods['encode_images']
    start = next(i for i, n in enumerate(encode.body) if isinstance(n, ast.Assign)
        and ast.unparse(n) == 'device = select_idx.device')
    end = next(i for i, n in enumerate(encode.body[start:], start) if isinstance(n, ast.If))
    mask = ast.Module(body=copy.deepcopy(encode.body[start:end]), type_ignores=[])
    proof = dict(source_importance_AST_sha256=digest(source), q03_importance_AST_sha256=digest(changed),
        ratio_call_inverse_restores_source=True,
        sole_math_change='Actual entrpy_filter entropy_keep_ratio keyword .2 to .3; original floor/topk/minimum-one/filter/aggregation/fusion/normalization/smoothing/power/greedy retained',
        source_filter_AST_sha256=digest(methods['entrpy_filter']),
        source_local_aggregation_AST_sha256=digest(methods['local_aggregation']),
        source_greedy_AST_sha256=old_proof['native_greedy_ast_sha256'],
        traced_greedy_AST_sha256=old_proof['traced_greedy_ast_sha256'],
        greedy_observer_removed_restores_original=old_proof['greedy_observer_removed_restores_original'],
        source_mask_AST_sha256=digest(mask))
    return source, changed, traced, mask, proof


def measure(torch, global_sim, dense, visual_sim, projected, *, q):
    assert q in (.2, .3)
    original = old_helpers()
    helper, _, _, _, _ = original.bind_source(torch, ARCH)
    source, changed, traced, mask, _ = nodes()
    scope = dict(self=helper, alpha=.5, beta=1., global_sim=global_sim, local_sim_all=dense,
        B=1, N=576, B_joint=1, N_joint=576)
    # Native FP32 saved full77 dense similarities are the graph input. There is
    # no new cross-modal matmul, vision forward, feature adapter or text encoder.
    program = source if q == .2 else changed
    exec(compile(ast.fix_missing_locations(copy.deepcopy(program)), '<frozen_POPE_q_ratio_importance>', 'exec'), {}, scope)
    importance = scope['importance']
    plain_ids, plain_quota = helper.greed_select(importance, visual_sim, K=1, strategy='importance_based')
    ids, quotas, gains = original.traced_selection(torch, helper, traced, importance, visual_sim)
    assert torch.equal(ids, plain_ids) and torch.equal(quotas, plain_quota) and quotas.tolist() == [32]
    mask_scope = dict(torch=torch, select_idx=ids, token_quotas=quotas, B=1, N=576)
    exec(compile(ast.fix_missing_locations(copy.deepcopy(mask)), '<frozen_POPE_native_effective_mask>', 'exec'), mask_scope)
    effective_mask = mask_scope['index_masks']
    spatial = effective_mask[0].nonzero(as_tuple=False).flatten()
    return dict(importance=importance, ordered_ids=ids, gains=gains, quotas=quotas,
        final_mask=effective_mask, spatial_ids=spatial, final_projected_FP16_sequence=projected[0, spatial],
        filtered_local_sim=scope['filtered_local_sim'], selected_entropy=scope['top_entropy_vals'],
        local_sim=scope['local_sim'])
