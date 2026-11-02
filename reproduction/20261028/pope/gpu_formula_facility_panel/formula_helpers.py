"""Source-derived importance and traced greedy helpers; imports no tensor runtime."""
import ast
import copy
import hashlib
import types


def digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def source_parts(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LlavaMetaForCausalLM')
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    encode = methods['encode_images']
    start = next(i for i, n in enumerate(encode.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Tuple)
                 and [x.id for x in n.targets[0].elts] == ['filtered_local_sim', 'top_entropy_vals'])
    end = next(i for i, n in enumerate(encode.body) if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'importance'
               and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Pow))
    program = ast.Module(body=copy.deepcopy(encode.body[start:end + 1]), type_ignores=[])
    eq10 = copy.deepcopy(program)
    normalization = next(n for n in eq10.body if isinstance(n, ast.Assign)
                         and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'importance'
                         and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Div))
    old_numerator = copy.deepcopy(normalization.value.left)
    assert isinstance(old_numerator, ast.BinOp) and isinstance(old_numerator.op, ast.Add)
    assert isinstance(old_numerator.right, ast.Constant) and old_numerator.right.value == 1e-6
    normalization.value.left = normalization.value.left.left
    restored = copy.deepcopy(eq10)
    next(n for n in restored.body if isinstance(n, ast.Assign)
         and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'importance'
         and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Div)).value.left = old_numerator
    assert digest(restored) == digest(program)
    greedy = copy.deepcopy(methods['greed_select'])
    traced = copy.deepcopy(greedy)
    loop = next(n for n in traced.body if isinstance(n, ast.For))
    at = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'best_idx')
    observer = ast.parse('trace.append(total_gain.detach().clone())').body[0]
    loop.body.insert(at + 1, observer)
    restore_greedy = copy.deepcopy(traced)
    next(n for n in restore_greedy.body if isinstance(n, ast.For)).body.pop(at + 1)
    assert digest(restore_greedy) == digest(greedy)
    proof = {'native_importance_ast_sha256': digest(program),
             'eq10_importance_ast_sha256': digest(eq10), 'eq10_restoration_identical': True,
             'native_greedy_ast_sha256': digest(greedy), 'traced_greedy_ast_sha256': digest(traced),
             'greedy_observer_removed_restores_original': True,
             'greedy_observer': 'trace.append(total_gain.detach().clone())',
             'eq10_change': 'Only remove numerator +1e-6; denominator +1e-6 remains.'}
    return methods, program, eq10, traced, proof


def bind_source(torch, path):
    methods, source, eq10, traced, proof = source_parts(path)
    names = ['entrpy_filter', 'local_aggregation', 'spatial_smoothing', 'greed_select',
             'strategy_importance_allocation', 'strategy_global_guided_allocation']
    namespace = {'torch': torch}
    module = ast.Module(body=[copy.deepcopy(methods[n]) for n in names], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    helper = types.SimpleNamespace(visual_token_num=32, is_joint=False)
    for name in names:
        setattr(helper, name, types.MethodType(namespace[name], helper))
    return helper, source, eq10, traced, proof


def quantile_filter(torch, dense, *, T, entropy_keep_ratio):
    """Change membership only: preserve source top-k order, append extra positions."""
    probabilities = torch.softmax(dense * T, dim=1)
    entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum(dim=1)
    assert entropy.shape == (1, 1, 77), 'Only the frozen v15 one-image POPE panel is supported'
    source_indices = entropy.topk(max(1, int(77 * entropy_keep_ratio)), largest=False).indices
    threshold = torch.quantile(entropy, entropy_keep_ratio, dim=-1, keepdim=True, interpolation='linear')
    keep = entropy <= threshold
    assert bool(keep.gather(-1, source_indices).all())
    added_mask = keep.clone()
    added_mask.scatter_(-1, source_indices, False)
    added = added_mask[0, 0].nonzero(as_tuple=False).flatten()
    indices = torch.cat([source_indices[0, 0], added]).view(1, 1, -1)
    selected_entropy = entropy.gather(-1, indices)
    selected = dense.gather(-1, indices.unsqueeze(1).expand(-1, dense.shape[1], -1, -1))
    info = {'source_positions': source_indices[0, 0].tolist(), 'positions': indices[0, 0].tolist(),
            'threshold': float(threshold[0, 0, 0]), 'ties_at_threshold': int((entropy == threshold).sum()),
            'added_positions': added.tolist(), 'source_prefix_order_preserved': True}
    return selected, selected_entropy, info


def importance(torch, helper, program, eq10, global_sim, dense, alpha, beta, variant):
    assert variant in {'source', 'source_members_order_identity', 'eq6_quantile_membership_only', 'eq10_numerator_only'}
    original_filter = helper.entrpy_filter
    details = {}
    if variant == 'source_members_order_identity':
        def identity(local_sim_all, T=100., entropy_keep_ratio=.2):
            local, ent = original_filter(local_sim_all, T=T, entropy_keep_ratio=entropy_keep_ratio)
            details.update(control='Exact source members and order; return original filter tensors unchanged')
            return local, ent
        helper.entrpy_filter = identity
    elif variant == 'eq6_quantile_membership_only':
        def replacement(local_sim_all, T=100., entropy_keep_ratio=.2):
            local, ent, info = quantile_filter(torch, local_sim_all, T=T, entropy_keep_ratio=entropy_keep_ratio)
            details.update(info)
            return local, ent
        helper.entrpy_filter = replacement
    B, N = global_sim.shape[:2]
    scope = {'self': helper, 'alpha': alpha, 'beta': beta, 'global_sim': global_sim,
             'local_sim_all': dense, 'B': B, 'N': N, 'B_joint': B, 'N_joint': N}
    try:
        selected = eq10 if variant == 'eq10_numerator_only' else program
        exec(compile(ast.fix_missing_locations(copy.deepcopy(selected)), '<frozen_author_importance>', 'exec'), {}, scope)
    finally:
        helper.entrpy_filter = original_filter
    return scope['importance'], details


def traced_selection(torch, helper, traced, importance_values, sim_matrix):
    trace = []
    namespace = {'torch': torch, 'trace': trace}
    module = ast.Module(body=[copy.deepcopy(traced)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<author_greedy_readonly_trace>', 'exec'), namespace)
    selected, quotas = namespace['greed_select'](helper, importance_values, sim_matrix, K=1, strategy='importance_based')
    return selected, quotas, torch.stack(trace)
