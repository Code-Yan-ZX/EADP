"""Source-derived Qwen LLM-space formulas; importing this file imports no torch."""
import ast
import copy
import hashlib
from pathlib import Path


def digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


class RaggedQuantile(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__('Different threshold member counts; original batch arithmetic cannot be retained')


def parts(path):
    tree = ast.parse(Path(path).read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VisualTokenPruner')
    compute = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_importance')
    changed = copy.deepcopy(compute)
    assignment = next(n for n in changed.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'importance'
                      and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Div))
    numerator = assignment.value.left
    assert isinstance(numerator, ast.BinOp) and isinstance(numerator.op, ast.Add)
    assert isinstance(numerator.right, ast.Constant) and numerator.right.value == 1e-6
    assignment.value.left = copy.deepcopy(numerator.left)
    restored = copy.deepcopy(changed)
    replacement = next(n for n in restored.body if isinstance(n, ast.Assign)
                       and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'importance'
                       and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Div))
    replacement.value.left = copy.deepcopy(numerator)
    assert ast.dump(restored) == ast.dump(compute)
    greedy = functions['_greed_select_impl']
    traced = copy.deepcopy(greedy)
    loop = next(n for n in traced.body if isinstance(n, ast.For))
    position = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'best_idx')
    observer = ast.parse('trace.append(total_gain.detach().clone())').body[0]
    loop.body.insert(position, observer)
    inverse = copy.deepcopy(traced)
    inverse_loop = next(n for n in inverse.body if isinstance(n, ast.For))
    del inverse_loop.body[position]
    assert ast.dump(inverse) == ast.dump(greedy)
    entropy = functions['_entropy_filter_impl']
    topk_position = next(i for i, n in enumerate(entropy.body) if isinstance(n, ast.Assign)
                        and isinstance(n.targets[0], ast.Tuple)
                        and any(isinstance(t, ast.Name) and t.id == 'top_entropy_indices' for t in n.targets[0].elts))
    prefix = ast.Module(body=copy.deepcopy(entropy.body[:topk_position + 1]), type_ignores=[])
    proof = dict(source_compute_AST_sha256=digest(compute), Eq10_compute_AST_sha256=digest(changed),
                 Eq10_inverse_restores_source=True, source_greedy_AST_sha256=digest(greedy),
                 traced_greedy_AST_sha256=digest(traced), observer_removed_restores_source=True,
                 source_entropy_prefix_AST_sha256=digest(prefix),
                 Eq10_change='Remove only numerator +1e-6; every denominator and all other statements unchanged')
    return functions, compute, changed, traced, prefix, proof


def quantile_filter(torch, prefix, dense, T=100., entropy_keep_ratio=.2):
    scope = dict(torch=torch, local_sim_all=dense, T=T, entropy_keep_ratio=entropy_keep_ratio)
    exec(compile(ast.fix_missing_locations(copy.deepcopy(prefix)), '<original_Qwen_entropy_and_topk>', 'exec'), scope)
    entropy = scope['text_entropy']; original = scope['top_entropy_indices']
    threshold = torch.quantile(entropy, entropy_keep_ratio, dim=-1, keepdim=True, interpolation='linear')
    keep = entropy <= threshold
    assert bool(keep.gather(-1, original).all())
    rows, lengths, selections = [], [], []
    for batch in range(entropy.shape[0]):
        batch_indices = []
        for segment in range(entropy.shape[1]):
            old = original[batch, segment]
            extra_mask = keep[batch, segment].clone(); extra_mask[old] = False
            extra = extra_mask.nonzero(as_tuple=False).flatten()
            selected = torch.cat([old, extra])
            lengths.append(selected.numel()); batch_indices.append(selected)
            rows.append(dict(batch=batch, segment=segment, L=entropy.shape[-1],
                             source_positions=old.tolist(), positions=selected.tolist(),
                             added_positions=extra.tolist(), threshold=float(threshold[batch, segment, 0]),
                             ties_at_threshold=int((entropy[batch, segment] == threshold[batch, segment, 0]).sum())))
        selections.append(batch_indices)
    details = dict(rows=rows, lengths=lengths, all_rows_same_length=len(set(lengths)) == 1,
                   source_members_and_order_retained=True, padding_or_rowwise_reduction_used=False)
    if len(set(lengths)) != 1:
        raise RaggedQuantile(details)
    selected = torch.stack([torch.stack(row) for row in selections])
    local = dense.gather(-1, selected.unsqueeze(1).expand(-1, dense.shape[1], -1, -1))
    return local, entropy.gather(-1, selected), details


def bind(torch, native_module, source_path):
    functions, source, changed, traced, prefix, proof = parts(source_path)
    # Freeze references to the original callable namespace before any observers.
    original = dict(native_module.__dict__)
    def compute(pruner, features, global_text, dense_text, grid_h, grid_w, variant):
        assert variant in {'source', 'source_members_order_identity', 'eq6_quantile_membership_only', 'eq10_numerator_only'}
        namespace = dict(original); details = {}
        if variant == 'source_members_order_identity':
            def identity(*args, **kwargs):
                value = original['_entropy_filter_impl'](*args, **kwargs)
                details['operation'] = 'Return original members/order/tensors unchanged'
                return value
            namespace['_entropy_filter_impl'] = identity
        elif variant == 'eq6_quantile_membership_only':
            def quantile(dense, T=100., entropy_keep_ratio=.2):
                local, entropy, info = quantile_filter(torch, prefix, dense, T, entropy_keep_ratio)
                details.update(info)
                return local, entropy
            namespace['_entropy_filter_impl'] = quantile
        node = changed if variant == 'eq10_numerator_only' else source
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[])),
                     '<original_Qwen_compute_importance>', 'exec'), namespace)
        return namespace['compute_importance'](pruner, features, global_text, dense_text, grid_h, grid_w), details
    def select(weights, sim, token_num=256):
        ids, quotas = original['_greed_select_impl'](weights, sim, token_num)
        namespace = dict(original, trace=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(traced)], type_ignores=[])),
                     '<original_Qwen_greedy_post_gain_observer>', 'exec'), namespace)
        traced_ids, traced_quotas = namespace['_greed_select_impl'](weights, sim, token_num)
        assert torch.equal(ids, traced_ids) and torch.equal(quotas, traced_quotas)
        return ids, quotas, torch.stack(namespace['trace'])
    return compute, select, proof
