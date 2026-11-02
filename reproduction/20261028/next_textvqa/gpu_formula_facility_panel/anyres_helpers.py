"""Source-derived AnyRes importance and effective-mask helpers; no runtime import."""
import ast
import copy
import importlib.util
from pathlib import Path
from functools import lru_cache

COMMON = Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261028/pope/gpu_formula_facility_panel/formula_helpers.py')
spec = importlib.util.spec_from_file_location('frozen_pope_formula_helpers', COMMON)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


class RaggedQuantile(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__('Quantile member lengths differ across crop/segment rows; no implicit padding/reduction change')


def quantile_filter(torch, dense, T, q):
    probabilities = torch.softmax(dense * T, dim=1)
    entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum(dim=1)
    B, M, L = entropy.shape
    assert B == 5 and L == 77
    original = entropy.topk(max(1, int(L * q)), dim=-1, largest=False).indices
    threshold = torch.quantile(entropy, q, dim=-1, keepdim=True, interpolation='linear')
    keep = entropy <= threshold
    assert bool(keep.gather(-1, original).all())
    rows, lengths = [], []
    indices = []
    for crop in range(B):
        crop_indices = []
        for segment in range(M):
            prefix = original[crop, segment]
            mask = keep[crop, segment].clone(); mask[prefix] = False
            added = mask.nonzero(as_tuple=False).flatten()
            selected = torch.cat([prefix, added])
            lengths.append(selected.numel()); crop_indices.append(selected)
            rows.append({'crop': crop, 'segment': segment, 'source_positions': prefix.tolist(),
                         'positions': selected.tolist(), 'threshold': float(threshold[crop, segment, 0]),
                         'ties_at_threshold': int((entropy[crop, segment] == threshold[crop, segment, 0]).sum()),
                         'added_positions': added.tolist()})
        indices.append(crop_indices)
    details = {'candidate_domain': 'All77 original candidates, source top15 order retained, extras in original position order',
               'rows': rows, 'lengths': lengths, 'all_rows_same_length': len(set(lengths)) == 1,
               'padding_or_rowwise_aggregation_used': False}
    if len(set(lengths)) != 1:
        raise RaggedQuantile(details)
    selected = torch.stack([torch.stack(row) for row in indices])
    local = dense.gather(-1, selected.unsqueeze(1).expand(-1, dense.shape[1], -1, -1))
    return local, entropy.gather(-1, selected), details


def importance(torch, helper, source, eq10, global_sim, dense, beta, variant):
    assert variant in {'source', 'source_members_order_identity', 'eq6_quantile_membership_only', 'eq10_numerator_only'}
    B, N = global_sim.shape[:2]
    assert B == 5 and N == 576
    original = helper.entrpy_filter
    details = {}
    if variant == 'source_members_order_identity':
        def identity(*args, **kwargs):
            result = original(*args, **kwargs)
            details['control'] = 'Return exact original members/order/tensors unchanged'
            return result
        helper.entrpy_filter = identity
    elif variant == 'eq6_quantile_membership_only':
        def quantile(local_sim_all, T=100., entropy_keep_ratio=.2):
            local, ent, info = quantile_filter(torch, local_sim_all, T, entropy_keep_ratio)
            details.update(info)
            return local, ent
        helper.entrpy_filter = quantile
    scope = {'self': helper, 'alpha': .5, 'beta': beta, 'global_sim': global_sim,
             'local_sim_all': dense, 'B': B, 'N': N, 'B_joint': 1, 'N_joint': B * N}
    try:
        code = eq10 if variant == 'eq10_numerator_only' else source
        exec(compile(ast.fix_missing_locations(copy.deepcopy(code)), '<source_AnyRes_global_normalization>', 'exec'), {}, scope)
    finally:
        helper.entrpy_filter = original
    return scope['importance'], details


def traced_select(torch, helper, traced, weights, sim):
    trace = []
    namespace = {'torch': torch, 'trace': trace}
    tree = ast.Module(body=[copy.deepcopy(traced)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), '<source_AnyRes_greedy_trace>', 'exec'), namespace)
    ids, quotas = namespace['greed_select'](helper, weights, sim, K=5, strategy='importance_based')
    return ids, quotas, torch.stack(trace)


def effective_mask(torch, ids, quotas):
    scope = {'select_idx': ids, 'token_quotas': quotas, 'B': 5, 'N': 576}
    exec(compile(copy.deepcopy(source_mask_ast()), '<source_create_output_masks>', 'exec'), {'torch': torch}, scope)
    return scope['index_masks']


@lru_cache(maxsize=1)
def source_mask_ast():
    path = Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/LLaVA/llava/model/llava_arch.py')
    encode = common.source_parts(path)[0]['encode_images']
    start = next(i for i, node in enumerate(encode.body) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'device'
                 and isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name)
                 and node.value.value.id == 'select_idx')
    end = next(i for i in range(start + 1, len(encode.body)) if isinstance(encode.body[i], ast.If))
    return ast.fix_missing_locations(ast.Module(body=copy.deepcopy(encode.body[start:end]), type_ignores=[]))


def effective_ids(mask, ids, quotas):
    prefixes = [ids[crop, :int(quotas[crop])].tolist() for crop in range(5)]
    spatial = [mask[crop].nonzero(as_tuple=False).flatten().tolist() for crop in range(5)]
    return prefixes, spatial
