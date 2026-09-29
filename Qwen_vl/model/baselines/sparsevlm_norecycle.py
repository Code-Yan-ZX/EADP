"""SparseVLM without token recycling -- the prereg §4 downgrade.

The recycling port passes the N4 mechanical invariants but produces degenerate
generation (20/30 capped at 2048 new tokens in the 30-question smoke vs 0/30
for every other arm) -- i.e. a semantic defect in the merged-token insertion.
Per prereg §4 ("若 Qwen3 上无法完整移植,降级为不含 recycling 的版本,并明确
标注") the recycling-free variant is provided under a distinct arm id so the
full version's defect is not silently attributed to SparseVLM.  Everything
else (pruning layers, schedule, text-guided scoring) is identical to
sparsevlm.py.
"""
from model.baselines import sparsevlm as _full


def build_prune_layers(K, n_vis=1024):
    def make_fn(stage):
        def fn(ctx):
            keep_local, info, _ = _full.build_prune_layers(K, n_vis)[ctx["layer_idx"]][1](ctx)
            return keep_local, info, None
        return fn
    return {l: ("after", make_fn(i)) for i, l in enumerate(_full.PRUNE_LAYERS)}


def selector(K, ctx):
    return dict(keep_idx=None, prune_layers=build_prune_layers(K, ctx["prep"]["n_vis"]))
