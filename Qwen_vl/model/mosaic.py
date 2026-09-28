"""
MosaicPrune -- adaptive-resolution spatial token compression for Qwen3-VL.

Instead of scoring individual visual tokens and keeping a top-k subset, we
partition the merged visual token grid (32x32 for 1024x1024 inputs) into a
quadtree whose leaves are refined adaptively, and mean-pool each leaf into a
single token. Every region of the image keeps at least one representation;
fine-grained regions simply end up with more tokens.

Three first-phase partition rules (no learned parameters, no attention, no
query guidance):

  uniform      regular pooling; the only spatially-local, non-adaptive rule
  random       same adaptive quadtree mechanics, random split choice --
               isolates the value of the hierarchy itself from the signal
  dispersion   split the leaf with the highest within-block feature
               dispersion D(B) = mean_i ||x_i - mean(x_B)||^2

Budget mechanics: a quadtree starts with 1 leaf and each quadrant split adds
3, so K = 256 is reached exactly with 85 quadrant splits. For budgets not
congruent to 1 (mod 3), the loop finishes with half-splits (+1 each). Leaves
are emitted in raster order so the spatial reading order of the sequence is
preserved.

All features used here are the *post-merger* LLM-space visual embeddings (the
same tensor EADP prunes), so a leaf cell is already a 2x2 patch aggregate and
the finest leaf (1x1 cells) is the native resolution the LLM consumes.
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Tuple

import numpy as np
import torch


MOSAIC_MODES = ("uniform", "random", "dispersion")


# ---------------------------------------------------------------------------
# Leaf bookkeeping
# ---------------------------------------------------------------------------

class _Leaf:

    __slots__ = ("top", "left", "h", "w", "score")

    def __init__(self, top: int, left: int, h: int, w: int, score: float = 0.0):
        self.top, self.left, self.h, self.w = top, left, h, w
        self.score = score

    @property
    def can_quad_split(self) -> bool:
        return self.h >= 2 and self.w >= 2

    @property
    def can_half_split(self) -> bool:
        return self.h >= 2 or self.w >= 2

    def key(self) -> Tuple[int, int]:
        # raster-order tie-breaker for deterministic heaps
        return (self.top, self.left)


# ---------------------------------------------------------------------------
# Dispersion integral images (O(1) per block query, float32 accumulation)
# ---------------------------------------------------------------------------

def _integral_images(feats: torch.Tensor, grid_h: int, grid_w: int, device=None):
    """feats: (grid_h*grid_w, D) -> per-cell sum-of-vectors and sum-of-squares
    integral images of shape (grid_h+1, grid_w+1, D) and (grid_h+1, grid_w+1).

    The partition loop reads one scalar dispersion per candidate leaf, so the
    integral images are materialised on CPU (a single ~16 MB copy) to keep the
    whole split loop sync-free."""
    x = feats.to(torch.float32)
    sq = (x * x).sum(dim=-1)
    xg = x.view(grid_h, grid_w, -1)
    sqg = sq.view(grid_h, grid_w)
    dev = device if device is not None else x.device
    s1 = torch.zeros(grid_h + 1, grid_w + 1, xg.shape[-1], device=dev)
    s2 = torch.zeros(grid_h + 1, grid_w + 1, device=dev)
    s1[1:, 1:] = xg.cumsum(0).cumsum(1)
    s2[1:, 1:] = sqg.cumsum(0).cumsum(1)
    return s1, s2


def _block_sums(s1, s2, top, left, h, w):
    a = s1[top + h, left + w] + s1[top, left] - s1[top + h, left] - s1[top, left + w]
    b = s2[top + h, left + w] + s2[top, left] - s2[top + h, left] - s2[top, left + w]
    return a, b


def block_dispersion(s1, s2, leaf) -> float:
    """D(B) = mean_i ||x_i - mean(x_B)||^2 = E||x||^2 - ||E x||^2.

    ``s1``/``s2`` may be numpy arrays (fast path used by the split loop)."""
    n = leaf.h * leaf.w
    a, b = _block_sums(s1, s2, leaf.top, leaf.left, leaf.h, leaf.w)
    return float((b - (a * a).sum() / n) / n)


def _refresh_children_scores(s1, s2, children: List[_Leaf]) -> None:
    for ch in children:
        ch.score = block_dispersion(s1, s2, ch) if (ch.h > 1 or ch.w > 1) else 0.0


# ---------------------------------------------------------------------------
# Partition builders
# ---------------------------------------------------------------------------

def uniform_partition(grid_h: int, grid_w: int, K: int) -> List[_Leaf]:
    """Regular pooling: pick per-axis factors (fh, fw) with exact division and
    (grid_h/fh)*(grid_w/fw) == K, preferring the most square-ish pooling."""
    best = None
    for fh in range(1, grid_h + 1):
        if grid_h % fh:
            continue
        for fw in range(1, grid_w + 1):
            if grid_w % fw:
                continue
            if (grid_h // fh) * (grid_w // fw) != K:
                continue
            aspect = abs(np.log(fh) - np.log(fw))
            cand = (aspect, fh, fw)
            if best is None or cand < best:
                best = cand
    if best is None:
        raise ValueError(f"no uniform pooling achieves K={K} on {grid_h}x{grid_w}")
    _, fh, fw = best
    leaves = []
    for t in range(0, grid_h, fh):
        for l in range(0, grid_w, fw):
            leaves.append(_Leaf(t, l, fh, fw))
    return leaves


def adaptive_partition(feats: torch.Tensor, grid_h: int, grid_w: int, K: int,
                       mode: str, seed: int = 0) -> List[_Leaf]:
    """Quadtree partition to exactly K leaves. 'random' or 'dispersion'.

    Runs entirely on the CPU against a pre-materialised integral image so the
    split loop never synchronises with the GPU."""
    if mode not in ("random", "dispersion"):
        raise ValueError(mode)
    if K < 1 or K > grid_h * grid_w:
        raise ValueError(f"K={K} out of range for {grid_h}x{grid_w}")

    s1, s2 = _integral_images(feats, grid_h, grid_w, device="cpu")
    # numpy views: the split loop issues thousands of tiny block queries, and
    # numpy indexing overhead is ~10x lower than torch CPU tensor ops.
    s1 = s1.numpy()
    s2 = s2.numpy()
    rng = np.random.default_rng(seed)

    root = _Leaf(0, 0, grid_h, grid_w)
    root.score = block_dispersion(s1, s2, root)
    leaves = [root]

    def quad_children(leaf: _Leaf) -> List[_Leaf]:
        h2, w2 = leaf.h // 2, leaf.w // 2
        kids = [
            _Leaf(leaf.top, leaf.left, h2, w2),
            _Leaf(leaf.top, leaf.left + w2, h2, w2),
            _Leaf(leaf.top + h2, leaf.left, h2, w2),
            _Leaf(leaf.top + h2, leaf.left + w2, h2, w2),
        ]
        _refresh_children_scores(s1, s2, kids)
        return kids

    def half_children(leaf: _Leaf) -> List[_Leaf]:
        if leaf.h >= leaf.w and leaf.h >= 2:
            h1 = leaf.h // 2
            parts = [(leaf.top, leaf.left, h1, leaf.w),
                     (leaf.top + h1, leaf.left, leaf.h - h1, leaf.w)]
        elif leaf.w >= 2:
            w1 = leaf.w // 2
            parts = [(leaf.top, leaf.left, leaf.h, w1),
                     (leaf.top, leaf.left + w1, leaf.h, leaf.w - w1)]
        else:
            raise AssertionError("half-split of an unsplittable leaf")
        kids = [_Leaf(*p) for p in parts]
        _refresh_children_scores(s1, s2, kids)
        return kids

    while len(leaves) < K:
        quadable = [lf for lf in leaves if lf.can_quad_split]
        if len(leaves) + 3 <= K and quadable:
            if mode == "dispersion":
                pick = max(quadable, key=lambda lf: (lf.score, tuple(-v for v in lf.key())))
            else:
                pick = quadable[int(rng.integers(len(quadable)))]
            leaves.remove(pick)
            leaves.extend(quad_children(pick))
        else:
            halfable = [lf for lf in leaves if lf.can_half_split]
            if not halfable:
                raise RuntimeError("no splittable leaf left before reaching K")
            if mode == "dispersion":
                pick = max(halfable, key=lambda lf: (lf.score, tuple(-v for v in lf.key())))
            else:
                pick = halfable[int(rng.integers(len(halfable)))]
            leaves.remove(pick)
            leaves.extend(half_children(pick))

    leaves.sort(key=lambda lf: lf.key())
    return leaves


def build_partition(feats: torch.Tensor, grid_h: int, grid_w: int, K: int,
                    mode: str, seed: int = 0) -> List[_Leaf]:
    if mode == "uniform":
        return uniform_partition(grid_h, grid_w, K)
    return adaptive_partition(feats, grid_h, grid_w, K, mode, seed=seed)


# ---------------------------------------------------------------------------
# Pooling + statistics
# ---------------------------------------------------------------------------

def pool_leaves(feats: torch.Tensor, grid_h: int, grid_w: int,
                leaves: List[_Leaf]) -> torch.Tensor:
    """Mean-pool each leaf into one token via an integral-image gather
    (a handful of kernels instead of one slice-mean per leaf).

    Returns (num_leaves, D), leaves in raster order."""
    s1, _ = _integral_images(feats, grid_h, grid_w, device=feats.device)
    t = torch.tensor([lf.top for lf in leaves], device=feats.device)
    l = torch.tensor([lf.left for lf in leaves], device=feats.device)
    b = t + torch.tensor([lf.h for lf in leaves], device=feats.device)
    r = l + torch.tensor([lf.w for lf in leaves], device=feats.device)
    sums = s1[b, r] - s1[t, r] - s1[b, l] + s1[t, l]          # (K, D)
    areas = ((b - t) * (r - l)).unsqueeze(1)
    return (sums / areas).to(feats.dtype)


def leaf_statistics(leaves: List[_Leaf]) -> dict:
    """Leaf-size histograms: by side length (cells) and by area."""
    by_side, by_area = {}, {}
    for lf in leaves:
        side = f"{lf.h}x{lf.w}"
        area = lf.h * lf.w
        by_side[side] = by_side.get(side, 0) + 1
        by_area[area] = by_area.get(area, 0) + 1
    n = len(leaves)
    return {
        "n_leaves": n,
        "pct_by_side": {k: round(100.0 * v / n, 2) for k, v in sorted(by_side.items())},
        "pct_by_area": {k: round(100.0 * v / n, 2) for k, v in sorted(by_area.items())},
    }


# ---------------------------------------------------------------------------
# Entry point used by the model wrapper
# ---------------------------------------------------------------------------

def stats_path_for(mode: str, K: int) -> str:
    root = os.environ.get(
        "MOSAIC_STATS_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "outputs", "discovery", "mosaic"),
    )
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"stats_{mode}_{K}.jsonl")


def mosaic_compress(image_features: torch.Tensor, grid_thw: torch.Tensor,
                    visual_token_num: int, mode: str, seed: int = 0,
                    stats_file: str = None):
    """Compress post-merger visual features with an adaptive-resolution map.

    Args mirror VisualTokenPruner.forward: image_features (total_tokens, D)
    concatenated across images, grid_thw (num_images, 3).

    Returns (pooled_features, split_sizes) and, when ``stats_file`` is set,
    appends one JSON line per image with the partition statistics and the
    wall-clock cost of the partition + pooling.
    """
    spatial_merge_size = 2  # Qwen3-VL convention
    split_sizes = (grid_thw.prod(-1) // (spatial_merge_size ** 2)).tolist()
    feature_list = torch.split(image_features, split_sizes, dim=0)

    pooled_list = []
    out_split_sizes = []
    for img_idx, img_feats in enumerate(feature_list):
        t_i, h_i, w_i = grid_thw[img_idx].tolist()
        grid_h = int(h_i) // spatial_merge_size
        grid_w = int(w_i) // spatial_merge_size
        K = min(visual_token_num, grid_h * grid_w)

        t0 = time.perf_counter()
        if K >= grid_h * grid_w:
            # no compression: one leaf per cell, raster order
            leaves = [_Leaf(t, l, 1, 1) for t in range(grid_h) for l in range(grid_w)]
        else:
            leaves = build_partition(img_feats, grid_h, grid_w, K, mode, seed=seed)
        pooled = pool_leaves(img_feats, grid_h, grid_w, leaves)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pooled_list.append(pooled)
        out_split_sizes.append(pooled.shape[0])

        if stats_file is not None:
            rec = {"grid": [grid_h, grid_w], "target_K": int(visual_token_num),
                   "actual_K": int(pooled.shape[0]), "mode": mode, "seed": seed,
                   "compress_ms": round(elapsed_ms, 3),
                   **leaf_statistics(leaves)}
            with open(stats_file, "a") as f:
                f.write(json.dumps(rec) + "\n")

    return torch.cat(pooled_list, dim=0), out_split_sizes


def leaves_to_mask(leaves: List[_Leaf], grid_h: int, grid_w: int) -> np.ndarray:
    """Debug helper: (grid_h, grid_w) int array of leaf ids in raster order."""
    m = np.zeros((grid_h, grid_w), dtype=np.int64)
    for i, lf in enumerate(leaves):
        m[lf.top:lf.top + lf.h, lf.left:lf.left + lf.w] = i
    return m
