"""Answer-independent, budget-matched geometry on the native 32 x 32 grid.

Coordinates are half-open boxes [row, column, height, width].  Flattened indices
are row-major and must index V, every DeepStack stream and native positions in
exactly the same way.  These functions never resize, merge, fill a short mask,
inspect an image/question/answer, or import a model framework.

The 12 single-region actions and 12 dual-region actions have identical A and B
unit marginals.  Their *pairing* differs.  Each unit contains 104 native tokens,
including 8 of the fixed global anchors: G64 + 96 + 96 = K256 exactly.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
from typing import Any, Sequence

GRID_SIZE = 32
N = GRID_SIZE * GRID_SIZE
K = 256
Box = tuple[int, int, int, int]


def global_indices() -> list[int]:
    """The fixed 8 x 8 anchor lattice, at row/column == 0 (mod 4)."""
    return [r * GRID_SIZE + c for r in range(0, GRID_SIZE, 4)
            for c in range(0, GRID_SIZE, 4)]


def uniform256_indices() -> list[int]:
    """The fixed 16 x 16 reference lattice, at row/column == 0 (mod 2)."""
    return [r * GRID_SIZE + c for r in range(0, GRID_SIZE, 2)
            for c in range(0, GRID_SIZE, 2)]


def _box(box: Sequence[int]) -> Box:
    if len(box) != 4 or any(type(x) is not int for x in box):
        raise ValueError("box must have four integer coordinates [r, c, h, w]")
    r, c, h, w = box
    if (h, w) not in ((8, 13), (13, 8)) or r % 4 or c % 4:
        raise ValueError("unit must be 8x13 or 13x8 with a multiple-of-four start")
    if min(r, c) < 0 or r + h > GRID_SIZE or c + w > GRID_SIZE:
        raise ValueError("unit extends outside the native 32x32 grid")
    return r, c, h, w


def _indices(box: Sequence[int]) -> set[int]:
    r, c, h, w = _box(box)
    return {i * GRID_SIZE + j for i in range(r, r + h)
            for j in range(c, c + w)}


def _transpose(box: Box) -> Box:
    r, c, h, w = box
    return c, r, w, h


def unit_catalog() -> list[list[int]]:
    """All 70 legal units, in a frozen orientation/row/column order."""
    return [[r, c, h, w] for h, w in ((8, 13), (13, 8))
            for r in range(0, GRID_SIZE - h + 1, 4)
            for c in range(0, GRID_SIZE - w + 1, 4)]


def _unit(box: Box) -> dict[str, Any]:
    cells = _indices(box)
    anchors = cells.intersection(global_indices())
    if len(cells) != 104 or len(anchors) != 8:
        raise ValueError("unit does not supply exactly 96 non-anchor tokens")
    return {"box": list(box), "raw_tokens": 104, "anchor_tokens": 8,
            "extra_tokens": 96}


def _selection(a: Box, b: Box) -> tuple[list[int], dict[str, Any]]:
    ac, bc = _indices(a), _indices(b)
    if ac & bc:
        raise ValueError("selected units must not overlap, including anchors")
    keep = sorted(set(global_indices()) | ac | bc)
    if len(keep) != K:
        raise ValueError(f"selection has {len(keep)} tokens, expected {K}; no top-up")
    ar, acol, ah, aw = a
    br, bcol, bh, bw = b
    row_gap = max(0, ar - br - bh, br - ar - ah)
    col_gap = max(0, acol - bcol - bw, bcol - acol - aw)
    row_overlap = min(ar + ah, br + bh) - max(ar, br)
    col_overlap = min(acol + aw, bcol + bw) - max(acol, bcol)
    adjacent = ((row_gap == 0 and col_overlap > 0)
                or (col_gap == 0 and row_overlap > 0))
    r0, c0 = min(ar, br), min(acol, bcol)
    r1, c1 = max(ar + ah, br + bh), max(acol + aw, bcol + bw)
    coverage = {
        "grid_shape": [GRID_SIZE, GRID_SIZE], "keep_tokens": len(keep),
        "global_tokens": 64, "detail_extra_tokens": 192,
        "detail_raw_area": len(ac | bc), "anchors_inside_detail": 16,
        "detail_density": 1.0,
        "detail_connected_components": 1 if adjacent else 2,
        "row_gap": row_gap, "column_gap": col_gap,
        "bounding_box": [r0, c0, r1 - r0, c1 - c0],
        "bounding_box_area": (r1 - r0) * (c1 - c0),
        "center_distance_l1": abs(ar + ah / 2 - br - bh / 2)
                              + abs(acol + aw / 2 - bcol - bw / 2),
    }
    return keep, coverage


def build_bank() -> list[dict[str, Any]]:
    """Return 12 single then 12 dual masks, with identical A/B marginals.

    In vertical coordinates A starts at (8*ri, c), ri in {0,1,2},
    c in {0,16}.  Single B starts at (8*ri+8,c).  Dual B starts at
    (8*((ri+1)%3)+8,16-c): a permutation of the same six B units.
    Horizontal actions are exact transposes, including the global lattice.
    """
    bank = []
    for family in ("single", "dual"):
        for orientation in ("vertical", "horizontal"):
            for ri in range(3):
                for c in (0, 16):
                    a = (8 * ri, c, 8, 13)
                    b = ((8 * ri + 8, c, 8, 13) if family == "single"
                         else (8 * ((ri + 1) % 3) + 8, 16 - c, 8, 13))
                    if orientation == "horizontal":
                        a, b = _transpose(a), _transpose(b)
                    keep, coverage = _selection(a, b)
                    bank.append({
                        "action_id": f"{family}_{orientation}_r{ri}_c{c}",
                        "family": family, "orientation": orientation,
                        "pair_index": ri * 2 + c // 16,
                        "unit_a": list(a), "unit_b": list(b),
                        "units": {"a": _unit(a), "b": _unit(b)},
                        "coverage": coverage, "keep_idx": keep,
                    })
    if len({tuple(a["keep_idx"]) for a in bank}) != 24:
        raise ValueError("the preregistered 24 actions are not distinct")
    for role in ("unit_a", "unit_b"):
        counts = [Counter(tuple(a[role]) for a in bank if a["family"] == f)
                  for f in ("single", "dual")]
        if counts[0] != counts[1]:
            raise ValueError(f"single/dual {role} marginals do not match")
    return bank


def factorial_controls(dual_action: dict[str, Any], replicates: int = 2
                       ) -> list[dict[str, Any]]:
    """Return 00,10,01 for each fixed donor-pair replicate; AB is in the bank.

    FA and FB are disjoint from the original A/B and from each other, in raw
    coordinates (not merely after removing anchors).  For vertical actions,
    take the first legal unordered pairs in unit_catalog order whose three
    masks have not already occurred.  Horizontal actions use the transpose of
    that exact catalog order.  There is no answer-dependent donor selection.
    Different replicates can overlap spatially, but their donor pairs and all
    returned masks must differ.  More replicates never alter earlier choices.

    unit_a/unit_b always describe the original AB; unit_FA/unit_FB the donor
    pair.  `units` and `coverage` describe the actually selected two units.
    """
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if not isinstance(dual_action, dict):
        raise ValueError("dual_action must be a registered dual action")
    registered = {a["action_id"]: a for a in build_bank() if a["family"] == "dual"}
    canonical = registered.get(dual_action.get("action_id"))
    if canonical is None or any(dual_action.get(k) != canonical[k]
                                for k in ("family", "unit_a", "unit_b", "keep_idx")):
        raise ValueError("dual_action differs from the preregistered dual bank")
    a, b = _box(canonical["unit_a"]), _box(canonical["unit_b"])
    forbidden = _indices(a) | _indices(b)
    catalog = [tuple(x) for x in unit_catalog()]
    if canonical["orientation"] == "horizontal":
        catalog = [_transpose(x) for x in catalog]
    donors = [x for x in catalog if not _indices(x) & forbidden]
    seen = {tuple(canonical["keep_idx"])}
    controls = []
    count = 0
    for fa, fb in combinations(donors, 2):
        if _indices(fa) & _indices(fb):
            continue
        selected = [("00", "FA", fa, "FB", fb),
                    ("10", "a", a, "FB", fb),
                    ("01", "FA", fa, "b", b)]
        choices = [_selection(x, y) for _, _, x, _, y in selected]
        keys = [tuple(keep) for keep, _ in choices]
        if len(set(keys)) != 3 or any(key in seen for key in keys):
            continue
        for (condition, xn, x, yn, y), (keep, coverage) in zip(selected, choices):
            controls.append({
                "action_id": f"{canonical['action_id']}__f{count}__{condition}",
                "family": "factorial", "joint_action_id": canonical["action_id"],
                "orientation": canonical["orientation"],
                "replicate": count, "condition": condition,
                "unit_a": list(a), "unit_b": list(b),
                "unit_FA": list(fa), "unit_FB": list(fb),
                "units": {xn: _unit(x), yn: _unit(y)},
                "coverage": coverage, "keep_idx": keep,
            })
        seen.update(keys)
        count += 1
        if count == replicates:
            return controls
    raise ValueError(f"fixed donor rule supplies only {count} unique replicates for "
                     f"{canonical['action_id']}; requested {replicates}")


def geometry_report() -> dict[str, Any]:
    """JSON-safe geometry audit, including the spatial-coverage confound checks."""
    bank = build_bank()
    pools = {f: [a for a in bank if a["family"] == f] for f in ("single", "dual")}
    frequencies = {f: [sum(i in a["keep_idx"] for a in pool) for i in range(N)]
                   for f, pool in pools.items()}
    unit_marginals = {}
    for role in ("unit_a", "unit_b"):
        counts = {f: Counter(tuple(a[role]) for a in pool) for f, pool in pools.items()}
        unit_marginals[role] = counts["single"] == counts["dual"]
    return {
        "grid_shape": [GRID_SIZE, GRID_SIZE], "n": N, "k": K,
        "global_tokens": len(global_indices()), "unit_catalog_count": len(unit_catalog()),
        "unique_bank_masks": len({tuple(a["keep_idx"]) for a in bank}),
        "unit_marginals_match": unit_marginals,
        "per_token_selection_frequency_matches": frequencies["single"] == frequencies["dual"],
        "pools": {f: {
            "count": len(pool), "orientation_counts": dict(Counter(a["orientation"] for a in pool)),
            "union_coverage_tokens": sum(v > 0 for v in frequencies[f]),
            "selection_frequency_histogram": dict(sorted(Counter(frequencies[f]).items())),
            "detail_raw_areas": sorted({a["coverage"]["detail_raw_area"] for a in pool}),
            "detail_extra_tokens": sorted({a["coverage"]["detail_extra_tokens"] for a in pool}),
            "detail_component_counts": sorted({a["coverage"]["detail_connected_components"] for a in pool}),
            "bounding_box_areas": sorted({a["coverage"]["bounding_box_area"] for a in pool}),
        } for f, pool in pools.items()},
        "factorial_control_counts": {a["action_id"]: len(factorial_controls(a))
                                      for a in pools["dual"]},
        "limitations": [
            "Matched marginals do not imply that every image region is covered equally.",
            "Detail components count raw window connectivity, not connectivity of global anchors.",
            "The entire native high-resolution encoder cost is paid before these selections.",
            "Donors are geometric replacements; they are not guaranteed semantically irrelevant.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(geometry_report(), indent=2, sort_keys=True))
