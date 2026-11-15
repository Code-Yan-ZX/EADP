"""VisionZip assignment diagnostics with fixed budgets and native anchors.

The historical arm calls the existing E0 port verbatim.  The corrected arms
use ``anchor + mean(other members)`` for V and every DeepStack stream.  All
arms keep the same dominant set; region arms repair contextual-anchor coverage
before comparing unrestricted and same-region assignments on identical anchors.
No unassigned token is discarded and no extra anchor is silently allocated.

``select_binding(V, DS, attn, key, K, xy, regions, variant, seed)`` returns
``(keep_idx, V_sel, DS_sel, diag)``.  ``regions`` is either a length-N integer
label vector or a dict with ``labels``/``field_labels`` and optional
``line_labels``.  ``xy`` contains original token centers [N,2] in [x,y] order;
normalized page coordinates are preferred.  The variant may be a string or a
dict with ``name`` and optional ``dominant_num``/``contextual_num``.  Defaults
at N=1024,K=256 are D=205,C=51; ``historical_exact`` always uses the E0 ratio.

JSON-compatible diagnostics are separate from ``diag['tensors']``.  The latter
contains cluster membership tensors for subsequent position experiments.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class AnchorCoverageError(ValueError):
    """An honest invalid arm, with machine-readable coverage diagnostics."""

    def __init__(self, message: str, diag: dict):
        super().__init__(message)
        self.diag = diag


def _options(variant):
    options = {"name": variant} if isinstance(variant, str) else dict(variant)
    if not isinstance(options.get("name"), str):
        raise ValueError("variant must be a name or a dict containing name")
    aliases = {"historical": "historical_exact", "exact": "historical_exact",
               "fixed": "baseline_fixed", "field": "field_oracle",
               "line": "line_oracle", "random": "random_oracle",
               "spatial": "spatial_oracle"}
    options["name"] = aliases.get(options["name"], options["name"])
    allowed = {"historical_exact", "baseline_fixed"} | {
        f"{region}_{mode}" for region in ("field", "line", "random", "spatial")
        for mode in ("oracle", "anchor_control")}
    if options["name"] not in allowed:
        raise ValueError(f"unknown binding selector: {options['name']}")
    return options


def _budgets(N, K, options):
    if not (1 < K <= N):
        raise ValueError("require 1 < K <= N")
    if options["name"] == "historical_exact":
        dominant = int(round(K * 0.65 / (0.65 + 0.05)))
        contextual = max(K - dominant, 1)
    elif "dominant_num" in options:
        dominant = int(options["dominant_num"])
        contextual = int(options.get("contextual_num", K - dominant))
    elif "contextual_num" in options:
        contextual = int(options["contextual_num"])
        dominant = K - contextual
    elif "dominant_fraction" in options:
        dominant = int(round(K * float(options["dominant_fraction"])))
        contextual = K - dominant
    else:
        contextual = int(round(N * float(options.get("contextual_fraction", 0.05))))
        contextual = min(K - 1, max(1, contextual))
        dominant = K - contextual
    if dominant < 1 or contextual < 1 or dominant + contextual != K:
        raise ValueError("dominant/contextual counts must be positive and sum exactly to K")
    return dominant, contextual


def _initial_sets(attn, dominant_num, contextual_num):
    N = int(attn.numel())
    dominant = torch.topk(attn, dominant_num).indices
    dominant_mask = torch.zeros(N, dtype=torch.bool, device=attn.device)
    dominant_mask[dominant] = True
    contextual = torch.nonzero(~dominant_mask, as_tuple=True)[0]
    step = max(1, contextual.numel() // contextual_num)
    local = torch.arange(0, contextual.numel(), step, device=attn.device)[:contextual_num]
    anchors = contextual[local]
    if anchors.numel() != contextual_num:
        raise AnchorCoverageError("not enough contextual candidates for the anchor budget", {
            "valid": False, "reason": "insufficient_contextual_candidates",
            "requested_contextual": contextual_num, "actual_contextual": int(anchors.numel())})
    return dominant, contextual, anchors


def _label_vector(regions, N, device, name="field"):
    if isinstance(regions, dict):
        if name == "line":
            value = regions.get("line_labels")
        else:
            value = regions.get("labels", regions.get("field_labels"))
    else:
        value = regions if name != "line" else None
    if value is None:
        raise ValueError(f"{name} labels are required for this selector")
    raw = torch.as_tensor(value, device=device)
    if raw.ndim != 1 or raw.numel() != N:
        raise ValueError("region labels must have shape [N]")
    if raw.is_floating_point() and not bool((raw == raw.round()).all()):
        raise ValueError("region labels must be integer identifiers")
    return raw.long()


def _default_xy(N, device):
    side = math.isqrt(N)
    if side * side != N:
        raise ValueError("pass xy explicitly for a non-square visual raster")
    i = torch.arange(N, device=device)
    return torch.stack(((i % side + 0.5) / side,
                        (i // side + 0.5) / side), dim=-1)


def _spatial_labels(xy, N, device, bins):
    if bins < 1:
        raise ValueError("spatial_bins must be positive")
    xy = _default_xy(N, device) if xy is None else torch.as_tensor(xy, device=device).float()
    if tuple(xy.shape) != (N, 2) or not bool(torch.isfinite(xy).all()):
        raise ValueError("xy must be finite [N,2] token centers in [x,y] order")
    if bool((xy >= 0).all() and (xy <= 1).all()):
        normalized = xy
        mode = "page_normalized"
    else:
        # Token-grid/pixel centers: include half a cell at either edge, rather
        # than treating the first/last centers as the page boundaries.
        axes = []
        for axis in range(2):
            unique = xy[:, axis].unique(sorted=True)
            spacing = (unique[1:] - unique[:-1]).median() if unique.numel() > 1 else xy.new_tensor(1.0)
            extent = unique[-1] - unique[0] + spacing
            axes.append((xy[:, axis] - unique[0] + 0.5 * spacing) / extent)
        normalized = torch.stack(axes, dim=-1)
        mode = "inferred_from_raster_centers"
    cells = (normalized * bins).floor().long().clamp(0, bins - 1)
    return cells[:, 1] * bins + cells[:, 0], mode


def _histogram(labels):
    values, counts = labels.unique(sorted=True, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _repair_anchors(anchors, contextual, labels, attn):
    """Use exactly one replacement per missing contextual label, if feasible."""
    needed = labels[contextual].unique(sorted=True)
    present = set(int(x) for x in labels[anchors].tolist())
    missing = [int(x) for x in needed.tolist() if int(x) not in present]
    diag = dict(original_anchor_idx=anchors.tolist(),
                original_missing_labels=missing,
                required_contextual_labels=int(needed.numel()),
                contextual_label_counts=_histogram(labels[contextual]),
                replacements=[], replacement_count=0)
    if needed.numel() > anchors.numel():
        diag.update(valid=False, reason="insufficient_contextual_budget_for_labels",
                    contextual_budget=int(anchors.numel()))
        raise AnchorCoverageError("more nonempty contextual labels than available anchors", diag)
    repaired = anchors.clone()
    counts = {int(label): int((labels[repaired] == label).sum()) for label in needed.tolist()}
    for label in missing:
        candidates = contextual[labels[contextual] == label]  # sorted raster
        added = candidates[attn[candidates].argmax()]          # ties: first raster token
        removable = torch.tensor([counts[int(labels[index])] >= 2
                                  for index in repaired.tolist()],
                                 dtype=torch.bool, device=anchors.device)
        eligible_rows = removable.nonzero(as_tuple=True)[0]
        if eligible_rows.numel() == 0:
            diag.update(valid=False, reason="no_redundant_anchor_available_for_repair")
            raise AnchorCoverageError("coverage repair cannot preserve its anchor budget", diag)
        removed_row = eligible_rows[attn[repaired[eligible_rows]].argmin()]
        removed = int(repaired[removed_row])
        removed_label = int(labels[removed])
        repaired[removed_row] = added
        counts[removed_label] -= 1
        counts[label] = counts.get(label, 0) + 1
        diag["replacements"].append(dict(removed=removed, added=int(added),
                                         removed_label=removed_label, added_label=label))
    repaired = repaired.sort().values
    diag.update(anchor_idx=repaired.tolist(), replacement_count=len(missing),
                final_missing_labels=[int(x) for x in needed.tolist()
                                      if not bool((labels[repaired] == x).any())])
    if diag["final_missing_labels"]:
        diag.update(valid=False, reason="coverage_repair_failed")
        raise AnchorCoverageError("coverage repair left an unrepresented region", diag)
    return repaired, diag


def _assign(key, contextual, anchors, labels=None):
    assignees = contextual[~torch.isin(contextual, anchors)]
    metric = key / key.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    similarity = torch.bmm(metric[assignees].unsqueeze(0),
                           metric[anchors].unsqueeze(0).transpose(1, 2))[0]
    if labels is not None:
        allowed = labels[assignees, None] == labels[anchors][None, :]
        if not bool(allowed.any(dim=-1).all()):
            raise AnchorCoverageError("a token has no same-label anchor", {
                "valid": False, "reason": "unassigned_tokens_after_coverage_repair",
                "unassigned_count": int((~allowed.any(-1)).sum())})
        similarity = similarity.masked_fill(~allowed, -torch.inf)
    assignment = similarity.argmax(dim=-1)
    return assignees, assignment


def _aggregate(stream, keep, anchors, assignees, assignment):
    counts = torch.zeros(anchors.numel(), device=stream.device)
    counts.index_add_(0, assignment, torch.ones_like(assignment, dtype=counts.dtype))
    sums = torch.zeros(anchors.numel(), stream.shape[1], device=stream.device, dtype=stream.dtype)
    sums.index_add_(0, assignment, stream[assignees])
    rows = stream[anchors] + sums / counts.clamp_min(1)[:, None]
    out = stream[keep].clone()
    out[torch.isin(keep, anchors)] = rows.to(stream.dtype)
    return out


@torch.no_grad()
def select_binding(V, DS, attn, key, K, xy, regions, variant, seed=0):
    """Select/merge exactly K native visual anchors with auditable membership.

    ``field_anchor_control`` and ``field_oracle`` repair to identical anchors;
    only the latter masks cross-label assignments.  The same pairing applies
    to ``line``, ``random`` and ``spatial``.  Random labels preserve the field
    histogram across all N tokens exactly; spatial bins are an equal-area
    control, explicitly not a label-size-matched control.
    """
    options = _options(variant)
    name = options["name"]
    if V.ndim != 2 or not V.is_floating_point():
        raise ValueError("V must be a floating-point [N,D] tensor")
    N, width = V.shape
    attn = torch.as_tensor(attn, device=V.device).reshape(-1)
    key = torch.as_tensor(key, device=V.device)
    if attn.numel() != N or key.ndim != 2 or key.shape[0] != N:
        raise ValueError("attention and key arrays must align with V's N tokens")
    if not bool(torch.isfinite(attn).all() and torch.isfinite(key).all()):
        raise ValueError("attention/key inputs must be finite")
    if DS is not None and any(stream.shape != V.shape or stream.device != V.device for stream in DS):
        raise ValueError("DeepStack streams must match V's shape and device")
    dominant_num, contextual_num = _budgets(N, int(K), options)
    dominant, contextual, anchors = _initial_sets(attn, dominant_num, contextual_num)
    original_anchors = anchors.clone()
    labels = None
    spatial_mode = None
    repair = dict(original_anchor_idx=anchors.tolist(), anchor_idx=anchors.tolist(),
                  replacement_count=0, replacements=[], original_missing_labels=[],
                  final_missing_labels=[])
    if name.startswith(("field_", "line_", "random_", "spatial_")):
        label_source = name.split("_", 1)[0]
        if label_source == "spatial":
            labels, spatial_mode = _spatial_labels(xy, N, V.device, int(options.get("spatial_bins", 4)))
        else:
            labels = _label_vector(regions, N, V.device, "line" if label_source == "line" else "field")
            if label_source == "random":
                generator = torch.Generator(device="cpu").manual_seed(int(seed))
                permutation = torch.randperm(N, generator=generator).to(V.device)
                labels = labels[permutation]
        try:
            anchors, repair = _repair_anchors(anchors, contextual, labels, attn)
        except AnchorCoverageError as error:
            error.diag.update(variant=name, N=N, K=int(K), dominant_num=dominant_num,
                              contextual_num=contextual_num, seed=int(seed))
            raise
    constrained = name.endswith("_oracle")
    assignees, assignment = _assign(key, contextual, anchors, labels if constrained else None)
    keep = torch.cat((dominant, anchors)).sort().values
    if keep.numel() != K or keep.unique().numel() != K:
        raise RuntimeError("selector did not preserve exactly K unique anchors")
    if name == "historical_exact":
        # Importing this existing port does not load model weights or a GPU.
        from model.baselines.visionzip import select_visionzip
        exact_keep, V_sel, DS_sel = select_visionzip(V, DS, attn, key, K)
        if not torch.equal(keep, exact_keep):
            raise RuntimeError("historical audit bookkeeping differs from the existing native port")
    else:
        V_sel = _aggregate(V, keep, anchors, assignees, assignment)
        DS_sel = None if DS is None else [_aggregate(ds, keep, anchors, assignees, assignment) for ds in DS]

    cluster_id = torch.full((N,), -1, dtype=torch.long, device=V.device)
    cluster_id[anchors] = torch.arange(contextual_num, device=V.device)
    cluster_id[assignees] = assignment
    sizes = torch.bincount(cluster_id[contextual], minlength=contextual_num)
    if int(sizes.sum()) != N - dominant_num or bool((cluster_id[contextual] < 0).any()):
        raise RuntimeError("a non-dominant token was silently lost")
    clusters = {str(int(anchor)): contextual[cluster_id[contextual] == index].tolist()
                for index, anchor in enumerate(anchors)}
    cross = (int((labels[assignees] != labels[anchors[assignment]]).sum())
             if labels is not None else None)
    if constrained and cross != 0:
        raise RuntimeError("same-label assignment contains a cross-label member")
    diag = dict(valid=True, variant=name, N=N, K=int(K), dominant_num=dominant_num,
                contextual_num=contextual_num, seed=int(seed),
                dominant_idx=dominant.tolist(), **repair,
                constrained_assignment=constrained, clusters=clusters,
                cluster_sizes=sizes.tolist(), empty_assignee_clusters=int((sizes == 1).sum()),
                empty_clusters=int((sizes == 0).sum()), unassigned_count=0,
                non_dominant_covered=int(sizes.sum()),
                all_tokens_accounted_once=int(sizes.sum()) + dominant_num == N,
                cross_label_assignees=cross, anchor_coordinates="original native raster subset",
                stream_rule="historical V=anchor+mean,DS=mean" if name == "historical_exact"
                else "V and every DS = anchor + mean(other members)",
                tensor_diagnostics_key="tensors")
    if labels is not None:
        diag["label_counts"] = _histogram(labels)
    if spatial_mode is not None:
        diag.update(spatial_bins=int(options.get("spatial_bins", 4)),
                    spatial_coordinate_mode=spatial_mode,
                    spatial_matches_field_label_sizes=False)
    diag["tensors"] = dict(keep_idx=keep, dominant_idx=dominant,
                            original_anchor_idx=original_anchors, anchor_idx=anchors,
                            contextual_idx=contextual, assignee_idx=assignees,
                            assignment=assignment, assignments=assignment,
                            cluster_id=cluster_id, cluster_sizes=sizes,
                            anchor_rows=torch.searchsorted(keep, anchors), labels=labels)
    return keep, V_sel, DS_sel, diag
