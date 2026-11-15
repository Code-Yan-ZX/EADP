"""CPU-only gates for the fixed-budget VisionZip binding oracle helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

QWEN_ROOT = Path(__file__).resolve().parents[2]
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

import torch

from scripts.binding_oracle.binding_selectors import AnchorCoverageError, select_binding


def audit_native_layout(eng, prep, keep_idx, V_sel, DS_sel):
    """Check original native coordinates/stream lengths without running a VLM."""
    seq = eng.keep_sequence(prep, keep_idx)
    full_positions, _ = eng.positions_full(prep)
    selected_positions = full_positions[:, :, seq]
    visual = (seq >= prep["img_start"]) & (seq < prep["img_end"])
    expected = full_positions[:, :, prep["img_start"] + keep_idx.to(seq.device)]
    checks = dict(exact_visual_count=int(visual.sum()) == keep_idx.numel(),
                  positions_subset=torch.equal(selected_positions[:, :, visual], expected),
                  main_stream_length=V_sel.shape[0] == keep_idx.numel(),
                  deepstack_stream_lengths=DS_sel is None or all(ds.shape[0] == keep_idx.numel() for ds in DS_sel),
                  ordered_unique_anchors=bool((keep_idx[1:] > keep_idx[:-1]).all()))
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def run_cpu_gates():
    from model.baselines.visionzip import select_visionzip
    generator = torch.Generator(device="cpu").manual_seed(20261108)
    V = torch.randn(1024, 12, generator=generator).bfloat16()
    DS = [V.clone()] + [torch.randn(1024, 12, generator=generator).bfloat16() for _ in range(2)]
    attn = torch.randn(1024, generator=generator)
    key = torch.randn(1024, 16, generator=generator)
    keep, main, stack, hist = select_binding(V, DS, attn, key, 256, None, None, "historical_exact", 7)
    reference = select_visionzip(V, DS, attn, key, 256)
    assert torch.equal(keep, reference[0]) and torch.equal(main, reference[1])
    assert all(torch.equal(a, b) for a, b in zip(stack, reference[2]))
    assert (hist["dominant_num"], hist["contextual_num"]) == (238, 18)
    _, fixed_main, fixed_stack, fixed_hist = select_binding(
        V, DS, attn, key, 256, None, None,
        dict(name="baseline_fixed", dominant_num=238, contextual_num=18), 7)
    assert torch.equal(fixed_main, main)
    assert torch.equal(fixed_main, fixed_stack[0])
    assert not torch.equal(main, stack[0])  # historical DS omits anchor features
    base = select_binding(V, DS, attn, key, 256, None, None, "baseline_fixed", 7)
    assert (base[3]["dominant_num"], base[3]["contextual_num"]) == (205, 51)
    original_anchors = base[3]["tensors"]["anchor_idx"]
    candidates = base[3]["tensors"]["assignee_idx"]
    labels = torch.zeros(1024, dtype=torch.long)
    # Six singleton regions initially have no anchor, forcing six explicit
    # replacements and singleton clusters (no other members).
    labels[candidates[:6]] = torch.arange(1, 7)
    regions = dict(labels=labels, line_labels=labels.clone())
    control = select_binding(V, DS, attn, key, 256, None, regions, "field_anchor_control", 7)
    oracle = select_binding(V, DS, attn, key, 256, None, regions, "field_oracle", 7)
    assert torch.equal(control[0], oracle[0])
    assert oracle[3]["replacement_count"] == 6
    assert oracle[3]["cross_label_assignees"] == 0
    assert oracle[3]["empty_assignee_clusters"] >= 6 and oracle[3]["empty_clusters"] == 0
    assert not torch.equal(original_anchors, oracle[3]["tensors"]["anchor_idx"])
    assert oracle[3]["all_tokens_accounted_once"] and oracle[3]["non_dominant_covered"] == 819
    assert torch.equal(oracle[1], oracle[2][0])
    # Every singleton anchor remains its original feature in ALL streams.
    anchor_ids = oracle[3]["tensors"]["anchor_idx"]
    sizes = oracle[3]["tensors"]["cluster_sizes"]
    for anchor in anchor_ids[sizes == 1]:
        row = torch.searchsorted(oracle[0], anchor)
        assert torch.equal(oracle[1][row], V[anchor])
        assert all(torch.equal(out[row], src[anchor]) for out, src in zip(oracle[2], DS))
    line = select_binding(V, DS, attn, key, 256, None, regions, "line_oracle", 7)
    assert torch.equal(line[0], oracle[0]) and torch.equal(line[1], oracle[1])
    rnd = select_binding(V, DS, attn, key, 256, None, regions, "random_oracle", 7)
    rnd_again = select_binding(V, DS, attn, key, 256, None, regions, "random_oracle", 7)
    assert torch.equal(rnd[1], rnd_again[1])
    assert rnd[3]["label_counts"] == oracle[3]["label_counts"]
    spatial = select_binding(V, DS, attn, key, 256, None, regions, "spatial_oracle", 7)
    assert len(spatial[3]["label_counts"]) == 16 and set(spatial[3]["label_counts"].values()) == {64}
    assert spatial[3]["cross_label_assignees"] == 0
    excessive = torch.arange(1024)
    try:
        select_binding(V, DS, attn, key, 256, None, excessive, "field_oracle", 7)
    except AnchorCoverageError as error:
        assert error.diag["reason"] == "insufficient_contextual_budget_for_labels"
    else:
        raise AssertionError("infeasible region coverage must not discard tokens")
    json.dumps({k: v for k, v in oracle[3].items() if k != "tensors"})
    return dict(all_passed=True, device="cpu", N=1024, K=256,
                historical_bitwise_main_and_three_DS=True,
                historical_split=[238, 18], corrected_split=[205, 51],
                corrected_all_streams_same_rule=True,
                coverage_replacements=oracle[3]["replacement_count"],
                shared_control_oracle_anchors=True, silent_dropped_tokens=0,
                same_label_assignments_only=True, singleton_clusters_explicit=True,
                random_label_histogram_exact=True, random_seed_reproducible=True,
                equal_area_spatial_cells=16, infeasible_coverage_rejected=True,
                dominant_contextual_mass_accounted=205 + sum(oracle[3]["cluster_sizes"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    result = run_cpu_gates()
    print(json.dumps(result, indent=2))
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
