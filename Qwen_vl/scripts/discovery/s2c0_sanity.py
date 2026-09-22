"""
S2-C0 step 0: correctness sanity for the hand-computed last-query attention.

The proxy scores come from a forward pre-hook that re-derives q/k/v with the
layer's own projections, norms and rotary embedding. This script checks that
derivation against the *official* code path on a real case: one layer's
``config._attn_implementation`` is temporarily switched to ``eager`` and a
forward hook captures the attention weights the official
``eager_attention_forward`` returns. The last-query row over the 1024 visual
keys is then compared element-wise and rank-wise.

If the hand-rolled row is not the official row, every proxy score in S2-C0 is
meaningless, so this runs before anything else.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import eadp_model_name
from s1_audit import OUT
from s2b_gradient_scores import frozen_bank
from s2c0_forward_proxy import LastQueryAttention, build_instance, load_instance_tensors

CASES = [("TextVQA_VAL", 105), ("DocVQA_VAL", 241)]
LAYERS = [1, 2, 4, 8]


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def main():
    import copy

    bank = frozen_bank(["TextVQA_VAL", "DocVQA_VAL"])
    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=32)
    model.model.eval()
    torch.set_grad_enabled(False)
    image_token_id = model.model.config.image_token_id
    lm = model.model.model.language_model
    layers = lm.layers

    report = {}
    for ds, idx in CASES:
        b = bank[ds]
        model.set_dump_image(b["dataset"].dump_image)
        row = b["dataset"].data.iloc[int(idx)]
        _, inputs = build_instance(model, b["dataset"], ds, row)
        prompt, mask, s, e = load_instance_tensors(model, inputs, image_token_id)

        for L in LAYERS:
            attn_mod = layers[L].self_attn
            original_cfg = attn_mod.config
            attn_mod.config = copy.copy(original_cfg)
            attn_mod.config._attn_implementation = "eager"

            captured = {}

            def grab(module, args, output, _c=captured):
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    _c["w"] = output[1].detach()

            h = attn_mod.register_forward_hook(grab)
            probe = LastQueryAttention(L)
            probe.visual_slice = (s, e)
            probe.attach(layers[L])
            with torch.no_grad():
                model.model.model(inputs_embeds=prompt, attention_mask=mask,
                                  use_cache=False, return_dict=True)
            h.remove()
            probe.detach()
            attn_mod.config = original_cfg

            if "w" not in captured:
                raise RuntimeError(f"eager path returned no attention weights (L={L})")
            ref = captured["w"][0].float().cpu().numpy()      # [H, S, S]
            ref_row = ref[:, -1, :]                           # [H, S] official last-query row
            ref_vis = ref_row[:, s:e]
            mine = probe.attn.float().cpu().numpy()           # [H, n_vis]
            mirror = probe.attn_bf16.float().cpu().numpy()    # same row, bf16 matmul

            # sanity: every head's official row sums to 1 over the visible prefix
            row_sums = ref_row.sum(-1)
            report[f"{ds}_{idx}_L{L}"] = dict(
                n_heads=int(ref.shape[0]), seq_len=int(ref.shape[1]), n_vis=int(e - s),
                dtype=str(captured["w"].dtype),
                max_abs_diff=float(np.abs(ref_vis - mine).max()),
                mean_abs_diff=float(np.abs(ref_vis - mine).mean()),
                max_rel_diff=float((np.abs(ref_vis - mine) /
                                    np.clip(np.abs(ref_vis), 1e-9, None)).max()),
                spearman_per_head_mean=float(np.mean(
                    [spearman(ref_vis[h], mine[h]) for h in range(ref_vis.shape[0])])),
                spearman_per_head_min=float(np.min(
                    [spearman(ref_vis[h], mine[h]) for h in range(ref_vis.shape[0])])),
                # bf16 mirror: if this matches the official row tightly, the fp32
                # residual above is dtype rounding, not a derivation error
                mirror_max_abs_diff=float(np.abs(ref_vis - mirror).max()),
                mirror_mean_abs_diff=float(np.abs(ref_vis - mirror).mean()),
                official_row_sum_min=float(row_sums.min()),
                official_row_sum_max=float(row_sums.max()),
                vis_mass_official=float(ref_vis.sum(-1).mean()),
                vis_mass_mine=float(mine.sum(-1).mean()),
            )
            r = report[f"{ds}_{idx}_L{L}"]
            print(f"{ds}_{idx} L={L:2d}  fp32: maxabs={r['max_abs_diff']:.3e} "
                  f"spearman(h-mean)={r['spearman_per_head_mean']:.6f} "
                  f"min={r['spearman_per_head_min']:.6f} | "
                  f"bf16 mirror: maxabs={r['mirror_max_abs_diff']:.3e} "
                  f"meanabs={r['mirror_mean_abs_diff']:.3e} | "
                  f"vis-mass ref={r['vis_mass_official']:.4f} mine={r['vis_mass_mine']:.4f}")
            del ref, ref_row, ref_vis, mine, mirror
            torch.cuda.empty_cache()
        del prompt, mask, inputs
        torch.cuda.empty_cache()

    worst = max(v["max_abs_diff"] for v in report.values())
    worst_rank = min(v["spearman_per_head_mean"] for v in report.values())
    worst_mirror = max(v["mirror_max_abs_diff"] for v in report.values())
    # The official row is returned in bf16, so a normalised row sums to 1 only to
    # ~2e-3. A masking error would show up as a *systematic* deficit (the causal
    # mask would hide real keys), i.e. sums well below 1, not as this jitter.
    row_sum_ok = all(0.995 < v["official_row_sum_min"] and v["official_row_sum_max"] < 1.005
                     for v in report.values())
    ok = worst < 5e-3 and worst_rank > 0.9999 and worst_mirror < 1e-4 and row_sum_ok
    json.dump({"cases": report, "max_abs_diff_worst": worst,
               "spearman_head_mean_worst": worst_rank,
               "mirror_max_abs_diff_worst": worst_mirror,
               "official_row_sums_normalised": bool(row_sum_ok), "ok": bool(ok),
               "criterion": "against the official eager_attention_forward last-query row: "
                            "per-head Spearman > 0.9999, max abs diff < 5e-3, and the "
                            "bf16 mirror of the same derivation < 1e-4 (so the fp32 "
                            "residual is dtype rounding, not a logic error)"},
              open(os.path.join(OUT, "s2c0_sanity.json"), "w"), indent=1)
    print(f"\n[sanity] worst fp32 max-abs-diff {worst:.3e}, worst head-mean Spearman "
          f"{worst_rank:.8f}, worst bf16-mirror max-abs-diff {worst_mirror:.3e}, "
          f"row-sums normalised {row_sum_ok} -> {'OK' if ok else 'FAIL'}")
    print(f"[saved] {os.path.join(OUT, 's2c0_sanity.json')}")


if __name__ == "__main__":
    main()
