"""
M4-v0 step 4 -- figures.

Four figures, all from stored artefacts plus one vision-tower forward per case
(no generation):

  m4_mechanism.png   one instance, end to end: the image, B2's anchors and the
                     evicted set, the residual/unexplainedness map u, and the r
                     spatial cells with the effective sample size each capsule
                     actually pooled.
  m4_profile.png     the arm table as a per-benchmark profile, with the paired
                     CI on the macro delta beside each arm.
  m4_cases.png       B2 wrong -> REC right, and REC breaks.
  m4_diagnostics.png capsule norm ratio and weight entropy vs r, from the
                     offline pass.

The residual maps are recomputed here through the SAME live functions the pruner
calls (`m4_common.evict_t` / `residual_u` / `build_capsules`), so a figure can
never show a mechanism the arm does not run.

Usage
    python scripts/discovery/m4_figures.py --tag m4_accuracy
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                        # noqa: E402
from common import OUTPUT_DIR, eadp_model_name                       # noqa: E402
from m2_accuracy import MAX_NEW, heldout                             # noqa: E402
from m2_gdep import MODE_PRELLM, GDEPConfig, GDEPEngine              # noqa: E402
from m4_common import (BUDGET, EVICT_RULE, GRID, N_VIS, R_GRID, TAU,  # noqa: E402
                       build_capsules, evict_t, partition_ids, region_factor,
                       residual_u)
from instrumented import attach_pruner                               # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.colors import ListedColormap                         # noqa: E402
from PIL import Image                                                # noqa: E402

FIGS = os.path.join(OUTPUT_DIR, "figures")
DS_ORDER = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]


def load(tag):
    with open(os.path.join(OUTPUT_DIR, f"{tag}.json")) as f:
        return json.load(f)


# ===========================================================================
# one vision-tower forward per case: the same tensors the live pruner sees
# ===========================================================================
def make_fig_engine(model):
    """One engine for every figure, built exactly as m4_accuracy builds B2's."""
    cfg = GDEPConfig(mode=MODE_PRELLM, budget=BUDGET, selector="block8", tag="M4FIG")
    eng = GDEPEngine.from_checkpoint(cfg, model=model, max_new_tokens=MAX_NEW)
    pruner = attach_pruner(model, selector="block8", capture=False)
    pruner.visual_token_num = BUDGET
    pruner.sim_mode = "rebound"
    pruner.keep_gpu = True
    eng.pruner = pruner
    model.pruner = pruner
    torch.set_grad_enabled(False)
    return eng


def case_state(eng, item, r=16):
    """The tensors the live pruner sees for one instance -- no generation."""
    pruner = eng.pruner
    pruner.last_gpu = {}
    prep = eng.prepare(item["msg"], item["ds"])
    text_llm, text_seq = eng._instruction_embeds(prep)
    pruner(prep["vis"], text_llm, text_seq, prep["gthw"])
    g = pruner.last_gpu
    pruner.last_gpu = {}
    del prep
    vis = g["image_features"].float()
    imp = g["importance"].reshape(-1).float()
    sim = g["sim_matrix"].reshape(N_VIS, N_VIS).float()
    s0 = torch.sort(g["select_idx"][0].to(torch.long)).values
    ev = evict_t(s0, imp, sim, r, rule=EVICT_RULE)
    A = s0[~torch.isin(s0, ev)]
    keep = torch.zeros(N_VIS, dtype=torch.bool)
    keep[A] = True
    dropped = (~keep).nonzero(as_tuple=True)[0]
    u = residual_u(vis, A)
    caps, st = build_capsules(vis, imp, u, A, dropped, r, "residual", "spatial", TAU)
    return dict(vis=vis.cpu(), imp=imp.cpu(), s0=s0.cpu().numpy(),
                ev=ev.cpu().numpy(), A=A.cpu().numpy(), u=u.cpu().numpy(),
                caps=caps.cpu(), stats=st,
                ess_by_cell=_ess_by_cell(u, A, dropped, r))


def _ess_by_cell(u, A, dropped, r):
    rid = partition_ids(r)
    out = {}
    for g in range(r):
        idx = dropped[rid[dropped] == g]
        if idx.numel() == 0:
            out[g] = 0.0
            continue
        w = torch.softmax(u[idx] / TAU, dim=0)
        out[g] = float(1.0 / (w ** 2).sum())
    return out


def question_of(msg):
    """The instruction text out of a VLMEvalKit message (a list of typed parts)."""
    if isinstance(msg, str):
        return msg.replace("\n", " ")
    if isinstance(msg, list):
        parts = [str(p.get("value", "")) for p in msg
                 if isinstance(p, dict) and p.get("type") == "text"]
        return " ".join(parts).replace("\n", " ")
    return str(msg)[:120]


def load_image(item):
    dataset = item["dataset"]
    path = dataset.dump_image(item["row"])
    if isinstance(path, list):
        path = path[0]
    return Image.open(path).convert("RGB")


def map_panel(ax, vec, title, cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(np.asarray(vec).reshape(GRID, GRID), cmap=cmap,
                   interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def region_overlay(ax, r, ess=None):
    """The r spatial cells, drawn with their boundaries and ESS."""
    rows, cols = region_factor(r)
    m = partition_ids(r).reshape(GRID, GRID).numpy()
    ax.imshow(m, cmap="tab20", interpolation="nearest", alpha=0.85)
    for a in range(1, rows):
        ax.axhline(a * GRID / rows - 0.5, color="w", lw=1.2)
    for b in range(1, cols):
        ax.axvline(b * GRID / cols - 0.5, color="w", lw=1.2)
    if ess:
        for a in range(rows):
            for b in range(cols):
                g = a * cols + b
                ax.text(b * GRID / cols + GRID / cols / 2 - 0.5,
                        a * GRID / rows + GRID / rows / 2 - 0.5,
                        f"{ess[g]:.0f}", ha="center", va="center", fontsize=6,
                        color="k")
    ax.set_title(f"r={r} spatial cells (label = capsule ESS)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def fig_mechanism(eng, item, out, r=16, tag=""):
    st = case_state(eng, item, r)
    fig = plt.figure(figsize=(15, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(load_image(item))
    ax.set_title(f"{item['key']}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    kind = np.zeros(N_VIS)
    kind[st["s0"]] = 1.0
    kind[st["ev"]] = 2.0
    ax.imshow(kind.reshape(GRID, GRID), cmap=ListedColormap(
        ["#e8e8e8", "#1f77b4", "#d62728"]), interpolation="nearest")
    ax.set_title(f"B2 anchors (blue {BUDGET}) / evicted (red {r})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    im = map_panel(ax, st["u"], "residual u = 1 - max$_{a\\in A}$ cos(v, v$_a$)",
                   "magma", 0.0, float(np.percentile(st["u"], 99)))
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 3])
    region_overlay(ax, r, st["ess_by_cell"])

    ax = fig.add_subplot(gs[1, :2])
    ev = st["ev"]
    uu = st["u"]
    ax.hist(uu[~np.isin(np.arange(N_VIS), st["A"])], bins=60, color="#888",
            label="dropped (768 + r)", density=True)
    ax.hist(uu[ev], bins=60, color="#d62728", alpha=0.7, density=True,
            label="evicted from S0")
    ax.axvline(float(np.median(uu[~np.isin(np.arange(N_VIS), st["A"])])),
               color="k", ls="--", lw=1)
    ax.set_xlabel("u (unexplainedness)"); ax.set_ylabel("density")
    ax.legend(fontsize=8); ax.set_title("residual weight distribution", fontsize=9)

    ax = fig.add_subplot(gs[1, 2:])
    tok = st["vis"][~np.isin(np.arange(N_VIS), st["A"])]
    ax.hist(tok.norm(dim=1).numpy(), bins=60, color="#1f77b4", alpha=0.6,
            density=True, label="pooled dropped tokens")
    ax.hist(st["caps"].norm(dim=1).numpy(), bins=30, color="#ff7f0e",
            density=True, label="r capsules")
    ax.axvline(float(tok.norm(dim=1).mean()), color="k", ls="--", lw=1,
               label="pooled mean norm")
    ax.set_xlabel("||feature||"); ax.legend(fontsize=8)
    ax.set_title("why the norm-restore arm exists: a convex combination "
                 "shrinks", fontsize=9)

    fig.suptitle(f"REC mechanism -- {item['key']}   "
                 f"(u p50 {np.median(uu):.3f}, capsule/pooled norm "
                 f"{float(st['caps'].norm(dim=1).median()/tok.norm(dim=1).mean()):.3f})",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(FIGS, f"m4_mechanism{tag}.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  -> {path}")
    return path


def fig_profile(rows, gate, out):
    arms = [k for k in ["B2", "EVICT-r8", "EVICT-r16", "EVICT-r32", "MEAN-r16",
                        "IMP-r16", "SHUF-r16", "ANCH-r16", "NORM-r16", "FPS-r16",
                        "REC-r8", "REC-r16", "REC-r32", "B1", "B0"] if k in rows
            and "hits" in rows[k]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4),
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    y = np.arange(len(arms))
    for i, k in enumerate(arms):
        r = rows[k]
        d = r.get("vs_B2")
        lo, hi = (d["ci"] if d else (0, 0))
        color = ("#1f77b4" if k == "REC-r16" else
                 "#d62728" if r.get("kind") == "diagnostic" else
                 "#2ca02c" if r.get("kind") == "baseline" else "#999")
        ax.barh(i, r["macro"], color=color, alpha=0.85)
        if d:
            ax.plot([r["macro"] - (d["delta"] - lo), r["macro"] + (hi - d["delta"])],
                    [i, i], color="k", lw=1.2)
    ax.axvline(rows["B2"]["macro"], color="k", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(arms, fontsize=8)
    ax.set_xlabel("macro (paired CI on the delta drawn)")
    ax.set_title("held-out 150, all arms", fontsize=10)
    ax.invert_yaxis()

    ax = axes[1]
    w = 0.26
    x = np.arange(len(DS_ORDER))
    for j, k in enumerate(["B2", "MEAN-r16", "REC-r16"]):
        if k not in rows or "hits" not in rows[k]:
            continue
        vals = [rows[k]["per_ds"][ds] - rows["B2"]["per_ds"][ds] for ds in DS_ORDER]
        ax.bar(x + (j - 1) * w, vals, w, label=k,
               color=["#999", "#ff7f0e", "#1f77b4"][j])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([d.replace("_VAL", "") for d in DS_ORDER])
    ax.set_ylabel("points vs B2"); ax.legend(fontsize=8)
    ax.set_title("benchmark profile (OCRBench is the one to watch)", fontsize=10)
    fig.suptitle(f"REC-v0 -- verdict {gate['verdict']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(FIGS, "m4_profile.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  -> {path}")
    return path


def fig_diagnostics(off):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    rgrid = [r for r in R_GRID]
    ax = axes[0]
    ax.plot(rgrid, [off["norm_ratio"][str(r)]["median"] for r in rgrid], "o-")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("r"); ax.set_ylabel("||capsule|| / ||pooled token||")
    ax.set_title("capsule shrinkage", fontsize=10)
    ax = axes[1]
    for t in off["tau_grid"]:
        ax.plot(rgrid, [off["ess"][str(r)][str(t)] for r in rgrid], "o-",
                label=f"tau={t}")
    ax.plot(rgrid, [N_VIS / r for r in rgrid], "k:", label="uniform (all pooled)")
    ax.set_yscale("log"); ax.set_xlabel("r"); ax.set_ylabel("effective sample size")
    ax.legend(fontsize=7); ax.set_title("weight concentration", fontsize=10)
    ax = axes[2]
    P = off["u_dropped_pct"]
    ax.bar(["p10", "p50", "p90"], [P["p10"], P["p50"], P["p90"]], color="#1f77b4")
    ax.set_ylabel("u"); ax.set_title("residual spread on dropped tokens", fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIGS, "m4_diagnostics.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  -> {path}")
    return path


def fig_cases(eng, items, rows, out, n_fix=2, n_break=1):
    """B2 wrong -> REC right, and REC broken.  Same panels as the mechanism."""
    b2, rec = rows["B2"], rows["REC-r16"]
    # `per_benchmark[ds]["hits"]` is ordered by the instances of that benchmark
    # in `items` order, so a per-benchmark running counter is what indexes it.
    seen = {ds: 0 for ds in DS_ORDER}
    fixed, broken = [], []
    for j, it in enumerate(items):
        ds = it["ds"]
        i = seen[ds]
        seen[ds] += 1
        b = float(b2["hits"][ds][i])
        a = float(rec["hits"][ds][i])
        if b == 0 and a > 0:
            fixed.append((j, it, b, a))
        elif b > 0 and a == 0:
            broken.append((j, it, b, a))
    print(f"  [cases] B2 wrong -> REC right: {len(fixed)};  REC breaks: {len(broken)}")
    picks = [("fixed", x) for x in fixed[:n_fix]] + [("broken", x) for x in broken[:n_break]]
    if not picks:
        print("  [cases] nothing to draw")
        return None
    fig, axes = plt.subplots(len(picks), 4, figsize=(15, 3.7 * len(picks)),
                             squeeze=False)
    for row, (kind, (j, it, b, a)) in enumerate(picks):
        st = case_state(eng, it, 16)
        pred_b2 = b2["predictions"][j]
        pred_rec = rec["predictions"][j]
        ax = axes[row][0]
        ax.imshow(load_image(it))
        q = question_of(it["msg"])
        ax.set_title(f"{kind.upper()}  {it['key']}\nQ: {q[:78]}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax = axes[row][1]
        map_panel(ax, st["u"], "residual u", "magma", 0.0,
                  float(np.percentile(st["u"], 99)))
        ax = axes[row][2]
        region_overlay(ax, 16, st["ess_by_cell"])
        ax = axes[row][3]
        ax.axis("off")
        ax.text(0.0, 0.95, f"B2  : {pred_b2[:120]}", fontsize=8, va="top",
                wrap=True)
        ax.text(0.0, 0.55, f"REC : {pred_rec[:120]}", fontsize=8, va="top",
                wrap=True)
        ax.text(0.0, 0.12, f"hit  B2={b:.2f}  REC={a:.2f}", fontsize=8, va="top")
    fig.tight_layout()
    path = os.path.join(FIGS, "m4_cases.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m4_accuracy")
    ap.add_argument("--which", default="mechanism,profile,diag,cases")
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    want = set(args.which.split(","))

    m4 = load(args.tag)
    m2 = load("m2_accuracy")
    from m4_analyze import add_contrasts, rows_of, gate as gate_of
    rows = rows_of(m4, m2)
    add_contrasts(rows)
    g = gate_of(rows)

    if "diag" in want:
        fig_diagnostics(load("m4_offline"))
    if "profile" in want:
        fig_profile(rows, g, None)
    if "mechanism" in want or "cases" in want:
        model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                                  max_new_tokens=MAX_NEW)
        model.model.eval()
        items = heldout(model)
        eng = make_fig_engine(model)
        if "mechanism" in want:
            for ds in DS_ORDER:
                it = next(x for x in items if x["ds"] == ds)
                fig_mechanism(eng, it, None, 16, tag="_" + ds.replace("_VAL", ""))
        if "cases" in want:
            fig_cases(eng, items, rows, None)
        del eng
        torch.cuda.empty_cache()
    print("[figures done]")


if __name__ == "__main__":
    main()
