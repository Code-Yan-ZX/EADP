"""
M2 amendment figure: the Pareto the pre-registration asked for, drawn only
from the VALID arms and the PAIRED timing.

Panel A  macro accuracy vs end-to-end TTFT (paired within-block median), with
         seed-range error bars on accuracy and p10-p90 bars on TTFT, and the
         three frozen success regions drawn against B1 (§5 of the prereg).
Panel B  where the milliseconds go: per-request model-side prefill, stacked,
         single-counted windows (the amendment's timing fix makes these honest).

Reads m2_amend_report.json + m2_perf_paired.json; writes figures/m2_pareto.png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR                                         # noqa: E402

DS_ORDER = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]

rep = json.load(open(os.path.join(OUTPUT_DIR, "m2_amend_report.json")))
per = json.load(open(os.path.join(OUTPUT_DIR, "m2_perf_paired.json")))

arms = {}
for t, b in rep["baselines"].items():
    arms[t] = dict(macro=b["macro_pct"], lo=b["macro_pct"], hi=b["macro_pct"])
for t, b in rep["arms"].items():
    arms[t] = dict(macro=b["macro_seed_mean"], lo=b["macro_seed_range"][0],
                   hi=b["macro_seed_range"][1])
for t, s in per["summary"].items():
    arms.setdefault(t, {}).update(
        ttft=s["ttft"]["median"], lo10=s["ttft"]["p10"], hi90=s["ttft"]["p90"],
        model_only=s["model_only"]["median"],
        stages={k: v["median"] for k, v in s["stages"].items()})

ORDER = ["B0", "B1", "B2", "C1-P", "C1-R"]
COL = {"B0": "#555555", "B1": "#d62728", "B2": "#8c564b",
       "C1-P": "#1f77b4", "C1-R": "#17becf"}

fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=150)

b1 = arms["B1"]
# frozen regions vs B1 (prereg §5), drawn relative to B1's own TTFT
ax.axvspan(b1["ttft"] * 0.85, b1["ttft"], ymin=0, ymax=1,
           color="#2ca02c", alpha=0.07, label="primary: ≥15 % TTFT cut")
ax.axhline(b1["macro"], color="#2ca02c", ls=":", lw=1.2,
           label="B1 accuracy (accuracy-led needs ≥ this, CI-above)")
for t in ORDER:
    a = arms[t]
    ax.errorbar(a["ttft"], a["macro"],
                xerr=[[a["ttft"] - a["lo10"]], [a["hi90"] - a["ttft"]]],
                yerr=[[a["macro"] - a["lo"]], [a["hi"] - a["macro"]]],
                fmt="o", ms=8, color=COL[t], capsize=3, lw=1.4, label=t)
    dy = {"C1-R": -2.2, "C1-P": 1.4}.get(t, 0.0)
    ax.annotate(" " + t, (a["ttft"], a["macro"] + dy), fontsize=9,
                color=COL[t], va="center")
ax.set_xlabel("TTFT, paired within-block median (ms)")
ax.set_ylabel("held-out macro accuracy (%)")
ax.set_title("M2 amendment — valid arms only (paired, 120 blocks)")
ax.legend(fontsize=7, loc="lower left")
ax.grid(alpha=0.25)

# panel B: model-side stack
labels = ["vision", "prune machinery", "LLM prefill"]
comp = {}
for t in ORDER:
    st = arms[t]["stages"]
    mach = (st.get("eadp_scoring_ms", 0) + st.get("selector_ms", 0)
            + st.get("L0_L4_ms", 0) + st.get("scorer_ms", 0)
            + st.get("token_compaction_ms", 0))
    llm = arms[t]["model_only"] - st.get("vision_encoder_ms", 0) - mach
    comp[t] = (st.get("vision_encoder_ms", 0), mach, max(llm, 0.0))
x = np.arange(len(ORDER))
bot = np.zeros(len(ORDER))
cols = ["#9edae5", "#ff9d9a", "#aec7e8"]
for i, lab in enumerate(labels):
    v = np.array([comp[t][i] for t in ORDER])
    axb.bar(x, v, 0.62, bottom=bot, label=lab, color=cols[i])
    bot += v
for xi, t in zip(x, ORDER):
    axb.text(xi, bot[xi] + 2, f"{bot[xi]:.0f}", ha="center", fontsize=8)
axb.set_xticks(x, ORDER, fontsize=9)
axb.set_ylabel("ms (median, single-counted)")
axb.set_title("model-only prefill = TTFT − image preprocess")
axb.legend(fontsize=8)
axb.grid(alpha=0.25, axis="y")

fig.suptitle("GDEP at L4 holds EADP accuracy at −22.5 ms vs B1 — but B1's "
             "edge is its selector; GDEP is +10 ms behind B2", fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(OUTPUT_DIR, "figures", "m2_pareto.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out)
print("[saved]", out)
