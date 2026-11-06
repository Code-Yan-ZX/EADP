"""Three-task exploratory quality and first-token latency, different cohorts."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path(__file__).resolve().parent.parent
D = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


quality_path = B / "e4_analysis/score_panel003/analysis.json"
quality_review = B / "e1_independent_review/e4score003_review.json"
timing_path = B / "e4_c_multibudget_analysis192_exec/analysis001.json"
timing_review = B / "e1_independent_review/e4c_multibudget_analysis192_review.json"
q, qr, t, tr = map(read, (quality_path, quality_review, timing_path, timing_review))
if qr.get("status") != "passed" or qr.get("analysis_sha256") != sha(quality_path): raise ValueError("quality acceptance")
if tr.get("status") != "passed" or tr.get("analysis_sha256") != sha(timing_path): raise ValueError("timing acceptance")
fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.5), sharey=False)
colors = ["#3479a5", "#208979", "#c3882f", "#a74743"]
for ax, (task, label, n) in zip(axes, [("DocVQA_VAL", "DocVQA", 64), ("TextVQA_VAL", "TextVQA", 64), ("AI2D_TEST", "AI2D", 64)]):
    for k, color in zip((128, 256, 512, 1024), colors):
        stat = t["summary"]["tasks"][task]["natural"][str(k)]["ttft_ms"]["time"]
        x, (xl, xh) = stat["median"], stat["bootstrap_ci95"]
        scores = q["tasks"][task]; arm = f"K{k}-D1"
        y = scores["arm_means_pct"][arm]; yl, yh = scores["arm_percentile_95_ci_pct"][arm]
        ax.errorbar(x, y, xerr=[[x-xl], [xh-x]], yerr=[[y-yl], [yh-y]], color=color, marker="o", capsize=3, lw=1.2, markersize=5)
        ax.annotate(f"K{k}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8, color=color)
    ax.set_title(f"{label}\nquality n=128; timing n={n}", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Natural first-token latency (ms)")
    ax.grid(alpha=.18); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("DEV score (0–100)")
fig.suptitle("Exploratory quality–latency view: EADP, DeepStack on", x=.07, y=.97, ha="left", fontweight="bold", fontsize=14)
fig.text(.07, .01, "Different, partially overlapping samples: quality uses 128 DEV questions/task; TTFT uses a posthoc 64-image cohort, five runs per image.\nBars are separate unadjusted 95% image-group bootstrap intervals; they are not a joint confidence region. Background interference is partly unknown.", fontsize=8)
fig.tight_layout(rect=(.02, .12, 1, .91))
for suffix in ("png", "svg"): fig.savefig(D / f"quality_vs_ttft.{suffix}", dpi=180, bbox_inches="tight")
metadata = {"quality_analysis_sha256": sha(quality_path), "quality_review_sha256": sha(quality_review), "timing_analysis_sha256": sha(timing_path), "timing_review_sha256": sha(timing_review), "script_sha256": sha(__file__), "png_sha256": sha(D / "quality_vs_ttft.png"), "svg_sha256": sha(D / "quality_vs_ttft.svg"), "limitations": ["quality and timing sample sizes differ", "TTFT is an exploratory posthoc 64-image cohort", "intervals are separate and unadjusted", "unknown residual background activity"]}
(D / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
