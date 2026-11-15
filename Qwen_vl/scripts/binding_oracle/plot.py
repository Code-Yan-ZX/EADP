"""Plot the binding-oracle analysis without treating unfinished arms as zero.

The two panels show descriptive rates on available data. The separate paired
effect annotations use only common complete pages, as supplied by analyze.py.
This is a CPU-only plotting entry point; it does not import a model or torch.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT = Path(__file__).resolve().parents[2] / "outputs/binding_oracle"
PRIMARY_CONTRAST = "restricted_aggregation_same_anchors"
ORDER = (
    "teacher", "visionzip", "visionzip_ppe", "field_anchor_control_ppe",
    "field_oracle", "field_oracle_ppe", "random_oracle_ppe",
    "spatial_oracle_ppe", "rres512", "historical", "historical_ds_fixed",
    "line_anchor_control_ppe", "line_oracle_ppe",
)
LABELS = {
    "teacher": "Full model (K=1024)",
    "visionzip": "VisionZip",
    "visionzip_ppe": "VisionZip + PPE",
    "field_anchor_control_ppe": "Field anchors, unrestricted + PPE",
    "field_oracle": "Field restriction",
    "field_oracle_ppe": "Field restriction + PPE",
    "random_oracle_ppe": "Random regions + PPE",
    "spatial_oracle_ppe": "Spatial regions + PPE",
    "rres512": "R-res512 (native K=256)",
    "historical": "Historical D238/C18",
    "historical_ds_fixed": "Historical, DS repaired",
    "line_anchor_control_ppe": "OCR-line anchors + PPE",
    "line_oracle_ppe": "OCR-line restriction + PPE",
}
COLORS = {
    "teacher": "#5b6470",
    "field_anchor_control_ppe": "#6b58a3",
    "field_oracle": "#bc672e",
    "field_oracle_ppe": "#bc672e",
    "rres512": "#247b69",
    "historical": "#80868e",
    "historical_ds_fixed": "#80868e",
    "line_anchor_control_ppe": "#9577ac",
    "line_oracle_ppe": "#9577ac",
}


def population_view(data, population):
    if population == "combined":
        # The root stores records as well as summaries; data['combined'] has a
        # more compact and intentionally different arms structure.
        return data
    try:
        return data["strata"][population]
    except KeyError as error:
        available = ", ".join(sorted(data.get("strata", {}))) or "none"
        raise ValueError(f"population {population!r} unavailable; strata: {available}") from error


def paired_annotation(contrast, question=False):
    endpoint = contrast.get("question_endpoint", {}) if question else contrast
    delta = endpoint.get("delta")
    interval = endpoint.get("ci95")
    n_pages = endpoint.get("n_pages", 0)
    n_questions = endpoint.get("n_questions", 0)
    sample = f"{n_pages} matched pages"
    if question:
        sample += f" / {n_questions} questions"
    status = contrast.get("status", "pending")
    if delta is None or not n_pages:
        return f"Same-anchor oracle minus control: pending\n{sample}; contrast {status}."
    effect = f"{100 * delta:+.1f} pp"
    if interval is None:
        effect += "; 95% CI unavailable"
    else:
        effect += f"; 95% CI [{100 * interval[0]:+.1f}, {100 * interval[1]:+.1f}] pp"
    return f"Same-anchor oracle minus control: {effect}\n{sample}; contrast {status}."


def plot_analysis(data, population="combined"):
    view = population_view(data, population)
    arms = view.get("arms", {})
    names = [name for name in ORDER if name in arms]
    names += sorted(set(arms) - set(ORDER))
    if not names:
        raise ValueError("analysis contains no arms to plot")
    screen = view.get("teacher_screen", {})
    planned = screen.get("planned", screen.get("attempted", "?"))
    title_population = "Combined VAL + TRAIN" if population == "combined" else population
    contrast = view.get("contrasts", {}).get(PRIMARY_CONTRAST, {})
    decision = view.get("decision", {}).get("verdict", "pending")
    stage = data.get("snapshot_stage", "unknown")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    height = max(8.0, .42 * len(names) + 3.0)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(17.6, height))
    fig.subplots_adjust(left=.225, right=.985, top=.75, bottom=.31, wspace=.15)
    fig.suptitle(f"VisionZip / PPE field-binding oracle | {title_population}",
                 x=.225, y=.965, ha="left", fontsize=17, weight="bold")
    subtitle = (f"Teacher screen: {screen.get('accepted', 0)}/{planned} pages accepted, "
                f"{screen.get('rejected', 0)} rejected, {screen.get('pending', 0)} pending.  "
                f"Snapshot: {stage}; decision: {decision}.")
    fig.text(.225, .915, subtitle, fontsize=10.5, color="#4c5663", va="top")
    fig.text(.225, .882,
             "All compressed arms use K=256. Rates below are descriptive; partial arms may have different samples.",
             fontsize=10.1, color="#4c5663")

    specifications = (
        ("A", "All four answers correct", "Primary endpoint", "page_accuracy",
         "all_correct", "complete_pages", "expected_pages", "pages"),
        ("B", "Individual answers correct", "Secondary endpoint", "question_accuracy",
         "individual_correct", "individual_questions", "expected_individual_questions", "questions"),
    )
    for ax, spec in zip(axes, specifications):
        panel, title, subtitle, rate_key, count_key, n_key, expected_key, unit = spec
        fig.text(ax.get_position().x0, .837, f"{panel}  {title}\n{subtitle}",
                 va="top", ha="left", fontsize=12.5)
        ax.set_xlim(-2, 150)
        ax.set_ylim(len(names) - .5, -.5)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Exact numeric-value accuracy (%)", labelpad=10)
        ax.tick_params(axis="y", length=0, pad=10, labelsize=10)
        ax.tick_params(axis="x", colors="#4c5663")
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_bounds(0, 100)
        ax.grid(axis="x", color="#dfe3e8", linewidth=.8)
        ax.set_axisbelow(True)
        ax.text(107, -.90, f"Correct / available {unit}\n(planned denominator)",
                fontsize=8.7, color="#4c5663", va="bottom")
        for index, name in enumerate(names):
            summary = arms[name]["summary"]
            n, expected = summary.get(n_key, 0), summary.get(expected_key, 0)
            count, rate = summary.get(count_key, 0), summary.get(rate_key)
            complete = summary.get("status") == "complete"
            color = COLORS.get(name, "#3378a5")
            if n and rate is not None and math.isfinite(float(rate)):
                if not 0 <= float(rate) <= 1:
                    raise ValueError(f"{name}.{rate_key} is outside [0, 1]")
                ax.hlines(index, 0, 100 * rate, color=color, alpha=.20, linewidth=3)
                ax.plot(100 * rate, index, marker="o", markersize=8,
                        markerfacecolor=color if complete else "white",
                        markeredgecolor=color, markeredgewidth=1.8,
                        linestyle="none", zorder=3)
                label = f"{count}/{n}  ({expected})"
            elif expected:
                # A true observed 0% has n>0 and is plotted above. No answer is
                # represented by pending text, never by a zero-valued point.
                label = f"pending  (0/{expected})"
            else:
                label = "not scheduled"
            ax.text(107, index, label, va="center", fontsize=9,
                    color="#374151" if complete else "#66717f")
        ax.set_yticks(range(len(names)), [LABELS.get(name, name) for name in names])
        # Prefix audits and line-sensitive populations have different planned
        # subsets. The rules and denominators remain visible in the same figure.
        for boundary in ("historical", "line_anchor_control_ppe"):
            if boundary in names:
                index = names.index(boundary)
                if index:
                    ax.axhline(index - .5, color="#cdd2d9", linewidth=.8, linestyle=(0, (3, 3)))
        annotation = paired_annotation(contrast, question=panel == "B")
        ax.text(0, -.34, annotation, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.7, linespacing=1.55,
                bbox=dict(boxstyle="round,pad=.5", facecolor="#f4f6f8", edgecolor="#d7dce2"))
    axes[1].tick_params(axis="y", labelleft=False)
    handles = [
        Line2D([], [], marker="o", color="#50667e", linestyle="none", markersize=7,
               markerfacecolor="#50667e", label="Arm complete on planned subset"),
        Line2D([], [], marker="o", color="#50667e", linestyle="none", markersize=7,
               markerfacecolor="white", label="Arm incomplete; available observations only"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.222, .225),
               frameon=False, ncol=2, fontsize=9.5, handletextpad=.5, columnspacing=2)
    note = ("Paired effects use common complete pages and source-document bootstrap 95% intervals; "
            "their samples may differ from descriptive rates. A [0, 0] interval does not establish equivalence. "
            "Historical audits and OCR-line sensitivity use separately declared subsets. "
            "PPE is a native Qwen3 operator adaptation; this is a region-oracle pilot, not official benchmark performance.")
    fig.text(.225, .075, textwrap.fill(note, 153), va="top", fontsize=9,
             color="#4c5663", linespacing=1.55)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", default=str(DEFAULT / "analysis.json"))
    parser.add_argument("--population", choices=("combined", "VAL", "TRAIN"), default="combined")
    parser.add_argument("--out", help="output prefix, or a .png/.pdf path; both formats are saved")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("dpi must be positive")
    source = Path(args.analysis)
    data = json.loads(source.read_text())
    try:
        fig = plot_analysis(data, args.population)
    except ValueError as error:
        parser.error(str(error))
    default_name = "results" if args.population == "combined" else f"results_{args.population}"
    prefix = Path(args.out) if args.out else source.parent / default_name
    if prefix.suffix.lower() in (".png", ".pdf"):
        prefix = prefix.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(str(prefix) + suffix) for suffix in (".png", ".pdf")]
    for path in paths:
        fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    print(json.dumps(dict(population=args.population, analysis=str(source.resolve()),
                          outputs=[str(path.resolve()) for path in paths]), indent=2))


if __name__ == "__main__":
    main()
