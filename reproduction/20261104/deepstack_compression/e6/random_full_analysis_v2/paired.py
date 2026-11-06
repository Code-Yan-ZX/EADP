"""Pure, question-weighted image-group bootstrap for the fixed three seeds."""
from collections import defaultdict

import numpy as np

SEEDS = (20261104, 20261105, 20261106)
ARMS = ("eadp", "random_20261104", "random_20261105", "random_20261106", "random_three_seed_mean")


def analyze(rows, replicates=10000, seed=20261104):
    if not rows or type(replicates) is not int or replicates < 1:
        raise ValueError("empty rows or invalid bootstrap count")
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if not isinstance(row.get("group_id"), str) or not row["group_id"]:
            raise ValueError("invalid group")
        if not isinstance(row.get("qid"), str) or not row["qid"] or row["qid"] in seen:
            raise ValueError("invalid or duplicate question")
        seen.add(row["qid"])
        scores = row.get("scores")
        if not isinstance(scores, dict) or set(scores) != {"eadp", *SEEDS}:
            raise ValueError("incomplete seed scores")
        values = [float(scores["eadp"])] + [float(scores[s]) for s in SEEDS]
        if any(not np.isfinite(x) or not 0 <= x <= 1 for x in values):
            raise ValueError("invalid score")
        values.append(sum(values[1:]) / len(SEEDS))
        groups[row["group_id"]].append(values)

    ids = sorted(groups)
    sums = np.asarray([np.asarray(groups[g]).sum(axis=0) for g in ids])
    sizes = np.asarray([len(groups[g]) for g in ids])
    values = 100 * sums.sum(axis=0) / sizes.sum()
    draws = np.random.default_rng(seed).integers(0, len(ids), (replicates, len(ids)))
    boot = 100 * sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None]
    interval = lambda x: [float(v) for v in np.quantile(x, [0.025, 0.975])]
    return {
        "questions": len(rows),
        "image_groups": len(ids),
        "seed_values": list(SEEDS),
        "arm_mean_pct": {a: float(values[i]) for i, a in enumerate(ARMS)},
        "arm_ci95_pct": {a: interval(boot[:, i]) for i, a in enumerate(ARMS)},
        "contrasts_vs_eadp": {
            a: {"difference_pp": float(values[i] - values[0]), "ci95_pp": interval(boot[:, i] - boot[:, 0])}
            for i, a in enumerate(ARMS) if i
        },
        "bootstrap": {"unit": "image_group", "group_order": "ascending", "weighting": "question", "replicates": replicates, "seed": seed, "numpy_version": np.__version__},
    }
