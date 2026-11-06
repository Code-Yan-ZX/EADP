"""Pure, fixed batch-level geometry and random budget assignments."""
import hashlib
import math

TASKS = ("DocVQA_VAL", "TextVQA_VAL", "AI2D_TEST")


def digest(text): return hashlib.sha256(text.encode("utf8")).hexdigest()
def key(r): return r["task"], str(r["qid"]), r["group_id"]


def analyze(features, score_rows):
    f = {key(r): r for r in features}; scores = {key(r): r for r in score_rows}
    if len(features) != 384 or len(f) != 384 or len(score_rows) != 384 or len(scores) != 384 or set(f) != set(scores): raise ValueError("incomplete paired inventory")
    report, assignments = {}, []
    for task in TASKS:
        rows = [r for r in features if r["task"] == task]
        if len(rows) != 128 or len({r["group_id"] for r in rows}) != 128: raise ValueError("task denominator")
        for r in rows:
            if any(not isinstance(r.get(k), (float, int)) or not math.isfinite(r[k]) for k in ("nearest_mean", "occupied8", "entropy4")): raise ValueError("invalid feature")
            for budget in (128, 256, 512):
                s = scores[key(r)]["arms"][f"K{budget}-D1"]
                if not 0 <= s["score"] <= 1 or not math.isfinite(s["independent_score"]) or abs(s["score"] - s["independent_score"]) > 1e-12: raise ValueError("invalid accepted score")
        tie = lambda r: digest("20261105|" + r["group_id"])
        ordered = {"nearest_low": sorted(rows, key=lambda r: (r["nearest_mean"], tie(r))),
                   "occupied8_high": sorted(rows, key=lambda r: (-r["occupied8"], tie(r))),
                   "entropy4_high": sorted(rows, key=lambda r: (-r["entropy4"], tie(r)))}
        for seed in range(3): ordered[f"random_{seed}"] = sorted(rows, key=lambda r: digest(f"20261105|random|{seed}|" + r["group_id"]))
        baseline = sum(scores[key(r)]["arms"]["K256-D1"]["score"] for r in rows) / 128
        task_report = {}
        for name, rank in ordered.items():
            high = {r["group_id"] for r in rank[:42]}; values = []
            if len(high) != 42: raise ValueError("high-budget denominator")
            for r in rows:
                arms = scores[key(r)]["arms"]; k = 512 if r["group_id"] in high else 128
                v, ref = arms[f"K{k}-D1"]["score"], arms["K256-D1"]["score"]
                values.append((v, ref)); assignments.append({"task": task, "qid": str(r["qid"]), "group_id": r["group_id"], "rule": name, "selected_k": k, "score": v, "baseline_score": ref})
            task_report[name] = {"mean_pct": 100 * sum(v for v, _ in values) / 128, "minus_fixed256_pp": 100 * (sum(v for v, _ in values) / 128 - baseline),
                                 "improved": sum(v > ref for v, ref in values), "declined": sum(v < ref for v, ref in values), "unchanged": sum(v == ref for v, ref in values), "high_k_count": 42, "mean_k": 254}
        report[task] = {"n": 128, "fixed_k256_mean_pct": 100 * baseline, "rules": task_report}
    return report, assignments
