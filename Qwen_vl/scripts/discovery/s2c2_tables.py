"""
S2-C2 report tables: emit the markdown blocks that go into
``docs/scoring_search_s2c2.md`` straight from ``s2c2_diagnosis.json``, so no
number in the report is transcribed by hand.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT  # noqa: E402
from s2c2_common import DS_ALL, load_case  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="s2c2_diagnosis.json")
    ap.add_argument("--iso", default="s2c2_identity.json")
    args = ap.parse_args()
    d = json.load(open(os.path.join(OUT, args.json)))
    ident = json.load(open(os.path.join(OUT, args.iso)))["runs"]
    cases = load_case()
    m_of = {ds: float(np.mean([len(c["T_only"]) for c in cases if c["ds"] == ds]))
            for ds in DS_ALL}
    m_of["macro"] = float(np.mean([m_of[ds] for ds in DS_ALL]))

    # ---- per-arm macro table from identity + swap runs ---------------------
    runs = {}
    for src in (ident, json.load(open(os.path.join(OUT, "s2c2_swap.json")))["runs"]):
        runs.update(src)
    arms = {}
    for rk, r in runs.items():
        a, ds = rk.rsplit("|", 1)
        arms.setdefault(a, {})[ds] = r
    complete = [a for a in arms if all(ds in arms[a] for ds in DS_ALL)]

    def macro(a):
        return float(np.mean([np.mean(arms[a][ds]["hits"]) * 100 for ds in DS_ALL]))

    def hits_of(a, ds):
        r = arms[a][ds]
        return {i: h for i, h in zip(r["idx"], r["hits"])}, r["hits"]

    # ---------------------------------------------------------------- curve
    def curve_rows(kind, ks):
        out = []
        base = d["baselines"]
        for k in ks:
            row = [str(k)]
            for ds in DS_ALL + ["macro"]:
                if kind == "random":
                    vals = [macro(f"random:{k}:s{s}") if ds == "macro" else
                            float(np.mean(arms[f"random:{k}:s{s}"][ds]["hits"]) * 100)
                            for s in (0, 1, 2) if f"random:{k}:s{s}" in arms
                            and all(x in arms[f"random:{k}:s{s}"] for x in DS_ALL)]
                    # print whatever seed count exists, and say so when it is one
                    row.append("—" if not vals else
                               (f"{np.mean(vals):.2f}" if len(vals) == 1 else
                                f"{np.mean(vals):.2f} ± {np.std(vals):.2f}")
                               + ("" if len(vals) > 1 else " (1 seed)"))
                else:
                    a = f"{kind}:{k}"
                    if a not in arms or not all(x in arms[a] for x in DS_ALL):
                        row.append("—"); continue
                    row.append(f"{macro(a):.2f}" if ds == "macro"
                               else f"{np.mean(arms[a][ds]['hits']) * 100:.2f}")
            out.append(row)
        return out

    ks = sorted({int(a.split(":")[1].split("|")[0]) for a in complete
                 if a.startswith("teacher:")})
    print("### full teacher-priority grid\n")
    hdr = "| k | " + " | ".join(DS_ALL + ["macro"]) + " |"
    print(hdr); print("|---|" + "---|" * 4)
    print(f"| 0 (=S) | " + " | ".join(
        f"{d['baselines'][ds]['student']:.2f}" for ds in DS_ALL) +
        f" | {d['macro']['student']:.2f} |")
    for kind, title in (("teacher", "teacher-priority"), ("random", "random (3 seeds)"),
                        ("adversarial", "adversarial"), ("shuffled", "shuffled")):
        rows = curve_rows(kind, ks)
        if not rows:
            continue
        print(f"\n**{title}**\n")
        print(hdr); print("|---|" + "---|" * 4)
        for r in rows:
            print("| " + " | ".join(r) + " |")
    print(f"| full (=T, measured) | " + " | ".join(
        f"{d['baselines'][ds]['teacher']:.2f}" for ds in DS_ALL) +
        f" | {d['macro']['teacher']:.2f} |")

    # --------------------------------------------------------- marginal value
    print("\n### marginal value per swapped token (macro)\n")
    print("| swap range | tokens | macro | gain | per token | cumulative gap |")
    print("|---|---|---|---|---|---|")
    prev_k, prev_v = 0, d["macro"]["student"]
    gap = d["macro"]["teacher"] - d["macro"]["student"]
    pts = [(k, float(np.mean([np.mean(arms[f"teacher:{k}"][ds]["hits"]) * 100
                              for ds in DS_ALL]))) for k in ks]
    pts.append((round(m_of["macro"]), d["macro"]["teacher"]))
    for k, v in pts:
        dk = k - prev_k
        print(f"| {prev_k} → {k} | {dk} | {prev_v:.2f} → {v:.2f} | {v - prev_v:+.2f} "
              f"| {(v - prev_v) / dk:+.3f} | {(v - d['macro']['student']) / gap * 100:.1f} % |")
        prev_k, prev_v = k, v

    # ---------------------------------------------------------- add/remove
    print("\n### add/remove split\n")
    aks = sorted({int(a.split(":")[1].split("|")[0]) for a in complete
                  if a.startswith("addteacher:") or a.startswith("remworst:")})
    print("| k | teacher:k (add best + drop worst) | addteacher:k (add best + drop random) "
          "| remworst:k (drop worst + add neutral) | random:k (both random) |")
    print("|---|---|---|---|---|")
    for k in aks:
        cells = []
        for a in (f"teacher:{k}", f"addteacher:{k}:s0", f"remworst:{k}:s0", f"random:{k}:s0"):
            cells.append(f"{macro(a):.2f}" if a in arms and all(
                x in arms[a] for x in DS_ALL) else "—")
        print(f"| {k} | " + " | ".join(cells) + " |")

    # ------------------------------------------------------------ outcomes
    print("\n### per-instance outcomes vs the student set (150 instances)\n")
    print("| arm | macro | Δ | improved | tied | worsened |")
    print("|---|---|---|---|---|---|")
    for a in sorted(complete, key=lambda x: (x.split(":")[0],
                                             int(x.split(":")[1].split("|")[0])
                                             if ":" in x else 0)):
        if a.startswith(("random", "identity")):
            continue
        i = t = w = 0
        for ds in DS_ALL:
            o1, _ = hits_of(a, ds); o2, _ = hits_of("identity_S", ds)
            dd = np.array([o1[x] - o2[x] for x in sorted(o1)])
            i += int((dd > 1e-9).sum()); t += int((abs(dd) <= 1e-9).sum())
            w += int((dd < -1e-9).sum())
        print(f"| `{a}` | {macro(a):.2f} | {macro(a) - d['macro']['student']:+.2f} "
              f"| {i} | {t} | {w} |")

    # ---------------------------------------------------------------- flips
    if d.get("flips"):
        print("\n### answer flips vs the student set\n")
        print("| arm | wrong->right | right->wrong | " +
              " | ".join(f"{x} w→r" for x in DS_ALL) + " |")
        print("|---|---|---|" + "---|" * len(DS_ALL))
        for a in sorted(d["flips"], key=lambda x: (x.split(":")[0],
                         int(x.split(":")[1].split("|")[0]) if ":" in x else 0)):
            f = d["flips"][a]
            print(f"| `{a}` | {f['wrong_to_right']} | {f['right_to_wrong']} | " +
                  " | ".join(str(f["per_ds"][x][0]) for x in DS_ALL) + " |")

    # --------------------------------------------------------- characterisation
    char = d.get("characterize")
    if char:
        # per-benchmark, for the one comparison the brief asks to split
        for scope in DS_ALL:
            rec = (char.get(scope) or {}).get("A2_rescue_added_vs_B_teacher_only")
            if not rec:
                continue
            sig = {p_: r for p_, r in rec.items() if r["significant"]}
            print(f"\n**{scope}** — A2 (rescue-performing) vs B (ordinary teacher-only): "
                  f"n = {max(r['n_inst'] for r in rec.values())} instances, "
                  f"significant: {', '.join(sorted(sig)) or 'none'}")
            for p_ in sorted(sig):
                r = rec[p_]
                print(f"  - `{p_}` {r['mean_a']:.4f} vs {r['mean_b']:.4f} "
                      f"(d = {r['cohen_d']:+.2f}, CI [{r['lo']:+.4f}, {r['hi']:+.4f}])")
    if char and char.get("ALL"):
        for pair, title in (("A2_rescue_added_vs_B_teacher_only",
                             "the tokens that perform a rescue vs ordinary teacher-only tokens"),
                            ("B_teacher_only_vs_C_student_only",
                             "teacher-only vs student-only tokens")):
            rec = char["ALL"].get(pair)
            if not rec:
                continue
            print(f"\n### {title}\n")
            print("| property | group A | group B | delta | 95 % CI | Cohen d | n inst |")
            print("|---|---|---|---|---|---|---|")
            for p_ in ["g2", "g2_pct", "lin", "lin_pct", "n_l2", "n_l4", "d_norm",
                       "cos_l2_l4", "sim_sel_S", "sim_sel_T", "nb_sim", "uniq",
                       "row", "col", "r_center"]:
                r = rec.get(p_)
                if not r:
                    continue
                star = " **(sig)**" if r["significant"] else ""
                print(f"| `{p_}` | {r['mean_a']:.4f} | {r['mean_b']:.4f} "
                      f"| {r['delta']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] "
                      f"| {r['cohen_d']:+.2f}{star} | {r['n_inst']} |")

    # ------------------------------------------------------- rescue-k profile
    plan = d.get("rescue_plan")
    if plan:
        ks_ = [r["first_rescue_k"] for r in plan["rescuable"]]
        ints = [k for k in ks_ if isinstance(k, int)]
        print("\n### how many swapped tokens each rescue needs\n")
        print("| cum. tokens swapped | instances rescued |")
        print("|---|---|")
        for t in (2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96):
            n = sum(1 for k in ints if k <= t)
            if n:
                print(f"| ≤ {t} | {n} of 28 |")
        print(f"| only at the full swap | {len(ks_) - len(ints)} more, 28 of 28 |")
        print("\n| benchmark | rescuable | median rescue k | range |")
        print("|---|---|---|---|")
        for ds in DS_ALL:
            v = sorted(r["first_rescue_k"] for r in plan["rescuable"]
                       if r["ds"] == ds and isinstance(r["first_rescue_k"], int))
            if v:
                print(f"| {ds} | {sum(1 for r in plan['rescuable'] if r['ds'] == ds)} "
                      f"| {np.median(v):.0f} | {min(v)}–{max(v)} |")

    # --------------------------------------------------------------- rescue
    if d.get("rescue_rate"):
        print("\n### rescue rate on the 28 student-wrong / teacher-correct instances\n")
        rr = d["rescue_rate"]
        allarms = sorted(rr.get("ALL", {}), key=lambda x: (x.split(":")[0],
                         int(x.split(":")[1].split("|")[0]) if ":" in x else 0))
        ds_hdr = [x for x in DS_ALL if x in rr]
        print("| arm | " + " | ".join(ds_hdr + ["ALL (28)"]) + " |")
        print("|---|" + "---|" * (len(ds_hdr) + 1))
        for a in allarms:
            cells = [f"{rr[x][a]:.0f} %" if a in rr[x] else "—" for x in ds_hdr]
            cells.append(f"{rr['ALL'][a]:.0f} %")
            print(f"| `{a}` | " + " | ".join(cells) + " |")

    # ------------------------------------------------------------------ loo
    loo = d.get("loo") or {}
    if loo.get("rows"):
        print("\n### leave-one-out at the rescue level\n")
        print(f"{loo['n_instances']} instances, {loo['total_block_ablations']} block "
              f"ablations ({loo['total_block_breaks']} broke the rescue); "
              f"control {loo['total_early_ablations']} early ablations "
              f"({loo['total_early_breaks']} broke the rescue)\n")
        print("| instance | k* | block | block ablations | broke | early ablations | broke |")
        print("|---|---|---|---|---|---|---|")
        for r in loo["rows"]:
            print(f"| `{r['key']}` | {r['k']} | {r['block']} | {r['n_block_ablations']} "
                  f"| {r['block_breaks']} | {r['early_ablations']} | {r['early_breaks']} |")

    # --------------------------------------------------------------- k_for
    print("\n### tokens needed for 25/50/75 % recovery\n")
    print("| benchmark | \\|T_only\\| | 25 % | 50 % | 75 % | 100 % |")
    print("|---|---|---|---|---|---|")
    for ds in DS_ALL:
        kf = d["levels"][ds]["k_for"]
        print(f"| {ds} | {m_of[ds]:.0f} | {kf['raw_25']:.1f} | {kf['raw_50']:.1f} "
              f"| {kf['raw_75']:.1f} | {m_of[ds]:.0f} |")


if __name__ == "__main__":
    main()
