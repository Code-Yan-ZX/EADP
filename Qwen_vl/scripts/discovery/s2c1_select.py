"""
S2-C1 helper: pick the one arm per scorer family that goes to the accuracy
translation, using ONLY the validation split.

Selection metric is val Top-256 overlap against the teacher -- the same objective
the early stopping uses, and the quantity the downstream Top-K selector actually
consumes. Deliberately not Spearman (S2-C0: agreement with the teacher is
anti-indicative) and deliberately not the held-out 150 (which must stay clean).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c1")
    ap.add_argument("--metric", default="overlap256",
                    choices=["overlap256", "auroc", "ap"])
    args = ap.parse_args()

    R = json.load(open(os.path.join(OUT, f"{args.tag}_train.json")))["results"]
    best = {}
    for arm, res in R.items():
        fam = res["family"]
        key = (res["val"][args.metric], -res["n_params"])
        if fam not in best or key > best[fam][0]:
            best[fam] = (key, arm)
    picks = [best[f][1] for f in sorted(best)]
    print(f"[select] val {args.metric}: "
          + ", ".join(f"{f}={best[f][1]} ({best[f][0][0]:.4f})" for f in sorted(best)))
    print(" ".join(picks))


if __name__ == "__main__":
    main()
