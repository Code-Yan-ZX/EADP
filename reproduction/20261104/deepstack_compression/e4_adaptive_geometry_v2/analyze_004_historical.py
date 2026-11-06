"""Evaluate all frozen descriptive routing rules after independent feature review."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

D = Path(__file__).resolve().parent; B = D.parent
GATE_SHA = '609cb37b8cd82e5f98fd312d5e4bfd5abd9cb74388d2949bff885fdea50245f2'
ROUTE_SHA = "20922f59d5608a50a41e746817f1b5fa6c89af30f6aca32f651e699ef0b22028"


def load(name, path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected or name in sys.modules: raise ValueError("unreviewed/preloaded module")
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    try: spec.loader.exec_module(m)
    except BaseException:
        del sys.modules[name]; raise
    return m


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    if a.out.resolve() != D / "analysis001" or a.out.exists(): raise ValueError("output destination")
    gate = load("_geometry_analysis_gate", D / "analysis_gate.py", GATE_SHA); evidence = gate.admit()
    route = load("_geometry_route", D / "route.py", ROUTE_SHA)
    def lines(path):
        with Path(path).open() as f: return [json.loads(x) for x in f if x.strip()]
    features = lines(D / "features001/features.jsonl"); scores = lines(B / "e4_analysis/score_panel003/scores.jsonl")
    tasks, assignments = route.analyze(features, scores)
    if gate.sha(D / "features001/features.jsonl") != evidence["feature_sha256"] or gate.sha(B / "e4_analysis/score_panel003/scores.jsonl") != evidence["scores_sha256"]: raise ValueError("inputs changed during analysis")
    a.out.mkdir(exist_ok=False)
    with (a.out / "assignments.jsonl").open("x") as f:
        for r in assignments: f.write(json.dumps(r, sort_keys=True) + "\n")
        f.flush(); os.fsync(f.fileno())
    result = {"status": "completed", "tasks": tasks, "assignments_sha256": gate.sha(a.out / "assignments.jsonl"), "admission": evidence,
              "GT_opened": False, "new_predictions_opened": False,
              "limitations": ["posthoc DEV exploration; all six fixed rules reported", "batch ranking is not an online policy", "token count is not equal compute or latency", "K128 feature pass and possible second K512 generation must be included in any real cost"]}
    with (a.out / "analysis.json").open("x") as f: json.dump(result, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())


if __name__ == "__main__": main()
