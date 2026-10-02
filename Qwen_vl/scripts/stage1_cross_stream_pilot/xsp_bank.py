"""Stage-1 Cross-Stream Pilot — freeze the scorer banks (protocol §5).

One pass per (scorer, dataset) over the DEV-100 manifest subset: official
facility anchors for the EADP scorer, cross_stream / uniform / main_residual
weights + official facility for the new scorers; cos-argmax assignment and
Stage-1 diagnostics.  Downstream accuracy arms ONLY read the frozen banks.

Usage: python xsp_bank.py --scorers eadp,cross_stream,uniform,main_residual
       python xsp_bank.py --scorers cross_stream --datasets DocVQA_VAL --limit 3
"""

from __future__ import annotations

import argparse

import xsp_common as XC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorers", default=",".join(XC.SCORERS))
    ap.add_argument("--datasets", default=",".join(XC.DS_LIST))
    ap.add_argument("--split", default=XC.SPLIT, choices=["dev"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    model = eng = None
    for scorer in args.scorers.split(","):
        for ds in args.datasets.split(","):
            _, model, eng = XC.build_scorer_bank(scorer, ds, args.split,
                                                 args.limit, model=model,
                                                 eng=eng)


if __name__ == "__main__":
    main()
