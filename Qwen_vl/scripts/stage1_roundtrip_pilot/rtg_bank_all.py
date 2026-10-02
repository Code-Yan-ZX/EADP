"""Build all RTG scorer banks (rtg/flat/shuf x 3 DEV datasets)."""

from __future__ import annotations

import rtg_common as RC


def main():
    model = eng = None
    for scorer in RC.SCORERS:
        for ds in RC.DS_LIST:
            _, model, eng = RC.build_scorer_bank(scorer, ds, model=model,
                                                 eng=eng)


if __name__ == "__main__":
    main()
