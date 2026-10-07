"""Generate llava_test_CQM-I.json for the OFFICIAL EADP/LLaVA SQA protocol.

The official EADP LLaVA fork (scripts/v1_5/eval/sqa.sh) evaluates SQA with
SPLIT="llava_test_CQM-I" + conv vicuna_v1, but no shipped converter has the
'-I' output branch.  For the *test* split build_prompt_chatbot is called with
test_example=True semantics, where every format collapses to output="Answer:"
-- so "CQM-I" is generated with is_test=True and the input side follows the
official CQM branch verbatim.  Post-processing mirrors convert_to_llava
exactly (prefix stripping, <image> placement, indent=2 dump).

Audit item #1 (2026-10-08): our previous SQA runs used llava_test_QCM-LEPA +
conv llava_v1, which is NOT the official protocol.
"""
import json
import os
import sys

sys.path.insert(0, "/media/disk2/YZX/research/EADP/LLaVA/scripts")
from convert_sqa_to_llava_base_prompt import build_prompt_chatbot  # noqa: E402

BASE = "/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval/scienceqa"


def main() -> None:
    splits = json.load(open(os.path.join(BASE, "pid_splits.json")))
    problems = json.load(open(os.path.join(BASE, "problems.json")))

    test_problems = build_prompt_chatbot(
        problems, splits["test"], "CQM-I", use_caption=False, is_test=True)

    target_format = []
    for prob_id, (inp, out) in test_problems.items():
        if inp.startswith("Question: "):
            inp = inp.replace("Question: ", "")
        if out.startswith("Answer: "):
            out = out.replace("Answer: ", "")
        raw = problems[prob_id]
        if raw["image"] is None:
            target_format.append({
                "id": prob_id,
                "conversations": [
                    {"from": "human", "value": f"{inp}"},
                    {"from": "gpt", "value": f"{out}"},
                ],
            })
        else:
            target_format.append({
                "id": prob_id,
                "image": os.path.join(prob_id, raw["image"]),
                "conversations": [
                    {"from": "human", "value": f"{inp}\n<image>"},
                    {"from": "gpt", "value": f"{out}"},
                ],
            })

    out_path = os.path.join(BASE, "llava_test_CQM-I.json")
    with open(out_path, "w") as f:
        json.dump(target_format, f, indent=2)
    print(f"n = {len(target_format)}")
    print(f"with image = {sum(1 for x in target_format if 'image' in x)}")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
