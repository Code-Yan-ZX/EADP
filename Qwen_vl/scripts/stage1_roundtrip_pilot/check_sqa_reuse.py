"""CPU gate for the local reconstructed CQM-I/vicuna SQA protocol.

This verifies local input identity; the author's actual CQM-I question
file is unavailable, so passing it does not establish paper equivalence.
"""
import argparse
import json

from rescore_official_20261008 import sqa_check


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--result-file", required=True)
    args = ap.parse_args()
    if not args.question_file.endswith("llava_test_CQM-I.json") or "_CQMI_vicuna" not in args.result_file:
        ap.exit(1, "SQA protocol gate failed: use CQM-I + separately named CQMI_vicuna predictions.\n")
    result = sqa_check(args.result_file, args.question_file)
    questions = json.load(open(args.question_file))
    expected = {}
    for q in questions:
        prompt = q["conversations"][0]["value"].replace("<image>", "").strip()
        if "image" in q:
            prompt = "<image>\n" + prompt
        prompt += "\nAnswer with the option's letter from the given choices directly."
        expected[q["id"]] = prompt
    result["prompt_mismatches"] = sum(
        row.get("prompt") != expected.get(row["question_id"])
        for row in (json.loads(line) for line in open(args.result_file)))
    if not result["complete"] or result["failed_or_empty_text"] or result["prompt_mismatches"]:
        ap.exit(1, f"SQA reuse gate failed: {json.dumps(result)}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
