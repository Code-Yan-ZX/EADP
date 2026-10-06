"""VizWiz-val scorer: official VQA accuracy (VizWiz repo evaluate_vqa.py /
vqa_eval.py logic: contractions + number map + articles + punctuation, then
min(1, #GT-matches/3)).  Community convention (user-approved 2026-10-06):
reproduce on the val split because the official test server is retired.
"""
import argparse
import json
import re
import string


def _prep(ans):
    ans = ans.lower()
    contractions = {"isnt": "is not", "arent": "are not", "wasnt": "was not",
                    "werent": "were not", "hasnt": "has not",
                    "havent": "have not", "wont": "will not", "dont": "do not",
                    "doesnt": "does not", "didnt": "did not", "cant": "cannot",
                    "couldnt": "could not", "shouldnt": "should not",
                    "wouldnt": "would not", "won't": "will not"}
    for k, v in contractions.items():
        ans = ans.replace(k, v)
    manual = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
              "four": "4", "five": "5", "six": "6", "seven": "7",
              "eight": "8", "nine": "9", "ten": "10"}
    ans = " ".join(manual.get(w, w) for w in ans.split())
    period = r"([.\-\"',:!()])"
    ans = re.sub(period, r" \1 ", ans)
    ans = re.sub(period, " ", ans)
    ans = " ".join(w for w in ans.split() if w not in
                   ("a", "an", "the"))
    return ans.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-file", required=True)
    p.add_argument("--result-file", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    gt = json.load(open(a.gt_file))
    gts = {i: [x["answer"] for x in q["answers"]]
           for i, q in enumerate(gt["questions"] if isinstance(gt, dict)
                                else gt)}
    preds = [json.loads(l) for l in open(a.result_file)]
    n = acc_sum = 0
    per = []
    for d in preds:
        qid = d["question_id"]
        g = gts.get(qid)
        if g is None:
            continue
        prepped = [_prep(x) for x in g]
        pred = _prep(str(d["text"]))
        m = sum(1 for x in prepped if x == pred)
        q_acc = min(1.0, m / 3.0)
        acc_sum += q_acc
        per.append({"question_id": qid, "acc": q_acc})
        n += 1
    out = {"n": n, "accuracy": round(100 * acc_sum / n, 2),
           "metric": "VQA-accuracy(min(1,match/3)), VizWiz val",
           "protocol": "val-reproduction (official test server retired)"}
    json.dump({"summary": out, "per_question": per}, open(a.out, "w"))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
