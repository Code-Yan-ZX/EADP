"""GQA testdev scorer: official GT (testdev_balanced_questions.json),
exact match after lowercase+strip -- the same rule used for the v1.5/NeXT
rounds (records_20261003 §14.2)."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-file", required=True)
    ap.add_argument("--result-file", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gt = json.load(open(a.gt_file))
    n = c = 0
    per = []
    for line in open(a.result_file):
        d = json.loads(line)
        qid = str(d["question_id"])
        g = gt.get(qid)
        if g is None:
            continue
        n += 1
        ok = str(d["text"]).strip().lower() == str(g["answer"]).strip().lower()
        c += ok
        per.append({"question_id": qid, "correct": bool(ok)})
    out = {"n": n, "accuracy": round(100 * c / n, 2),
           "metric": "exact-match lowercase-strip, GQA testdev"}
    json.dump({"summary": out, "per_question": per}, open(a.out, "w"))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
