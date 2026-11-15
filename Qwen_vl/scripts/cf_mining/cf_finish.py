"""Run the matched comparison and reference audits in one model process."""
from __future__ import annotations

import json
from pathlib import Path

import cf_compare


def train_generation_sanity(experiment, out):
    """Separate memorizing fit labels from a train/generation interface bug."""
    torch, api, eng = experiment.torch, experiment.api, experiment.eng
    previous = {k: v.detach().cpu().clone() for k, v in experiment.pool.state_dict().items()}
    result = dict(complete=False, purpose="post-hoc interface diagnostic, no model selection", arms={})
    try:
        api.set_decoder_checkpointing(eng, False)
        with torch.no_grad():
            for arm in ("ordinary", "online"):
                checkpoint = torch.load(out / arm / "checkpoint.pt", map_location="cpu", weights_only=False)
                experiment.pool.load_state_dict(checkpoint["compressor"])
                rows = []
                for pair in experiment.plan["fit"][:8]:
                    rec = experiment.moved(pair["cache_original"], pair["original_answer"])
                    prediction = api.generate_answer(eng, rec, experiment.pool,
                                                     max_new_tokens=experiment.args.max_new_tokens)
                    logits = api.forward_answer(eng, rec, experiment.pool).float()
                    target = api.answer_token_ids(eng, rec["answer"], logits.device)
                    nll = experiment.F.cross_entropy(logits, target).item()
                    rows.append(dict(pair_id=pair["pair_id"], answer=pair["original_answer"],
                                     teacher_training_answer=rec["answer"], prediction=prediction,
                                     correct=cf_compare.correct(prediction, pair["original_answer"]),
                                     teacher_forced_answer_nll=nll))
                result["arms"][arm] = dict(n=len(rows), correct=sum(r["correct"] for r in rows),
                                            records=rows)
                cf_compare.atomic_json(out / "train_generation_sanity.json", result)
    finally:
        experiment.pool.load_state_dict(previous)
    result["complete"] = True
    result["pool_restored"] = True
    cf_compare.atomic_json(out / "train_generation_sanity.json", result)


def main():
    instances = []
    original = cf_compare.Experiment

    class RetainedExperiment(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    cf_compare.Experiment = RetainedExperiment
    try:
        cf_compare.main()
    finally:
        cf_compare.Experiment = original
    if not instances:
        return
    experiment = instances[-1]
    if experiment.args.stage != "compare":
        return
    out = Path(experiment.args.out)
    result = json.loads((out / "comparison.json").read_text())
    if not result.get("complete"):
        raise RuntimeError("references require the complete matched comparison")
    train_generation_sanity(experiment, out)
    from cf_references import run_references
    mining = json.loads(Path(experiment.args.mining).read_text())
    run_references(experiment, mining, out,
                   max_sources=experiment.args.eval_sources, max_invariants=8)


if __name__ == "__main__":
    main()
