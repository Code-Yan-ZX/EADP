"""Supplementary native references, reusing the comparison's loaded model.

``run_references`` adds same-source and same-pair 512-pixel resolution
baselines and a small teacher-verified unrelated-question check. These are
quality checks, not a latency benchmark or a test of field swapping/arithmetic
relations. The function reuses a loaded model. The explicit ``--from-gate``
CLI can load one inference-only model even when the training gate failed.
"""
from __future__ import annotations

from pathlib import Path
import random
import time

from cf_compare import atomic_json, correct, group_id, normalize, paired_bootstrap, split_name


def _message(image, question):
    return [dict(type="image", value=str(image)),
            dict(type="text", value=question + "\nAnswer with the value only.")]


def _aliases(value, fallback=None):
    if isinstance(value, (list, tuple)):
        result = [str(item) for item in value if item is not None]
    elif value is None:
        result = []
    else:
        result = [str(value)]
    if not result and fallback is not None:
        result = [str(fallback)]
    return result


def _matches(prediction, aliases):
    return any(correct(prediction, answer) for answer in aliases)


def _rate(rows, key):
    available = [row for row in rows if row.get(key) is not None]
    return dict(n=len(available), correct=sum(bool(row[key]) for row in available),
                accuracy=sum(bool(row[key]) for row in available) / len(available)
                if available else None)


def _rres_summary(rows):
    conditional = [row for row in rows if row.get("teacher_correct")]
    return dict(n_sources=len(rows),
                all_sources={arm: _rate(rows, arm + "_correct")
                             for arm in ("teacher", "b2", "rres512")},
                teacher_correct_sources={arm: _rate(conditional, arm + "_correct")
                                         for arm in ("b2", "rres512")},
                rres512_minus_b2_rescued=sum(row["rres512_correct"] and not row["b2_correct"]
                                           for row in rows),
                rres512_minus_b2_broken=sum(row["b2_correct"] and not row["rres512_correct"]
                                          for row in rows))


def _invariant_summary(rows):
    accepted = [row for row in rows if row["teacher_both_correct"]]
    return dict(attempted_pairs=len(rows),
                teacher_both_correct_pairs=len(accepted),
                teacher_rejected_pairs=len(rows) - len(accepted),
                accepted={arm: _rate(accepted, arm + "_both_correct")
                          for arm in ("teacher", "b2", "online")},
                online_rescued=sum(row["online_both_correct"] and not row["b2_both_correct"]
                                   for row in accepted),
                online_broken=sum(row["b2_both_correct"] and not row["online_both_correct"]
                                  for row in accepted))


def _rres_pair_summary(rows, seed):
    def comparison_rows(arm):
        return [dict(pair_id=row["pair_id"], group_id=row["group_id"],
                     both_correct=row[arm + "_both_correct"]) for row in rows]

    return dict(n_pairs=len(rows), n_documents=len({row["group_id"] for row in rows}),
                both_correct={arm: _rate(rows, arm + "_both_correct")
                              for arm in ("teacher", "b2", "rres512")},
                strict_collapse={arm: dict(
                    n=len(rows), count=sum(row[arm + "_strict_collapse"] for row in rows),
                    rate=sum(row[arm + "_strict_collapse"] for row in rows) / len(rows) if rows else None)
                                 for arm in ("b2", "rres512")},
                rres512_minus_b2_rescued=sum(row["rres512_both_correct"] and not row["b2_both_correct"]
                                           for row in rows),
                rres512_minus_b2_broken=sum(row["b2_both_correct"] and not row["rres512_both_correct"]
                                          for row in rows),
                b2_strict_collapses_repaired_to_both_correct=sum(
                    row["b2_strict_collapse"] and row["rres512_both_correct"] for row in rows),
                rres512_minus_b2_paired=paired_bootstrap(
                    comparison_rows("b2"), comparison_rows("rres512"), seed))


def _run_rres(experiment, mining, outdir, max_sources, max_pairs=32):
    import vlmeval.vlm.qwen3_vl.model_fixed_res as fixed

    eng, torch = experiment.eng, experiment.torch
    wrapper = eng.vlm
    joined = {str(record["pair_id"]): record for record in mining.get("records", [])}
    source_pool = experiment.plan.get("sources", mining.get("sources", []))
    sources = [source for source in source_pool
               if split_name(source) != "fit"][:max_sources]
    heldout_pool = experiment.plan.get("heldout", [])
    pairs = [pair for pair in heldout_pool
             if split_name(pair) != "fit" and pair.get("teacher_valid")][:min(max_pairs, 32)]
    missing = [source["pair_id"] for source in sources if str(source["pair_id"]) not in joined]
    if missing:
        raise ValueError(f"untouched source IDs absent from mining records: {missing}")
    if int(fixed.QWEN3_FIXED_RESOLUTION) != 1024:
        raise ValueError("reference checks must start at the comparison's native 1024 resolution")
    absent = object()
    old_min = getattr(wrapper, "min_pixels", absent)
    old_max = getattr(wrapper, "max_pixels", absent)
    result = dict(reference="native keep-all at 512x512", input_visual_tokens=256,
                  decoder_visual_tokens=256, complete=False, records=[],
                  latency_claim=False)
    pair_result = dict(reference="native keep-all at 512x512 on the same held-out edited pairs",
                       input_visual_tokens=256, decoder_visual_tokens=256,
                       selection="comparison held-out teacher-valid pairs; no selection by compressed-model result",
                       complete=False, records=[], latency_claim=False)
    t0 = time.time()
    try:
        fixed.QWEN3_FIXED_RESOLUTION = 512
        wrapper.min_pixels = wrapper.max_pixels = 512 * 512
        with torch.no_grad():
            for source in sources:
                pair = joined[str(source["pair_id"])]
                aliases = _aliases(pair.get("answers"), source.get("answer", pair["original_answer"]))
                prep = eng.prepare(_message(pair["source_image"], pair["question"]), "DocVQA_VAL")
                if int(prep["n_vis"]) != 256:
                    raise RuntimeError(f"512 reference produced {prep['n_vis']} visual tokens, expected 256")
                V, DS = eng.encode(prep)
                if V.shape[0] != 256 or any(stream.shape[0] != 256 for stream in DS):
                    raise RuntimeError("512 reference visual/DeepStack token count mismatch")
                state = eng.prefill(prep, V, DS, None, deepstack=True, pos="mrope3d")
                _, prediction = eng.decode(state, max_new_tokens=experiment.args.max_new_tokens)
                teacher = source.get("teacher", source.get("teacher_prediction"))
                baseline = source.get("student", source.get("student_prediction"))
                if teacher is None or baseline is None:
                    raise ValueError("untouched source lacks stored native teacher/B2 predictions")
                row = dict(pair_id=str(pair["pair_id"]), group_id=group_id(pair),
                           split=split_name(source), question=pair["question"],
                           image=pair["source_image"], answers=aliases,
                           actual_input_visual_tokens=int(prep["n_vis"]),
                           actual_decoder_visual_tokens=int(state.n_vis),
                           teacher_prediction=teacher, b2_prediction=baseline,
                           rres512_prediction=prediction,
                           teacher_correct=_matches(teacher, aliases),
                           b2_correct=_matches(baseline, aliases),
                           rres512_correct=_matches(prediction, aliases))
                if row["actual_decoder_visual_tokens"] != 256:
                    raise RuntimeError("512 reference changed decoder token count")
                result["records"].append(row)
                result["summary"] = _rres_summary(result["records"])
                result["wall_seconds"] = time.time() - t0
                atomic_json(outdir / "rres512_sources.json", result)
                print(f"[reference rres512] {len(result['records'])}/{len(sources)} "
                      f"correct={row['rres512_correct']} tokens=256", flush=True)
                del prep, V, DS, state
            for pair in pairs:
                row = dict(pair_id=str(pair["pair_id"]), group_id=group_id(pair),
                           split=split_name(pair), question=pair["question"],
                           predictions={}, answers={}, input_tokens={}, decoder_tokens={})
                for side in ("original", "edited"):
                    alias_value = pair.get(side + "_answers")
                    if alias_value is None and side == "original":
                        alias_value = pair.get("answers")
                    aliases = _aliases(alias_value, pair[side + "_answer"])
                    prep = eng.prepare(_message(pair[side + "_image"], pair["question"]), "DocVQA_VAL")
                    if int(prep["n_vis"]) != 256:
                        raise RuntimeError("512 counterfactual reference requires 256 native visual tokens")
                    V, DS = eng.encode(prep)
                    if V.shape[0] != 256 or any(stream.shape[0] != 256 for stream in DS):
                        raise RuntimeError("512 pair reference visual/DeepStack count mismatch")
                    state = eng.prefill(prep, V, DS, None, deepstack=True, pos="mrope3d")
                    _, prediction = eng.decode(state, max_new_tokens=experiment.args.max_new_tokens)
                    if int(state.n_vis) != 256:
                        raise RuntimeError("512 pair reference decoder visual count changed")
                    row["input_tokens"][side] = int(prep["n_vis"])
                    row["decoder_tokens"][side] = int(state.n_vis)
                    row["answers"][side] = aliases
                    row["predictions"][side] = dict(
                        teacher=pair[side + "_teacher_prediction"],
                        b2=pair[side + "_student_prediction"], rres512=prediction)
                    del prep, V, DS, state
                for arm in ("teacher", "b2", "rres512"):
                    row[arm + "_both_correct"] = all(
                        _matches(row["predictions"][side][arm], row["answers"][side])
                        for side in ("original", "edited"))
                    row[arm + "_strict_collapse"] = (
                        not row[arm + "_both_correct"] and
                        normalize(row["predictions"]["original"][arm]) ==
                        normalize(row["predictions"]["edited"][arm]))
                if not row["teacher_both_correct"]:
                    raise ValueError("stored teacher-valid pair fails its ground-truth answer aliases")
                pair_result["records"].append(row)
                pair_result["summary"] = _rres_pair_summary(pair_result["records"], experiment.args.seed)
                pair_result["wall_seconds_including_source_reference"] = time.time() - t0
                atomic_json(outdir / "rres512_pairs.json", pair_result)
                print(f"[reference rres512 pairs] {len(pair_result['records'])}/{len(pairs)} "
                      f"both_correct={row['rres512_both_correct']} "
                      f"collapse={row['rres512_strict_collapse']} tokens=256", flush=True)
    finally:
        fixed.QWEN3_FIXED_RESOLUTION = 1024
        for name, value in (("min_pixels", old_min), ("max_pixels", old_max)):
            if value is absent:
                delattr(wrapper, name)
            else:
                setattr(wrapper, name, value)
    result["complete"] = True
    result["summary"] = _rres_summary(result["records"])
    result["resolution_restored"] = 1024
    result["wall_seconds"] = time.time() - t0
    pair_result["complete"] = True
    pair_result["summary"] = _rres_pair_summary(pair_result["records"], experiment.args.seed)
    pair_result["resolution_restored"] = 1024
    pair_result["wall_seconds_including_source_reference"] = time.time() - t0
    result["pair_summary"] = pair_result["summary"]
    atomic_json(outdir / "rres512_pairs.json", pair_result)
    atomic_json(outdir / "rres512_sources.json", result)
    return result


def _online_checkpoint(experiment, outdir):
    choices = [Path(experiment.args.out) / "online" / "checkpoint.pt",
               outdir / "online" / "checkpoint.pt",
               outdir.parent / "online" / "checkpoint.pt"]
    for path in choices:
        if path.exists():
            return path
    return None


def _run_invariants(experiment, mining, outdir, max_invariants):
    eng, torch = experiment.eng, experiment.torch
    candidates = [record for record in mining.get("records", [])
                  if split_name(record) != "fit" and record.get("invariant_questions")]
    candidates.sort(key=lambda record: str(record["pair_id"]))
    random.Random(experiment.args.seed + 41).shuffle(candidates)
    candidates = candidates[:max_invariants]
    result = dict(complete=False, candidate_pairs=len(candidates), records=[],
                  acceptance="full teacher correct on both images against any existing sibling-answer alias",
                  scope="unrelated sibling questions only; no field-swap or arithmetic claim")
    checkpoint_path = _online_checkpoint(experiment, outdir)
    if not candidates or checkpoint_path is None:
        result.update(complete=True, skipped_reason="no held-out sibling candidates" if not candidates
                      else "online checkpoint unavailable", summary=_invariant_summary([]))
        atomic_json(outdir / "invariant_questions.json", result)
        return result
    original_pool = {name: value.detach().cpu().clone()
                     for name, value in experiment.pool.state_dict().items()}
    saved_training_mode = experiment.pool.training
    t0 = time.time()
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        experiment.pool.load_state_dict(checkpoint["compressor"])
        experiment.pool.eval()
        result["checkpoint"] = str(checkpoint_path.resolve())
        with torch.no_grad():
            for pair in candidates:
                sibling = pair["invariant_questions"][0]
                aliases = _aliases(sibling.get("answers"), sibling.get("answer"))
                if not aliases:
                    raise ValueError("invariant sibling has no ground-truth aliases")
                row = dict(pair_id=str(pair["pair_id"]), group_id=group_id(pair),
                           split=split_name(pair), question=sibling["question"],
                           sibling_question_id=sibling.get("question_id"), answers=aliases,
                           encodes=0, predictions={})
                for side in ("original", "edited"):
                    message = _message(pair[side + "_image"], sibling["question"])
                    prep = eng.prepare(message, "DocVQA_VAL")
                    if int(prep["n_vis"]) != 1024:
                        raise RuntimeError("invariant checks require native 1024 visual input tokens")
                    V, DS = eng.encode(prep)
                    row["encodes"] += 1
                    text_mean, text_seq = eng.instruction_embeds(message, "DocVQA_VAL")
                    context = dict(engine=eng, prep=prep, V=V, DS=DS,
                                   text_mean=text_mean, text_seq=text_seq)
                    keep = eng.select("b2", 256, context)
                    if keep.numel() != 256:
                        raise RuntimeError("invariant B2 selector changed the visual token budget")
                    selected_V, selected_DS = experiment.pool(V, DS, keep)
                    side_predictions = {}
                    for arm in ("teacher", "b2", "online"):
                        selected = None if arm == "teacher" else keep
                        kwargs = dict(V_sel=selected_V, DS_sel=selected_DS) if arm == "online" else {}
                        state = eng.prefill(prep, V, DS, selected, deepstack=True, pos="mrope3d", **kwargs)
                        _, prediction = eng.decode(state, max_new_tokens=experiment.args.max_new_tokens)
                        side_predictions[arm] = prediction
                        del state
                    row["predictions"][side] = side_predictions
                    del prep, V, DS, text_mean, text_seq, context, selected_V, selected_DS
                for arm in ("teacher", "b2", "online"):
                    row[arm + "_both_correct"] = all(
                        _matches(row["predictions"][side][arm], aliases) for side in ("original", "edited"))
                row["accepted_invariant"] = row["teacher_both_correct"]
                result["records"].append(row)
                result["summary"] = _invariant_summary(result["records"])
                result["wall_seconds"] = time.time() - t0
                atomic_json(outdir / "invariant_questions.json", result)
                print(f"[reference invariant] {len(result['records'])}/{len(candidates)} "
                      f"teacher_valid={row['accepted_invariant']} "
                      f"b2={row['b2_both_correct']} online={row['online_both_correct']}", flush=True)
    finally:
        experiment.pool.load_state_dict(original_pool)
        experiment.pool.train(saved_training_mode)
    result["complete"] = True
    result["pool_restored"] = all(torch.equal(value.detach().cpu(), original_pool[name])
                                   for name, value in experiment.pool.state_dict().items())
    result["summary"] = _invariant_summary(result["records"])
    result["wall_seconds"] = time.time() - t0
    atomic_json(outdir / "invariant_questions.json", result)
    return result


def run_references(experiment, mining, outdir, max_sources=24, max_invariants=8, max_pairs=32):
    """Run same-source references with a live ``cf_compare.Experiment``.

    ``mining`` may be the loaded mining dictionary or its JSON file path.
    ``outdir`` holds supplementary artifacts; the online checkpoint is found
    under ``experiment.args.out``.  The model is never reloaded.  The pooling
    state and fixed-resolution wrapper settings are restored before returning.
    """
    if max_sources < 0 or max_invariants < 0 or not 0 <= max_pairs <= 32:
        raise ValueError("reference limits must be nonnegative and max_pairs must be at most 32")
    if not isinstance(mining, dict):
        import json
        mining = json.loads(Path(mining).read_text(encoding="utf-8"))
    if mining.get("stage") != "complete":
        raise ValueError("supplementary references require completed mining records")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    training_dir = Path(experiment.args.out)
    sanity_path = training_dir / "train_generation_sanity.json"
    if (not sanity_path.exists() and
            all((training_dir / arm / "checkpoint.pt").exists()
                for arm in ("ordinary", "online"))):
        # Import lazily to reuse the model retained by the finishing runner.
        from cf_finish import train_generation_sanity
        train_generation_sanity(experiment, training_dir)
    was_checkpointed = bool(getattr(experiment.eng.text, "is_gradient_checkpointing", False))
    was_training = experiment.eng.text.training
    experiment.api.set_decoder_checkpointing(experiment.eng, False)
    t0 = time.time()
    try:
        rres = _run_rres(experiment, mining, outdir, max_sources, max_pairs)
        invariants = _run_invariants(experiment, mining, outdir, max_invariants)
    finally:
        experiment.api.set_decoder_checkpointing(experiment.eng, was_checkpointed and was_training)
    result = dict(complete=True, rres512=rres["summary"], rres512_pairs=rres["pair_summary"],
                  invariants=invariants["summary"],
                  wall_seconds=time.time() - t0,
                  limitation="small supplementary quality checks; no unseen-layout, field-binding, arithmetic, or latency claim")
    atomic_json(outdir / "reference_summary.json", result)
    return result


def main():
    import argparse
    import json
    from types import SimpleNamespace
    from cf_compare import DEFAULT_OUT, Experiment, make_plan

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-gate", action="store_true",
                        help="load gate_plan.json configuration for an inference-only diagnostic")
    parser.add_argument("--out", default=str(DEFAULT_OUT / "training"),
                        help="training directory containing gate_plan.json")
    parser.add_argument("--mining", help="override the gate configuration's mining.json path")
    parser.add_argument("--references-out", help="output directory (default: <out>/references)")
    parser.add_argument("--max-sources", type=int, default=24)
    parser.add_argument("--max-invariants", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true",
                        help="check the saved configuration and manifest without loading a model")
    cli = parser.parse_args()
    if not cli.from_gate:
        parser.error("standalone inference requires explicit --from-gate")
    if cli.max_sources < 0 or cli.max_invariants < 0 or not 0 <= cli.max_pairs <= 32:
        parser.error("counts must be nonnegative; --max-pairs must be at most 32")
    training_dir = Path(cli.out).resolve()
    saved = json.loads((training_dir / "gate_plan.json").read_text(encoding="utf-8"))
    config = dict(saved["config"])
    config["out"] = str(training_dir)
    if cli.mining:
        config["mining"] = str(Path(cli.mining).resolve())
    # Older saved plans precede the later carrier/relationship diagnostics.
    for name, value in dict(carrier="pool", relation_weight=0., relation_margin=1.,
                            prepare_teacher_kl=False, checkpointing=False).items():
        config.setdefault(name, value)
    args = SimpleNamespace(**config)
    mining = json.loads(Path(args.mining).read_text(encoding="utf-8"))
    if mining.get("stage") != "complete":
        raise ValueError("reference CLI requires a completed mining manifest")
    plan = make_plan(mining, args)
    references_dir = Path(cli.references_out).resolve() if cli.references_out else training_dir / "references"
    summary = dict(mode="inference only; no optimization", training_dir=str(training_dir),
                   references_out=str(references_dir), carrier=args.carrier,
                   heldout_pairs=min(cli.max_pairs, len(plan["heldout"])),
                   untouched_sources=min(cli.max_sources, len(plan["sources"])),
                   invariant_attempt_cap=cli.max_invariants)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if cli.dry_run:
        return
    experiment = Experiment(args, plan)
    result = run_references(experiment, mining, references_dir,
                            max_sources=cli.max_sources, max_invariants=cli.max_invariants,
                            max_pairs=cli.max_pairs)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
