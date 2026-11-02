"""Export completed full runs without selecting the largest score or using prefixes."""
from pathlib import Path
from decimal import Decimal
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMPARISON = ROOT / "docs/versions/v2/eadp_comparison_20261028/comparison.json"
NEXT_PLAN_SHA = "7de9048ad555683ca95d1bba27193c30c641b2decd5ae2d4626a359b11e9462d"
NEXT_WORKER_SHA = "466e3c7be72cf5db3a12409e50fcb47286f6d054af4fa6f6f0fad351a6d4e94d"
POPE_PLAN_SHA = "bf40e511c27cf920dbd287d61126c405346f97fc37728cff3254af608aef5a70"
MMB_BETA_PLAN_SHA = "f4a2eff7485313e8ee1bec7553c4897b60abb48ccd4beb34e165e20a09e49f40"
QWEN_PLAN_SHA = "d1afef9e1f470d432105cb8669e25f54b9906d85b54f7685689a84b539feb467"
QWEN_AUDITOR_SHA = "f7e1981caa5405296d8a413a0b43883887af0c541f8c69033ccb12544a3d0ae1"
QWEN_PREPARATION_SHA = "2a52c673886038022f28ed62d6a8f5a45da82aa9edd41ba6f91375aaf1f9d2b6"
QWEN_POLICY_SHA = "fe0bad1314831c344aac49617e6da0e338e9a4a6da5254a3de5a0cdea46764f0"
QWEN_COUNTS = {"DocVQA_VAL": 5349, "AI2D_TEST": 3088}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PendingRun(Exception):
    pass


class MetricDisagreement(Exception):
    def __init__(self, row):
        self.row = row
        super().__init__("Complete MMB run has different public exact_matching and frozen local first-letter scores; retain both for root review")


def require_complete(record, label):
    status = str(record.get("status", "")).lower()
    assert not any(word in status for word in ("fail", "error", "cancel", "abort")), label + " reports terminal failure: " + status
    if status != "complete":
        raise PendingRun(label + " is still finalizing: " + status)


def terminal_files(directory, required, failures=()):
    """Only terminal metadata is inspected until every required proof exists."""
    failed = [str(directory / name) for name in failures if (directory / name).exists()]
    assert not failed, "Preserved failure artifact: " + ", ".join(failed)
    for name in required:
        if name.endswith("state.json") and (directory / name).exists():
            try:
                state = read(directory / name)
            except json.JSONDecodeError as error:
                raise PendingRun("Terminal state JSON record is being written: " + str(directory / name)) from error
            status = str(state.get("status", "")).lower()
            assert not any(word in status for word in ("fail", "error", "cancel", "abort")), "Preserved terminal state: " + status
            assert state.get("exit_code") in (None, 0) and state.get("generation_exit_code") in (None, 0), "Terminal state records a nonzero exit"
    missing = [str(directory / name) for name in required if not (directory / name).exists()]
    if missing:
        raise PendingRun("Missing terminal proof: " + ", ".join(missing))
    try:
        return {name: read(directory / name) for name in required}
    except json.JSONDecodeError as error:
        raise PendingRun("Terminal JSON record is being written") from error


def artifact(record, expected_path=None):
    path = Path(record["path"])
    if not path.exists():
        raise PendingRun("Required proof artifact is missing: " + str(path))
    if expected_path is not None:
        assert path.resolve() == expected_path.resolve(), "Artifact path differs from the registered run"
    assert sha(path) == record["sha256"], "Artifact SHA mismatch: " + str(path)
    return path


def raw_rows(path, n, expected_sha, require_newline=True):
    if not path.exists():
        raise PendingRun("Required complete raw artifact is missing: " + str(path))
    raw = path.read_bytes()
    assert not require_newline or raw.endswith(b"\n"), "Incomplete final raw record"
    assert hashlib.sha256(raw).hexdigest() == expected_sha
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == n, "Full raw denominator required"
    return rows


def stat_record(record):
    path = Path(record["path"])
    current = path.stat()
    assert current.st_size == record["bytes"] and current.st_mtime_ns == record["mtime_ns"], str(path)


def exit_proof(pid, ticks):
    current = None
    try:
        fields = Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()
        current = int(fields[19])
    except (FileNotFoundError, ProcessLookupError):
        pass
    if current == ticks:
        raise PendingRun("Waiting for actual original worker exit")
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise PendingRun("Actual GPU exit evidence is temporarily unavailable") from error
    if pid in {int(line.strip()) for line in gpu.splitlines() if line.strip()}:
        raise PendingRun("Waiting for original PID to leave the GPU process list")
    return {"pid": pid, "start_ticks": ticks, "original_worker_exited": True,
            "current_start_ticks_if_PID_reused": current, "gpu_PID_absent": True}


def new_row(run_id, model, task, budget, cli_budget, n, alpha, beta, value, paper,
            predictions, proof_path, metric, author_value, independent_value, provenance, stability, allow_author_disagreement=False):
    assert math.isfinite(value) and math.isfinite(author_value) and math.isfinite(independent_value)
    assert abs(value - independent_value) < 1e-10, "Primary and independent metric disagree"
    assert allow_author_disagreement or abs(value - author_value) < 1e-10, "Primary and author metric disagree"
    return {"run_id": run_id, "model": model, "task": task, "token_budget": budget,
            "budget_cli": cli_budget, "alpha": alpha, "beta": beta, "three_wait_nodes": True,
            "n": n, "n_circular_groups": 1176 if task == "MMB_EN" else None,
            "value": value, "paper": paper,
            "local_minus_paper": str(Decimal(str(value)) - Decimal(str(paper))),
            "unit": "percentage_points", "metric": metric,
            "source_recipe_identity": "Registered single-setting full run; paper complete argv remains unverified",
            "predictions": str(predictions), "prediction_sha256": sha(predictions),
            "verification": str(proof_path), "verification_sha256": sha(proof_path),
            "author_value": author_value, "independent_value": independent_value,
            "author_minus_independent": str(Decimal(str(author_value)) - Decimal(str(independent_value))),
            "author_independent_metric_agreement": abs(author_value - independent_value) < 1e-10,
            "complete_provenance": provenance, "stability_boundary": stability}


def next_full_row(paper):
    directory = HERE / "next_textvqa/full_default_K128_alpha05_beta1_wait"
    third_path = HERE / "pope/run03_single_alpha05_beta1_wait_k32/predecessor_verification.json"
    meta = terminal_files(directory, ["state.json", "official.score.json", "release.json"],
                          ["failure.json", "release_supervisor_failure.json"])
    if not third_path.exists():
        raise PendingRun("Missing third-party full5000 audit: " + str(third_path))
    state, score, release = (meta[name] for name in ["state.json", "official.score.json", "release.json"])
    require_complete(state, "NeXT native state")
    assert state["n"] == 5000
    assert score["n"] == score["token_decoding_checks"] == 5000
    assert release["n_rows"] == release["token_decoding_checks"] == 5000
    assert release["original_worker_exited"] and release["gpu_pid_absent"] and release["all_frozen_guards_passed"]
    assert release["official_score_sha256"] == state["score_sha256"] == sha(directory / "official.score.json")
    assert release["native_state_sha256"] == sha(directory / "state.json")
    supervisor_path = HERE / "next_textvqa/full_default_K128.supervisor_state.json"
    assert release["supervisor_sha256"] == sha(supervisor_path)
    supervisor = read(supervisor_path)
    require_complete(supervisor, "NeXT supervisor")
    assert supervisor["exit_code"] == 0 and supervisor["actual_worker_exited"]
    assert supervisor["worker"] == release["worker"] and state["pid"] == release["worker"]["pid"]
    plan_path = HERE / "next_textvqa/full_default_K128.plan.json"
    assert sha(plan_path) == release["plan_sha256"] == score["plan"]["sha256"] == NEXT_PLAN_SHA
    plan = read(plan_path)
    assert plan["n"] == 5000 and plan["parameters"]["visual_token_num"] == 128 and plan["parameters"]["budget_nominal"] == 640
    assert (plan["parameters"]["alpha"], plan["parameters"]["beta"]) == (.5, 1.)
    assert plan["worker"]["sha256"] == release["worker_sha256"] == NEXT_WORKER_SHA
    for record in [plan["worker"], plan["wait_helper"], plan["questions"], plan["annotation"], *plan["model_small_files"], *plan["evidence"], plan["clip_identity"]["audit"]]:
        artifact(record)
    for record in plan["source_comparisons"]:
        assert sha(Path(record["author_path"])) == record["author_sha256"]
    for record in [*plan["model_payloads"], *plan["clip_identity"]["files"], *read(HERE / "next_textvqa/full_default_K128.images.json")]:
        stat_record(record)
    third = read(third_path)
    assert third["success"] and third["n_rows"] == third["token_decoding_checks"] == third["per_question_score_checks"] == 5000
    assert third["plan_sha256"] == NEXT_PLAN_SHA and third["worker_sha256"] == NEXT_WORKER_SHA
    assert third["all_frozen_source_input_weight_image_guards_verified"]
    assert third["official_score_sha256"] == sha(directory / "official.score.json")
    assert third["predecessor_state_sha256"] == sha(directory / "state.json")
    assert third["predecessor_supervisor_sha256"] == sha(supervisor_path)
    artifact(third["verifier"])
    third_exit = third["worker_exit"]
    assert (third_exit["pid"], third_exit["startticks"]) == (release["worker"]["pid"], release["worker"]["startticks"])
    assert third_exit["original_worker_exited"] and third_exit["gpu_pid_absent"]
    for name, record in score["artifacts"].items():
        artifact(record, directory / name)
        assert third["artifacts"][name] == record
    for record in third["artifacts"].values():
        artifact(record)
    assert sha(directory / "per_question_scores.jsonl") == score["per_question_scores_sha256"]
    assert abs(score["value"] - third["official_accuracy_percent"]) < 1e-10
    assert abs(score["value"] - third["independent_accuracy_percent"]) < 1e-10
    worker_exit = exit_proof(release["worker"]["pid"], release["worker"]["startticks"])
    predictions = directory / "answers.jsonl"
    answers = raw_rows(predictions, 5000, score["artifacts"]["answers.jsonl"]["sha256"])
    questions = raw_rows(Path(plan["questions"]["path"]), 5000, plan["questions"]["sha256"], require_newline=False)
    assert [(str(a["question_id"]), a["prompt"]) for a in answers] == [(str(q["question_id"]), q["text"]) for q in questions]
    raw_rows(directory / "runtime.jsonl", 5000, score["artifacts"]["runtime.jsonl"]["sha256"])
    question_scores = raw_rows(directory / "per_question_scores.jsonl", 5000, score["per_question_scores_sha256"])
    for position, (question, scored) in enumerate(zip(questions, question_scores)):
        assert scored["position"] == position and str(scored["question_id"]) == str(question["question_id"])
        assert scored["prompt_sha256"] == hashlib.sha256(question["text"].encode()).hexdigest()
        assert math.isfinite(scored["score"]) and 0 <= scored["score"] <= 1
    assert abs(100 * sum(row["score"] for row in question_scores) / 5000 - score["value"]) < 1e-10
    assert abs(release["official_accuracy_percent"] - score["value"]) < 1e-10
    return new_row("NeXT_public_default_K128_alpha05_beta1_wait", "LLaVA-NeXT-7B", "TextVQA", 640, 128, 5000,
                   .5, 1., score["value"], paper("next", "TextVQA", 640), predictions, third_path,
                   "official TextVQA accuracy", score["value"], third["independent_accuracy_percent"],
                   {"plan": {"path": str(plan_path), "sha256": sha(plan_path)},
                    "author_score": {"path": str(directory / "official.score.json"), "sha256": sha(directory / "official.score.json")},
                    "release": {"path": str(directory / "release.json"), "sha256": sha(directory / "release.json")},
                    "third_party_audit": {"path": str(third_path), "sha256": sha(third_path)},
                    "source_hashes_rechecked": len(plan["source_comparisons"]), "worker_exit": worker_exit},
                   "Full5000 checked; no repeated-input conflict audit is claimed")


def pope_alpha_full_row(paper):
    directory = HERE / "pope/run03_single_alpha05_beta1_wait_k32"
    meta = terminal_files(directory, ["state.json", "full_result_summary.json", "completion_and_release_proof.json", "process_identity.json", "supervisor_state.json"], ["supervisor_failure.json"])
    state, score, release, process, supervisor = (meta[name] for name in ["state.json", "full_result_summary.json", "completion_and_release_proof.json", "process_identity.json", "supervisor_state.json"])
    require_complete(state, "POPE native state")
    require_complete(supervisor, "POPE supervisor")
    assert supervisor["exit_code"] == 0
    assert state["generation_exit_code"] == state["score_exit_code"] == score["official_scorer_exit_code"] == 0
    assert state["n_predictions"] == score["n_predictions"] == release["prediction_count"] == 8910
    assert score["complete"] and score["all_question_ids_order_verified"] and score["all_raw_prompts_verified"] and score["all_pinned_image_question_positions_verified"]
    assert score["gpu_worker_exit_verified"] and score["frozen_source_inputs_payload_stats_verified"] and score["actual_runtime_verified"]
    plan_path = directory / "run_plan.json"
    assert sha(plan_path) == state["plan_sha256"] == process["plan_sha256"] == score["run_plan_sha256"] == release["run_plan_sha256"] == POPE_PLAN_SHA
    plan = read(plan_path)
    assert (plan["parameters"]["alpha"], plan["parameters"]["beta"], plan["parameters"]["visual_token_num"]) == (.5, 1., 32)
    assert (score["actual_alpha"], score["actual_beta"], score["actual_visual_token_num"]) == (.5, 1., 32)
    for name, expected in plan["source_files"].items():
        assert sha(Path(plan["source_root"]) / name) == expected
    for group in ["small_file_guards", "implementation_sha256", "model_small_files"]:
        for name, expected in plan[group].items():
            assert sha(Path(name)) == expected
    for group in ["model_stat_guards", "image_stat_guards"]:
        for name, guard in plan[group].items():
            path = Path(name); stat = path.stat()
            current = {"resolved": str(path.resolve()), "bytes": stat.st_size, "device": stat.st_dev, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
            assert current == {k: v for k, v in guard.items() if k != "prior_verified_sha256"}
    assert release["status"] == "all_completion_and_exit_gates_passed" and release["GPU_slot"] == "B"
    assert release["gpu_pid_absent"] and not release["same_worker_still_alive"] and release["all_source_sha_match"] and release["all_frozen_source_input_weight_image_guards_verified"]
    assert release["independent_and_official_macro_match"] and release["generation_exit_code"] == release["official_score_exit_code"] == 0
    artifact(release["full_result_summary"], directory / "full_result_summary.json")
    assert release["state_sha256"] == sha(directory / "state.json") and release["process_identity_sha256"] == sha(directory / "process_identity.json")
    assert release["official_score_log_sha256"] == score["score_log_sha256"] == sha(directory / "official_score.log")
    assert supervisor["actual_worker_exited"] and supervisor["full_result_summary_sha256"] == sha(directory / "full_result_summary.json")
    assert supervisor["completion_and_release_proof_sha256"] == sha(directory / "completion_and_release_proof.json")
    third_path = artifact(release["independent_prediction_verification"], directory / "supervisor_prediction_verification/full_prediction_verification.json")
    third = read(third_path)
    assert third["complete"] and third["n_predictions"] == 8910 and third["all_question_ids_order_verified"] and third["all_raw_prompts_verified"]
    assert third["predictions_sha256"] == release["predictions_sha256"] == score["predictions_sha256"]
    assert abs(score["official_macro_f1_percent"] - score["independent_macro_f1_percent"]) < 1e-10
    assert abs(score["official_macro_f1_percent"] - third["independent_macro_f1_percent"]) < 1e-10
    assert abs(score["official_macro_f1_percent"] - third["official_macro_f1_percent"]) < 1e-10
    assert state["worker_pid"] == process["worker_pid"] == release["worker_pid"] and process["worker_start_ticks"] == release["worker_start_ticks"]
    worker_exit = exit_proof(process["worker_pid"], process["worker_start_ticks"])
    predictions = directory / "predictions.jsonl"
    answers = raw_rows(predictions, 8910, score["predictions_sha256"])
    questions = raw_rows(Path(plan["generation_command"][plan["generation_command"].index("--question-file") + 1]), 8910, score["canonical_questions_sha256"], require_newline=False)
    assert len({a["question_id"] for a in answers}) == 8910
    assert [(a["question_id"], a["prompt"]) for a in answers] == [(q["question_id"], q["text"]) for q in questions]
    author_log_value = float(re.search(r"Average F1 score:\s*([0-9.eE+-]+)", (directory / "official_score.log").read_text()).group(1)) * 100
    assert abs(author_log_value - score["official_macro_f1_percent"]) < 1e-10
    def positive(text):
        words = text.split(".")[0].replace(",", "").split(" ")
        return not any(word in words for word in ("No", "not", "no"))
    label_directory = Path(plan["score_command"][plan["score_command"].index("--annotation-dir") + 1])
    category_f1, repeats = [], defaultdict(list)
    for answer, question in zip(answers, questions):
        assert answer["model_id"] == "llava-v1.5-7b" and isinstance(answer["text"], str) and not answer["text"].startswith("FAILED")
        repeats[(question["image"], question["text"])].append(positive(answer["text"]))
    for category, expected_n in [("random", 2910), ("popular", 3000), ("adversarial", 3000)]:
        selected = [(a, q) for a, q in zip(answers, questions) if q["category"] == category]
        labels = [json.loads(line) for line in (label_directory / ("coco_pope_" + category + ".json")).read_text().splitlines()]
        assert len(selected) == len(labels) == expected_n
        for (_, question), label in zip(selected, labels):
            assert question["image"] == label["image"] and question["text"].replace("\nAnswer the question using a single word or phrase.", "") == label["text"]
        counts = Counter((positive(a["text"]), label["label"] != "no") for (a, _), label in zip(selected, labels))
        tp, fp, tn, fn = [counts[key] for key in [(True, True), (True, False), (False, False), (False, True)]]
        f1 = 200 * tp / (2 * tp + fp + fn)
        assert [score["categories"][category][key] for key in ["TP", "FP", "TN", "FN"]] == [tp, fp, tn, fn]
        assert abs(f1 - score["categories"][category]["f1_percent"]) < 1e-10
        category_f1.append(f1)
    assert abs(sum(category_f1) / 3 - third["independent_macro_f1_percent"]) < 1e-10
    repeated = [values for values in repeats.values() if len(values) > 1]
    assert score["same_input_repeated_groups"] == len(repeated)
    assert score["same_input_yes_no_conflict_groups"] == sum(len(set(values)) > 1 for values in repeated)
    return new_row("POPE_run03_single_alpha05_beta1_wait_k32", "LLaVA-1.5-7B", "POPE", 32, 32, 8910,
                   .5, 1., score["official_macro_f1_percent"], paper("v15", "POPE", 32), predictions, directory / "full_result_summary.json",
                   "three-category macro F1", score["official_macro_f1_percent"], third["independent_macro_f1_percent"],
                   {"plan": {"path": str(plan_path), "sha256": sha(plan_path)}, "release": {"path": str(directory / "completion_and_release_proof.json"), "sha256": sha(directory / "completion_and_release_proof.json")},
                    "third_party_audit": {"path": str(third_path), "sha256": sha(third_path)}, "worker_exit": worker_exit},
                   str(score["same_input_yes_no_conflict_groups"]) + "/" + str(score["same_input_repeated_groups"]) + " repeated-input conflicts")


def mmb_beta_full_row(paper):
    directory = HERE / "mmbench/beta2_control"
    meta = terminal_files(directory, ["state.json", "independent_verification.json", "local_circular_score.json", "conversion.json", "after_pope_alpha.state.json"], ["after_pope_alpha.failure.json"])
    state, proof, local, conversion, waiter = (meta[name] for name in ["state.json", "independent_verification.json", "local_circular_score.json", "conversion.json", "after_pope_alpha.state.json"])
    require_complete(state, "MMB beta2 native state")
    require_complete(waiter, "MMB beta2 waiter")
    assert state["exit_code"] == waiter["exit_code"] == 0
    assert waiter["actual_native_exited"] and waiter["independent_verification_sha256"] == sha(directory / "independent_verification.json")
    assert proof["success"] and proof["all_rows_checked"] and proof["n_rows"] == local["n_rows"] == 4377
    assert proof["n_circular_groups"] == local["n_groups"] == 1176 and proof["current_beta"] == 2
    assert abs(proof["circular_accuracy_percent"] - local["value"]) < 1e-12
    assert proof["circular_correct"] == local["circular_correct"]
    assert state["score_sha256"] == sha(directory / "local_circular_score.json") and state["conversion_record_sha256"] == sha(directory / "conversion.json")
    assert conversion["exit_code"] == 0 and not conversion["submitted_to_external_server"] and sha(Path(conversion["output"])) == proof["submission_sha256"]
    plan_path = directory / "protocol.json"
    assert sha(plan_path) == state["protocol_sha256"] == MMB_BETA_PLAN_SHA
    plan = read(plan_path)
    assert (plan["parameters"]["K"], plan["parameters"]["alpha"], plan["parameters"]["beta"]) == (128, .5, 2)
    for record in plan["source_files"]:
        artifact(record)
    assert sha(Path(plan["dataset"]["path"])) == proof["dataset_sha256"] == plan["dataset"]["sha256"]
    loaded = read(directory / "observed/loaded_model.json")
    assert (loaded["visual_token_num"], loaded["alpha"], loaded["beta"]) == (128, .5, 2.)
    assert loaded["native_argv"] == plan["command"][3:]
    patch = proof["controlled_stream_patch"]
    assert patch["applied_in_this_process"] and patch["restored_ast_identical"] and patch["exact_added_statement_count"] == 3
    assert patch["author_source_after_patch_sha256"] == patch["source_sha256"] == plan["runtime_patch"]["source_sha256"]
    native = state["native_process"]
    worker_exit = exit_proof(native["pid"], native["start_ticks"])
    predictions = Path(plan["command"][plan["command"].index("--answers-file") + 1])
    assert predictions.resolve() == (directory / "v15_EADP_K128_oldEN_beta2_alpha05_streamwait.jsonl").resolve()
    assert state["prediction_sha256"] == local["prediction_sha256"] == proof["prediction_sha256"]
    answers = raw_rows(predictions, 4377, proof["prediction_sha256"])
    raw_rows(directory / "observed/runtime.jsonl", 4377, proof["runtime_sha256"])
    # Reuse the root-owned checker's unchanged public helpers only after all
    # complete-run/exit gates. No running-prefix data or external API is used.
    checker_path = HERE / "mmbench/public_mcq_metric_check.py"
    spec = importlib.util.spec_from_file_location("ledger_public_mcq", checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    import copy
    import os
    import re
    import string
    import pandas as pd
    namespace = dict(cp=copy, os=type("ReadonlyEnv", (), {"environ": {}}), re=re, string=string, pd=pd)
    sources = [checker.original_helpers(checker.UTILS / "utils/matching_util.py", ["can_infer_option", "can_infer_text", "can_infer"], namespace),
               checker.original_helpers(checker.UTILS / "dataset/utils/multiple_choice.py", ["build_choices", "prefetch_answer", "prefetch_circular_group"], namespace)]
    frozen_public = read(HERE / "mmbench/public_mcq_metric_check.json")
    assert [r["sha256"] for r in sources] == [r["sha256"] for r in frozen_public["unchanged_public_functions"]]
    data = pd.read_table(plan["dataset"]["path"])
    assert len(data) == 4377 and data["index"].tolist() == [int(a["question_id"]) for a in answers]
    assert len({int(a["question_id"]) for a in answers}) == 4377
    data["prediction"] = [a["text"] for a in answers]
    data["GT"] = data["answer"]
    data["g_index"] = [int(i % 1000000) for i in data["index"]]
    groups = [namespace["prefetch_circular_group"](group) for _, group in data.groupby("g_index", sort=True)]
    assert len(groups) == 1176
    hits = [0 if group is None else group["hit"] for group in groups]
    assert all(hit in (0, 1) for hit in hits)
    author_value = 100 * sum(hits) / 1176
    parser_changes = []
    for _, item in data.iterrows():
        public = namespace["prefetch_answer"](item)
        letters = re.findall(r"\b[A-E]\b", item["prediction"].strip())
        local_letter = letters[0] if letters else None
        if public != local_letter:
            parser_changes.append({"index": int(item["index"]), "public": public, "local": local_letter})
    result = new_row("MMB_single_beta2_alpha05_streamwait", "LLaVA-1.5-7B", "MMB_EN", 128, 128, 4377,
                   .5, 2., proof["circular_accuracy_percent"], paper("v15", "MMB_EN", 128), predictions, directory / "independent_verification.json",
                   "exact-letter circular accuracy; public exact_matching rechecked", author_value, proof["circular_accuracy_percent"],
                   {"plan": {"path": str(plan_path), "sha256": sha(plan_path)},
                    "author_public_exact_matching": {"value_percent": author_value, "n_rows": 4377, "n_groups": 1176,
                                                     "unresolved_groups": sum(group is None for group in groups), "parser_changed_rows": parser_changes,
                                                     "sources": sources, "external_server_evaluation": False},
                    "waiter": {"path": str(directory / "after_pope_alpha.state.json"), "sha256": sha(directory / "after_pope_alpha.state.json")},
                    "local_score": {"path": str(directory / "local_circular_score.json"), "sha256": sha(directory / "local_circular_score.json")}, "worker_exit": worker_exit},
                   "Repeated-input conflicts not audited for this arm; no zero-conflict value is assumed", allow_author_disagreement=True)
    if not result["author_independent_metric_agreement"]:
        raise MetricDisagreement(result)
    return result


def qwen_terminal_metadata():
    """Require the whole public sequence's terminal proofs before reading raw rows."""
    directory = HERE / "qwen/independent_audit"
    prefix = "public_fa2_full_run02.terminal_audit"
    names = [prefix + suffix for suffix in (".state.json", ".report.json", ".model_stat_guards.json", ".claim.json", ".exec.json")]
    meta = terminal_files(directory, names, [prefix + ".failure.json"])
    state, report, model_guards, claim, execution = (meta[name] for name in names)
    require_complete(state, "Qwen whole-sequence terminal audit")
    report_path = directory / (prefix + ".report.json")
    assert state["audit_terminated"] and state["report_sha256"] == sha(report_path)
    assert report["audit_terminated"] and report["CPU_only"] and not report["GPU_or_API_used"]
    assert not report["generation_or_source_modified"] and not report["generation_retry_performed"]
    assert not state["paper_reproduction_success_claimed"] and not report["paper_reproduction_success_claimed"]
    assert report["completion_is_not_paper_reproduction_success"]
    assert state["full_prediction_audit_passed"] and report["full_prediction_audit_passed"], "Complete terminal audit records a failed full-prediction check"
    assert report["coverage_complete"] and report["generation_valid"], "Full coverage and valid model calls are required separately"
    assert report["existing_CPU_verifier_exit_code"] == 0 and report["all_native_stages_observed"]
    assert not report["errors"], "Preserved terminal audit errors: " + repr(report["errors"])
    plan_path = HERE / "qwen/public_fa2_full_run02_plan.json"
    preparation_path = directory / "terminal_audit_preparation.json"
    policy_path = directory / "empty_answer_policy.json"
    assert sha(plan_path) == report["generation_plan_sha256"] == QWEN_PLAN_SHA
    assert sha(preparation_path) == report["audit_preparation_sha256"] == claim["preparation_sha256"] == QWEN_PREPARATION_SHA
    assert sha(policy_path) == report["empty_answer_policy_sha256"] == QWEN_POLICY_SHA
    preparation, plan = read(preparation_path), read(plan_path)
    assert preparation["generation_plan_sha256"] == QWEN_PLAN_SHA and preparation["empty_policy_sha256"] == QWEN_POLICY_SHA
    assert state["identity"] == claim["identity"] == execution["identity"] and not claim["automatic_retry"]
    assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == preparation["boot_id"]
    for path, expected in preparation["frozen_files"].items():
        assert sha(Path(path)) == expected, "Frozen Qwen audit/source changed: " + path
    assert preparation["frozen_files"][str(directory / "audit_qwen.py")] == QWEN_AUDITOR_SHA
    assert plan["author_commit"] == "e1a08801461c181871b9ff1cb0803b8e9966f083"
    assert (plan["recipe"]["visual_token_num"], plan["recipe"]["alpha"], plan["recipe"]["beta"]) == (256, .5, 2.)
    for group in ("source_sha256", "frozen_evidence"):
        for path, expected in plan[group].items():
            assert sha(Path(path)) == expected, "Frozen Qwen generation input/source changed: " + path
    expected_guards = preparation["prior_SHA_and_current_model_stat_guards"]
    assert len(model_guards) == len(expected_guards)
    for original, guarded in zip(expected_guards, model_guards):
        assert guarded["path"] == original["path"] and guarded["current_stat_matches"]
        assert guarded["prior_verified_sha256"] == original["expected_sha256"] and guarded["kind"] == original["kind"]
        stat_record(original["stat"])
    gate = report["terminal_process_proof"]
    assert gate["reason"] == "whole_public_sequence_and_actual_processes_terminated"
    assert gate["controller_original_exited"] and gate["docker_client_original_exited"] and gate["container_init_original_exited"]
    for field, original in (("original_controller", "root_controller"), ("original_docker_client", "docker_client"), ("original_container_init", "container_init")):
        assert gate[field] == preparation[original]
    container = gate["container"]
    if container["exists"]:
        assert container["id"] == preparation["container_id"] and container["image"] == preparation["container_image"] and not container["running"]
    else:
        assert container["container_id"] == preparation["container_id"]
    run_directory = HERE / "qwen/public_fa2_full_run02"
    launcher_path = run_directory / "state.json"
    assert sha(launcher_path) == gate["launcher_state_sha256"] and read(launcher_path) == gate["launcher_state"]
    launcher = gate["launcher_state"]
    assert launcher["status"] == "generation_launcher_exited_validation_pending" and launcher["plan_sha256"] == QWEN_PLAN_SHA
    assert launcher["controller_pid"] == preparation["root_controller"]["pid"] and launcher["docker_client_pid"] == preparation["docker_client"]["pid"]
    assert type(launcher["launcher_exit_code"]) is int
    assert report["public_launcher_exit_code"] == launcher["launcher_exit_code"]
    assert report["public_launcher_exited_successfully"] == (launcher["launcher_exit_code"] == 0)
    proof_workers, observed = gate["worker_exits"], report["observed_native_workers"]
    assert {row["task"] for row in observed} == set(QWEN_COUNTS)
    assert len(proof_workers) == len(observed) and len({(row["identity"]["pid"], row["identity"]["startticks"]) for row in observed}) == len(observed)
    actual_exits = []
    for original in (preparation["root_controller"], preparation["docker_client"], preparation["container_init"]):
        actual_exits.append(exit_proof(original["pid"], original["startticks"]))
    python = plan["command"][plan["command"].index("--entrypoint") + 1]
    entry = next(path for path in plan["source_sha256"] if path.endswith("/Qwen_vl/VLMEvalKit/run.py"))
    for worker, proof in zip(observed, proof_workers):
        assert worker["identity"] == proof["original_identity"] and worker["task"] == proof["task"]
        assert proof["original_exited"] and proof["GPU_pid_absent"]
        assert worker["cmdline"] == [python, entry, "--model", plan["model_name"], "--data", worker["task"], "--work-dir", str(Path(plan["environment"]["OUTPUT_ROOT"]) / "eadp"), "--verbose"]
        groups = [line.split(":", 2)[2].split("/") for line in worker["cgroup"].splitlines() if line.strip()]
        assert groups and all(preparation["container_id"] in group or "docker-" + preparation["container_id"] + ".scope" in group for group in groups)
        actual_exits.append(exit_proof(worker["identity"]["pid"], worker["identity"]["startticks"]))
    verification_path = directory / "public_fa2_full_run02.verification.json"
    assert verification_path.exists(), "Terminal audit refers to a missing complete verification file"
    verification = read(verification_path)
    assert verification == report["existing_CPU_verification"]
    assert verification["complete"] and verification["coverage_complete"] and verification["generation_valid"]
    assert not verification["pending_datasets"] and set(verification["datasets"]) == set(QWEN_COUNTS)
    assert verification["full_generation_without_error_placeholders"]
    assert verification["plan_sha256"] == QWEN_PLAN_SHA and verification["launcher_state"] == launcher
    assert verification["auditor"]["sha256"] == QWEN_AUDITOR_SHA
    artifact(verification["auditor"], directory / "audit_qwen.py")
    artifact(verification["audit_empty_policy"], policy_path)
    assert not verification["generated_by_this_audit"] and not verification["GPU_loaded_by_this_audit"]
    assert verification["public_launcher_exited_successfully"] == report["public_launcher_exited_successfully"]
    assert verification["author_score_artifacts_complete"] == report["author_score_artifacts_complete"]
    assert verification["empty_count"] == report["empty_count"]
    assert verification["planned_no_empty_gate_passed"] == report["planned_no_empty_gate_passed"]
    for task, count in QWEN_COUNTS.items():
        diagnostic = report["raw_dataset_diagnostics"][task]
        validity = verification["datasets"][task]["prediction_validity"]
        assert diagnostic["expected_count"] == diagnostic["observed_count"] == validity["n"] == count
        assert diagnostic["coverage_complete"] and diagnostic["generation_valid"] and validity["coverage_complete"] and validity["generation_valid"]
        assert diagnostic["empty_count"] == validity["empty_count"] and diagnostic["planned_no_empty_gate_passed"] == validity["planned_no_empty_gate_passed"]
    assert verification["empty_count"] == sum(row["prediction_validity"]["empty_count"] for row in verification["datasets"].values())
    assert verification["planned_no_empty_gate_passed"] == all(row["prediction_validity"]["planned_no_empty_gate_passed"] for row in verification["datasets"].values())
    audited_inputs = read(directory / "public_fa2_full_run02_plan.preparation_audit.json")
    assert audited_inputs["plan"]["sha256"] == QWEN_PLAN_SHA
    for path, expected in audited_inputs["metric_source_sha256"].items():
        assert sha(Path(path)) == expected, "Frozen Qwen metric source changed: " + path
    image_manifest_path = artifact(audited_inputs["current_image_checks"]["manifest"], directory / "current_cached_images.json")
    images = read(image_manifest_path)
    assert len(images) == audited_inputs["current_image_checks"]["n"]
    for image in images:
        stat_record(image["stat"])
    for record in report["native_logs"]:
        path = artifact(record)
        assert path.is_relative_to(run_directory) and path.stat().st_size == record["bytes"]
    return directory, run_directory, plan, report, report_path, verification, verification_path, actual_exits


def qwen_full_row(paper, task):
    directory, run_directory, plan, terminal, terminal_path, verification, verification_path, exits = qwen_terminal_metadata()
    count = QWEN_COUNTS[task]
    score = verification["datasets"][task]
    assert score["n"] == score["pointwise_author_reference_matches"] == count
    predictions = artifact(score["raw_predictions"])
    assert predictions.is_relative_to(run_directory)
    assert predictions.stem == plan["model_name"] + "_" + task
    diagnostic = terminal["raw_dataset_diagnostics"][task]
    assert Path(diagnostic["raw_path"]).resolve() == predictions.resolve() and diagnostic["raw_sha256"] == sha(predictions)
    per_question_path = artifact(score["per_question"], directory / (run_directory.name + "_" + task + ".per_question.jsonl"))
    manifest = read(directory / (task + ".canonical_manifest.json"))
    assert manifest["n"] == count and manifest["source_sha256"] == plan["datasets"][task]["data_sha256"]
    assert manifest["source_stat"]["path"] == plan["datasets"][task]["data_path"]
    stat_record(manifest["source_stat"])
    canonical = raw_rows(directory / (task + ".canonical.jsonl"), count, manifest["canonical_sha256"])
    # Import only the frozen standard-library reader and coverage validator, never verify().
    spec = importlib.util.spec_from_file_location("ledger_qwen_raw_reader", directory / "audit_qwen.py")
    reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
    answers = reader.read_table(predictions)
    validity = reader.validate_predictions(canonical, answers, count)
    assert validity == score["prediction_validity"] and validity["coverage_complete"] and validity["generation_valid"]
    per_question = raw_rows(per_question_path, count, score["per_question"]["sha256"])
    for question, answer, scored in zip(canonical, answers, per_question):
        assert str(question["index"]) == str(scored["index"])
        assert answer["prediction"] == scored["raw_prediction"]
        assert reader.scoring_prediction(answer["prediction"]) == scored["scored_prediction"]
        assert reader.prediction_class(answer["prediction"]) == scored["prediction_class"]
        assert scored["prediction_changed_by_native_na_loader"] == (scored["raw_prediction"] != scored["scored_prediction"])
        assert math.isfinite(scored["author_score"]) and 0 <= scored["author_score"] <= 1
        assert math.isfinite(scored["independent_score"]) and 0 <= scored["independent_score"] <= 1
        assert abs(scored["author_score"] - scored["independent_score"]) < 1e-12
    value = 100 * sum(row["independent_score"] for row in per_question) / count
    author = 100 * sum(row["author_score"] for row in per_question) / count
    assert math.isfinite(value) and abs(value - score["value_percent"]) < 1e-10 and abs(author - value) < 1e-10
    assert score["native_na_prediction_conversions"] == sum(row["prediction_changed_by_native_na_loader"] for row in per_question)
    empty_zero = 100 * sum(0. if row["prediction_class"] == "empty_model_answer" else row["independent_score"] for row in per_question) / count
    assert abs(empty_zero - score["auxiliary_empty_answers_zero_score_percent"]) < 1e-10
    native_values = []
    for native in score["author_score_artifacts"]:
        path = artifact(native)
        assert path.is_relative_to(predictions.parent) and path.stem == predictions.stem + "_acc"
        assert math.isfinite(native["value_percent"]) and abs(native["value_percent"] - value) < 1e-7
        assert abs(native["value_percent"] - native["raw_value"] * (1 if task == "DocVQA_VAL" else 100)) < 1e-10
        native_values.append(native["value_percent"])
    paper_task = "DocVQA" if task == "DocVQA_VAL" else "AI2D"
    paper_value = paper("qwen", paper_task, 256)
    metric = "Released ANLS with native pandas NA conversion (%)" if task == "DocVQA_VAL" else "Released deterministic exact-matching can_infer accuracy (%); CPU, no API judge"
    row = {"run_id": "Qwen_author_FA2_" + task, "model": "Qwen3-VL-8B", "task": paper_task,
            "token_budget": 256, "budget_cli": 256, "alpha": .5, "beta": 2.,
            "three_wait_nodes": False, "three_wait_nodes_applicable": False, "attention_backend": "flash_attention_2",
            "n": count, "n_circular_groups": None, "value": value, "paper": paper_value,
            "local_minus_paper": str(Decimal(str(value)) - Decimal(str(paper_value))),
            "unit": "percentage_points", "metric": metric,
            "source_recipe_identity": "Full public author FA2 recipe with isolated missing-import helper; paper complete argv remains unverified",
            "predictions": str(predictions), "prediction_sha256": sha(predictions),
            "verification": str(verification_path), "verification_sha256": sha(verification_path),
            "author_value": author, "independent_value": value,
            "author_value_provenance": "Unchanged author metric helpers executed on all saved raw rows by the CPU auditor; native score artifacts are reported separately",
            "author_minus_independent": str(Decimal(str(author)) - Decimal(str(value))),
            "author_independent_metric_agreement": abs(author - value) < 1e-10,
            "native_author_score_artifacts": score["author_score_artifacts"], "native_author_value": native_values[0] if native_values else None,
            "native_author_score_status": score["author_score_status"],
            "native_public_launcher_exit_code": terminal["public_launcher_exit_code"],
            "native_public_launcher_exited_successfully": terminal["public_launcher_exited_successfully"],
            "native_author_score_artifacts_complete_for_sequence": terminal["author_score_artifacts_complete"],
            "API_judge_used_by_CPU_audit": False, "API_judge_reproduction_claimed": False,
            "prediction_validity": validity, "original_planned_no_empty_gate_passed": validity["planned_no_empty_gate_passed"],
            "empty_answer_policy": verification["audit_empty_policy"],
            "auxiliary_empty_answers_zero_score_percent": empty_zero,
            "native_na_prediction_conversions": score["native_na_prediction_conversions"],
            "complete_provenance": {"plan": {"path": str(HERE / "qwen/public_fa2_full_run02_plan.json"), "sha256": QWEN_PLAN_SHA},
                                    "terminal_audit": {"path": str(terminal_path), "sha256": sha(terminal_path)},
                                    "per_question": score["per_question"], "actual_process_exits": exits,
                                    "native_error_lines": terminal["native_error_lines"], "native_logs": terminal["native_logs"]},
            "stability_boundary": "Both tasks fully audited only after the whole native sequence exited; native API/scoring failures and the original No-empty gate are retained separately. No repeated-input conflict audit or paper/API-judge reproduction success is claimed."}
    if task == "DocVQA_VAL":
        literal = 100 * sum(item["literal_prediction_score"] for item in per_question) / count
        assert abs(literal - score["auxiliary_literal_prediction_ANLS_percent"]) < 1e-10
        row["auxiliary_literal_prediction_ANLS_percent"] = literal
    return row


def collect_additional(paper):
    completed, pending, failed = [], [], []
    for name, task, callback in [("NeXT_public_default_K128_alpha05_beta1_wait", "TextVQA", next_full_row),
                                 ("POPE_run03_single_alpha05_beta1_wait_k32", "POPE", pope_alpha_full_row),
                                 ("MMB_single_beta2_alpha05_streamwait", "MMB_EN", mmb_beta_full_row),
                                 ("Qwen_author_FA2_DocVQA_VAL", "DocVQA_VAL", lambda p: qwen_full_row(p, "DocVQA_VAL")),
                                 ("Qwen_author_FA2_AI2D_TEST", "AI2D_TEST", lambda p: qwen_full_row(p, "AI2D_TEST"))]:
        try:
            completed.append(callback(paper))
        except PendingRun as error:
            pending.append({"run_id": name, "task": task, "status": "pending_full_verification",
                            "reason": str(error) or type(error).__name__, "prefix_scored": False})
        except Exception as error:
            frame = traceback.extract_tb(error.__traceback__)[-1]
            record = {"run_id": name, "task": task, "status": "verification_failed",
                      "error_type": type(error).__name__, "reason": str(error) or frame.line or type(error).__name__,
                      "location": str(frame.filename) + ":" + str(frame.lineno), "prefix_scored": False}
            if isinstance(error, MetricDisagreement):
                record.update(review_required="Different complete-run metric definitions; primary score is not changed and neither score is selected as a maximum",
                              complete_run_with_both_metrics=error.row)
            failed.append(record)
    return completed, pending, failed


def full_row(run_id, proof_path, predictions, model, task, budget, alpha, beta, waits, value, paper):
    proof = read(proof_path)
    expected_n = 4377 if task == "MMB_EN" else 8910
    assert proof.get("n_rows", proof.get("prediction_count", proof.get("n_predictions"))) == expected_n
    assert proof.get("success", proof.get("complete", proof.get("all_8910_question_ids_order_verified"))) is True
    actual_sha = sha(predictions)
    assert actual_sha == proof.get("prediction_sha256", proof.get("predictions_sha256"))
    assert sum(1 for line in predictions.open() if line.strip()) == expected_n
    return {
        "run_id": run_id, "model": model, "task": task, "token_budget": budget,
        "alpha": alpha, "beta": beta, "three_wait_nodes": waits,
        "n": expected_n, "n_circular_groups": 1176 if task == "MMB_EN" else None,
        "value": value, "paper": paper,
        "local_minus_paper": str(Decimal(str(value)) - Decimal(str(paper))),
        "unit": "percentage_points", "metric": "exact-letter circular accuracy" if task == "MMB_EN" else "three-category macro F1",
        "source_recipe_identity": "Released source recipe; paper complete argv remains unverified",
        "predictions": str(predictions), "prediction_sha256": actual_sha,
        "verification": str(proof_path), "verification_sha256": sha(proof_path),
        "stability_boundary": "External official-server matching was not performed" if task == "MMB_EN" else
            "231/2334 repeated-input conflicts" if not waits else "0/2334 repeated-input conflicts",
    }


def main():
    comparison = read(COMPARISON)
    def paper(model, task, budget):
        rows = [r for r in comparison["rows"] if r["model_key"] == model and r["task"] == task and r["budget_nominal"] == budget]
        assert len(rows) == 1
        return rows[0]["paper_value"]
    runs = []
    for subdir, name, waits in (("mmbench", "MMB_source_as_is", False), ("mmbench/streamwait_control", "MMB_three_wait_only", True)):
        directory = HERE / subdir
        proof = directory / "independent_verification.json"
        data = read(proof)
        predictions = directory / ("v15_EADP_K128_oldEN_beta1_alpha05_streamwait.jsonl" if waits else "v15_EADP_K128_oldEN_beta1_alpha05.jsonl")
        assert data["n_circular_groups"] == 1176 and data["all_rows_checked"]
        runs.append(full_row(name, proof, predictions, "LLaVA-1.5-7B", "MMB_EN", 128, .5, 1., waits,
                             data["circular_accuracy_percent"], paper("v15", "MMB_EN", 128)))
    for name, waits in (("run01_source_as_is_k32", False), ("run02_source_wait_only_k32", True)):
        directory = HERE / "pope" / name
        proof = directory / "full_result_summary.json"
        data = read(proof)
        runs.append(full_row("POPE_" + name, proof, directory / "predictions.jsonl", "LLaVA-1.5-7B", "POPE", 32,
                             1., 1., waits, data["official_macro_f1_percent"], paper("v15", "POPE", 32)))
    paired = HERE / "mmbench/streamwait_control/paired_full_comparison.json"
    assert read(paired)["success"]
    public_mcq = HERE / "mmbench/public_mcq_metric_check.json"
    metric_check = read(public_mcq)
    assert metric_check["success"] and metric_check["CPU_only"] and not metric_check["API_used"]
    assert len(metric_check["runs"]) == 2
    for row in metric_check["runs"]:
        assert row["n_rows"] == 4377 and row["n_groups"] == 1176
        assert not row["parser_changed_rows"] and not row["unresolved_groups"]
        assert abs(row["public_exact_matching_percent"] - row["independent_local_percent"]) < 1e-12
    diagnosis = HERE / "next_textvqa/panel64_author_wait/result.json"
    panel = read(diagnosis)
    assert panel["n"] == 64 and panel["n_answer_changes"] == panel["n_retained_count_changes"] == 0
    baseline = HERE / "pope/baseline_full_rescore/baseline_full_summary.json"
    source_failure = HERE / "qwen/public_fa2_full/state.json"
    assert read(source_failure)["launcher_exit_code"] == 1
    additional, pending, failed = collect_additional(paper)
    runs.extend(additional)
    report = {
        "definition": "Every complete run is retained. No max selection, prefix accuracy, missing-component mean, or overwrite of the original V2 comparison.",
        "paper_source": {"path": str(COMPARISON), "sha256": sha(COMPARISON)},
        "full_runs": runs,
        "pending_full_runs": pending,
        "verification_failed_full_runs": failed,
        "paired_MMB": {"path": str(paired), "sha256": sha(paired)},
        "MMB_released_exact_matching_recheck": {"path": str(public_mcq), "sha256": sha(public_mcq),
            "all4377_parser_results_identical_in_both_runs": True,
            "external_server_evaluation": False},
        "NeXT_fixed64_diagnostic": {"path": str(diagnosis), "sha256": sha(diagnosis), "answers_changed": 0,
            "retained_counts_changed": 0, "full_benchmark": False, "paper_score_not_compared": True},
        "POPE_full_token_baseline_recheck": {"path": str(baseline), "sha256": sha(baseline),
            "f1_percent": read(baseline)["original_full"]["official_macro_f1_percent"], "paper_rounding_match": True},
        "preserved_failed_public_launch": {"path": str(source_failure), "sha256": sha(source_failure),
            "reason": "Author repository omits result_transfer module; failed before model generation"},
        "remaining": [row["run_id"] for row in pending + failed] + [
            "Remaining paper gaps and causal interpretation of retained complete runs",
            "Published main-table complete argv identity remains unverified"],
        "ledger_builder": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "goal_complete": False,
    }
    (HERE / "full_run_ledger.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    keys = ["run_id", "model", "task", "token_budget", "alpha", "beta", "three_wait_nodes", "n", "n_circular_groups",
            "value", "paper", "local_minus_paper", "unit", "metric", "stability_boundary", "verification", "verification_sha256"]
    with (HERE / "full_run_ledger.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: r[key] for key in keys} for r in runs)
    assert "torch" not in sys.modules, "Ledger export must not load torch or a model"
    print(json.dumps({"completed_full_runs": len(runs), "pending_full_runs": len(pending), "verification_failed_full_runs": len(failed), "goal_complete": False}))


if __name__ == "__main__":
    main()
