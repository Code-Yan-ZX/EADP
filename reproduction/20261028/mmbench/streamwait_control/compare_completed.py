#!/usr/bin/env python3
"""CPU-only full paired comparison; refuses incomplete or unverified runs."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


OUT = Path(__file__).resolve().parent
BASE = OUT.parent
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
jsonl = lambda path: [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def native_path(plan):
    return Path(plan["command"][plan["command"].index("--answers-file") + 1])


def parse_letter(text):
    tokens = re.findall(r"\b[A-E]\b", text.strip())
    return tokens[0] if tokens else None


def worker_exit_evidence(directory):
    identity = read(directory / "state.json")["native_process"]
    stat_path = Path("/proc") / str(identity["pid"]) / "stat"
    current_ticks = int(stat_path.read_text().rsplit(")", 1)[1].split()[19]) if stat_path.exists() else None
    assert current_ticks != identity["start_ticks"], "Original model worker still exists"
    return {"frozen_identity": identity, "current_start_ticks_if_pid_reused": current_ticks,
            "original_worker_exited": True}


def main():
    plans = [read(directory / "protocol.json") for directory in (BASE, OUT)]
    checks = [read(directory / "independent_verification.json") for directory in (BASE, OUT)]
    for check in checks:
        assert check["success"] and check["all_rows_checked"] and check["n_rows"] == 4377
        assert check["n_circular_groups"] == 1176
    assert plans[0]["dataset"] == plans[1]["dataset"]
    assert plans[0]["parameters"] == plans[1]["parameters"]
    assert plans[0]["image_manifest_sha256"] == plans[1]["image_manifest_sha256"]
    assert plans[0]["cwd"] == plans[1]["cwd"]
    commands = [list(plan["command"]) for plan in plans]
    for command in commands:
        command[1:3] = ["ISOLATED_OBSERVER_OR_WAIT_WRAPPER", "ISOLATED_OBSERVER_OUTPUT"]
        command[command.index("--answers-file") + 1] = "ISOLATED_PREDICTION_OUTPUT"
    assert commands[0] == commands[1], "Native arguments differ"
    for plan in plans:
        for item in plan["source_files"]:
            assert sha(item["path"]) == item["sha256"], item["path"]
    patch = read(OUT / "runtime_patch_validation.json")
    assert patch["applied_in_this_process"] and patch["restored_ast_identical"]
    assert patch["exact_added_statement_count"] == 3
    assert patch["author_source_after_patch_sha256"] == patch["source_sha256"]
    paths = [native_path(plan) for plan in plans]
    raw = [jsonl(path) for path in paths]
    for path, check in zip(paths, checks):
        assert sha(path) == check["prediction_sha256"]
    csv.field_size_limit(2**31 - 1)
    data_path = Path(plans[0]["dataset"]["path"])
    assert sha(data_path) == plans[0]["dataset"]["sha256"]
    with data_path.open(newline="") as stream:
        dataset = list(csv.DictReader(stream, delimiter="\t"))
    assert len(dataset) == len(raw[0]) == len(raw[1]) == 4377
    rows, group_correct = [], [defaultdict(list), defaultdict(list)]
    for position, (data, first, second) in enumerate(zip(dataset, *raw)):
        idx = int(data["index"])
        assert int(first["question_id"]) == int(second["question_id"]) == idx
        for key in ("prompt", "options", "option_char", "round_id", "model_id"):
            assert first[key] == second[key], (idx, key)
        letters = [parse_letter(answer["text"]) for answer in (first, second)]
        correct = [letter == data["answer"] for letter in letters]
        for groups, value in zip(group_correct, correct):
            groups[idx % 1000000].append(value)
        rows.append({"index": idx, "position": position, "base_group": idx % 1000000,
                     "public_GT": data["answer"], "as_is_text": first["text"], "wait_text": second["text"],
                     "as_is_letter": letters[0], "wait_letter": letters[1],
                     "as_is_correct": correct[0], "wait_correct": correct[1],
                     "raw_answer_changed": first["text"] != second["text"],
                     "parsed_letter_changed": letters[0] != letters[1]})
    assert set(group_correct[0]) == set(group_correct[1]) and len(group_correct[0]) == 1176
    group_rows = [{"base_group": idx, "n_rotations": len(group_correct[0][idx]),
                   "as_is_all_correct": all(group_correct[0][idx]),
                   "wait_all_correct": all(group_correct[1][idx])}
                  for idx in sorted(group_correct[0])]
    totals = [sum(row[key] for row in group_rows) for key in ("as_is_all_correct", "wait_all_correct")]
    for total, check in zip(totals, checks):
        assert total == check["circular_correct"]
        assert abs(total * 100 / 1176 - check["circular_accuracy_percent"]) <= 1e-12
    row_transitions = Counter((row["as_is_correct"], row["wait_correct"]) for row in rows)
    group_transitions = Counter((row["as_is_all_correct"], row["wait_all_correct"]) for row in group_rows)
    def transitions_json(counts):
        return {"correct_to_correct": counts[True, True], "wrong_to_correct": counts[False, True],
                "correct_to_wrong": counts[True, False], "wrong_to_wrong": counts[False, False]}
    duplicate_report_path = BASE / "duplicate_input_consistency.json"
    duplicate = read(duplicate_report_path)
    for item in duplicate["sources"]:
        assert sha(item["path"]) == item["sha256"]
    by_index = {row["index"]: row for row in rows}
    repeats = []
    for pair in duplicate["repeated_input_groups"]["exact_encoded_image"]:
        members = [by_index[idx] for idx in pair["indices"]]
        repeats.append({"indices": pair["indices"], "image_payload_sha256": pair["image_identity_sha256"],
                        "full_prompt_sha256": pair["full_prompt_sha256"], "guidance_sha256": pair["guidance_sha256"],
                        "LLM_token_ids_sha256": pair["LLM_token_ids_sha256"], "public_GT": pair["public_GT"],
                        "as_is_texts": [row["as_is_text"] for row in members],
                        "wait_texts": [row["wait_text"] for row in members],
                        "as_is_letter_conflict": len({row["as_is_letter"] for row in members}) > 1,
                        "wait_letter_conflict": len({row["wait_letter"] for row in members}) > 1})
    sources = [data_path, duplicate_report_path, OUT / "runtime_patch_validation.json", OUT / "launch_gate_evidence.json"]
    for directory, path in zip((BASE, OUT), paths):
        sources += [path, directory / "protocol.json", directory / "state.json",
                    directory / "independent_verification.json", directory / "observed/runtime.jsonl"]
    exits = [worker_exit_evidence(directory) for directory in (BASE, OUT)]
    gpu_apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
    gpu_pids = {int(line.split(",", 1)[0]) for line in gpu_apps.splitlines() if line.strip()}
    for evidence in exits:
        assert evidence["frozen_identity"]["pid"] not in gpu_pids, "Old native PID still listed as a GPU app"
    report = {"success": True, "completed_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Full4377 old EN20230712 K128 alpha0.5 beta1; only three caller/image/text stream-wait AST statements differ.",
              "n_rows": 4377, "n_circular_groups": 1176,
              "as_is_circular_correct": totals[0], "wait_circular_correct": totals[1],
              "as_is_circular_accuracy_percent": totals[0] * 100 / 1176,
              "wait_circular_accuracy_percent": totals[1] * 100 / 1176,
              "wait_minus_as_is_percentage_points": (totals[1] - totals[0]) * 100 / 1176,
              "paper_reported": checks[0]["paper_reported"],
              "wait_minus_paper_percentage_points": totals[1] * 100 / 1176 - checks[0]["paper_reported"],
              "raw_answer_changed_rows": sum(row["raw_answer_changed"] for row in rows),
              "parsed_letter_changed_rows": sum(row["parsed_letter_changed"] for row in rows),
              "row_correctness_transitions": transitions_json(row_transitions),
              "circular_group_correctness_transitions": transitions_json(group_transitions),
              "row_letter_transitions": dict(Counter(str(row["as_is_letter"]) + "->" + str(row["wait_letter"]) for row in rows)),
              "duplicate_pair_denominator": duplicate["summaries"]["exact_encoded_image"]["duplicate_pair_denominator"],
              "as_is_duplicate_conflict_groups": sum(pair["as_is_letter_conflict"] for pair in repeats),
              "wait_duplicate_conflict_groups": sum(pair["wait_letter_conflict"] for pair in repeats),
              "repeated_input_groups": repeats,
              "changed_rows": [row for row in rows if row["raw_answer_changed"]],
              "changed_circular_groups": [row for row in group_rows if row["as_is_all_correct"] != row["wait_all_correct"]],
              "worker_exit": exits, "GPU_processes_after_control": gpu_apps,
              "sources": [{"path": str(path), "sha256": sha(path)} for path in sources],
              "auditor_sha256": sha(__file__), "GPU_used_by_auditor": False,
              "limits": ["Same native inputs, parameters and metric, but companion GPU workload and low-level scheduling are not replayed identically.",
                         "Every outcome is retained; neither parameter selection nor repeated generation is based on the observed scores.",
                         "A score or answer change alone does not prove its cause; only four exact-input duplicate pairs are present.",
                         "Local exact-letter circular accuracy retains all1176 public GT groups; external official-server matching is separate."]}
    (OUT / "paired_full_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("changed_rows", "changed_circular_groups", "sources", "repeated_input_groups")}, indent=2))


if __name__ == "__main__":
    main()
