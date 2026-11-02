"""CPU full-run beta comparison and exact V11 overlap; never scores a prefix."""
import argparse
import ast
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
BETA1 = BASE / "streamwait_control"
BETA2 = BASE / "beta2_control"
OUT = BASE / "beta1_vs_beta2_and_v11_full"
OLD = Path("/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/audit_followup_20261008/mmb_eadp_immediate_20261010")
EXPECTED_PROTOCOL = "f4a2eff7485313e8ee1bec7553c4897b60abb48ccd4beb34e165e20a09e49f40"
EXPECTED_OLD_RAW = "c6a57feb6a7f7e27d0c3255cefa9ef409d82b713f6a431995b61c0f3e5d41706"


class Pending(Exception):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path, expected_n, expected_sha):
    assert sha(path) == expected_sha, str(path)
    raw = Path(path).read_bytes()
    assert raw.endswith(b"\n")
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == expected_n, "Full denominator required before any paired scoring"
    return rows


def original_exit(record, gpu_pids):
    path = Path("/proc", str(record["pid"]), "stat")
    try:
        ticks = int(path.read_text().rsplit(")", 1)[1].split()[19])
    except FileNotFoundError:
        ticks = None
    if ticks == record["start_ticks"] or record["pid"] in gpu_pids:
        raise Pending("Original native worker has not actually exited")
    return dict(original=record, original_exited=True, current_start_ticks_if_reused=ticks, GPU_pid_absent=True)


def complete_gate():
    # Only control-plane metadata is inspected before all terminal proofs exist.
    required = ["state.json", "independent_verification.json", "local_circular_score.json", "conversion.json", "after_pope_alpha.state.json"]
    assert not (BETA2 / "after_pope_alpha.failure.json").exists(), "Preserved beta2 supervisor failure"
    for name in ("state.json", "after_pope_alpha.state.json"):
        try:
            meta = read(BETA2 / name)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise Pending("Control record is missing or being written") from error
        assert not any(word in meta.get("status", "").lower() for word in ("fail", "error", "abort", "cancel")), meta
        assert meta.get("exit_code") in (None, 0), meta
    if any(not (BETA2 / name).exists() for name in required):
        raise Pending("Waiting for all4377/native score/XLSX/independent/waiter complete proof")
    try:
        state, proof, score, conversion, waiter = [read(BETA2 / name) for name in required]
    except json.JSONDecodeError as error:
        raise Pending("Terminal proof is being written") from error
    if state["status"] != "complete" or waiter["status"] != "complete":
        raise Pending("Native controller or supervisor is finalizing")
    assert state["exit_code"] == waiter["exit_code"] == conversion["exit_code"] == 0
    assert proof["success"] and proof["all_rows_checked"] and proof["n_rows"] == score["n_rows"] == 4377
    assert proof["n_circular_groups"] == score["n_groups"] == 1176 and proof["current_beta"] == 2
    assert state["protocol_sha256"] == sha(BETA2 / "protocol.json") == EXPECTED_PROTOCOL
    assert state["score_sha256"] == sha(BETA2 / "local_circular_score.json")
    assert state["conversion_record_sha256"] == sha(BETA2 / "conversion.json")
    assert waiter["actual_native_exited"] and waiter["independent_verification_sha256"] == sha(BETA2 / "independent_verification.json")
    assert not conversion["submitted_to_external_server"] and sha(conversion["output"]) == proof["submission_sha256"]
    for directory in (BASE, BETA1):
        prior = read(directory / "independent_verification.json")
        assert prior["success"] and prior["n_rows"] == 4377 and prior["n_circular_groups"] == 1176
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    pids = {int(line.strip()) for line in gpu.splitlines() if line.strip()}
    exits = [original_exit(read(directory / "state.json")["native_process"], pids) for directory in (BASE, BETA1, BETA2)]
    return proof, score, exits


def source_helpers(path):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("is_none", "get_options")]
    assert len(nodes) == 2
    ns = dict(math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def native_prompt(row, helpers):
    options = helpers["get_options"](row, list("ABCD"))
    prompt = row["question"]
    if not helpers["is_none"](row["hint"]):
        prompt = row["hint"] + "\n" + prompt
    for char, option in zip("ABCD", options):
        prompt += "\n" + char + ". " + option
    return prompt, options


def conversation_renderer(path, module_name):
    # Original prompt-only source imports PIL, not a model or torch.
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    def render(question):
        conv = module.conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], "<image>\n" + question + "\nAnswer with the option's letter from the given choices directly.")
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    return render


def image_hashes(data):
    # V11 rotations store base-row IDs; author old entry explicitly resolves them.
    payloads = {int(row["index"]) % 1000000: row["image"] for _, row in data.iterrows()
                if isinstance(row["image"], str) and len(row["image"]) > 100}
    cache = {}
    result = {}
    for _, row in data.iterrows():
        value = row["image"]
        payload = value if isinstance(value, str) and len(value) > 100 else payloads[int(row["index"]) % 1000000]
        if payload not in cache:
            cache[payload] = hashlib.sha256(base64.b64decode(payload)).hexdigest()
        result[int(row["index"])] = cache[payload]
    return result


def letter(text):
    tokens = re.findall(r"\b[A-E]\b", text.strip())
    return tokens[0] if tokens else None


def transitions(first, second):
    count = Counter(zip(first, second))
    return dict(correct_to_correct=count[True, True], wrong_to_correct=count[False, True],
                correct_to_wrong=count[True, False], wrong_to_wrong=count[False, False])


def compare():
    proof2, score2, exits = complete_gate()
    plans = [read(directory / "protocol.json") for directory in (BASE, BETA1, BETA2)]
    checks = [read(directory / "independent_verification.json") for directory in (BASE, BETA1, BETA2)]
    assert plans[0]["dataset"] == plans[1]["dataset"] == plans[2]["dataset"]
    assert plans[0]["image_manifest_sha256"] == plans[1]["image_manifest_sha256"] == plans[2]["image_manifest_sha256"]
    parameters = [dict(plan["parameters"]) for plan in plans]
    assert parameters[0] == parameters[1] and parameters[2] == dict(parameters[1], beta=2.)
    for plan in plans:
        for source in plan["source_files"]:
            assert sha(source["path"]) == source["sha256"], source["path"]
    paths = [Path(p["command"][p["command"].index("--answers-file") + 1]) for p in plans]
    raw = [jsonl(path, 4377, check["prediction_sha256"]) for path, check in zip(paths, checks)]
    runtime = jsonl(BETA2 / "observed/runtime.jsonl", 4377, proof2["runtime_sha256"])
    data_path = Path(plans[0]["dataset"]["path"])
    assert sha(data_path) == plans[0]["dataset"]["sha256"] == proof2["dataset_sha256"]
    data = pd.read_table(data_path)
    assert len(data) == 4377
    helper = source_helpers(Path(plans[2]["command"][3]))
    render_new = conversation_renderer(ROOT / "_reproduction/author_eadp/LLaVA/llava/conversation.py", "mmb_new_native_conversation")
    assert not read(Path(plans[2]["command"][5]) / "config.json")["mm_use_im_start_end"]
    encoded = image_hashes(data)
    manifest = read(BASE / "image_manifest.json")
    assert sha(BASE / "image_manifest.json") == plans[0]["image_manifest_sha256"]
    assert encoded == {int(row["index"]): row["payload_sha256"] for row in manifest}
    rows, grouped = [], [defaultdict(list) for _ in raw]
    for position, ((_, item), observed) in enumerate(zip(data.iterrows(), runtime)):
        idx = int(item["index"]); prompt, options = native_prompt(item, helper)
        answers = [arm[position] for arm in raw]
        for answer in answers:
            assert int(answer["question_id"]) == idx and answer["prompt"] == prompt and answer["options"] == options
            assert answer["round_id"] == 0 and answer["option_char"] == list("ABCD")[:len(options)]
            assert answer["model_id"] == "llava-v1.5-7b"
        assert observed["call"] == position + 1 and observed["guidance_text"] == prompt
        assert observed["guidance_text_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert observed["returned_visual_token_num"] == 128
        assert not observed["do_sample"] and observed["temperature"] == 0 and observed["num_beams"] == 1 and observed["max_new_tokens"] == 1024
        assert observed["image_dtype"] == "torch.float16" and observed["image_shape"] == [1, 3, 336, 336]
        assert answers[0]["text"] == answers[1]["text"], "Original versus wait baseline changed unexpectedly"
        parsed = [letter(answer["text"]) for answer in answers]
        correct = [value == item["answer"] for value in parsed]
        for groups, hit in zip(grouped, correct):
            groups[idx % 1000000].append(hit)
        rows.append(dict(index=idx, base_group=idx % 1000000, prompt=prompt,
                         rendered_full_prompt=render_new(prompt), options=options, GT=item["answer"],
                         image_sha256=encoded[idx], beta1_text=answers[1]["text"], beta2_text=answers[2]["text"],
                         beta1_letter=parsed[1], beta2_letter=parsed[2], beta1_correct=correct[1], beta2_correct=correct[2]))
    group_hits = [{key: all(values) for key, values in groups.items()} for groups in grouped]
    for check, hits in zip(checks, group_hits):
        assert len(hits) == 1176 and sum(hits.values()) == check["circular_correct"]
        assert abs(sum(hits.values()) * 100 / 1176 - check["circular_accuracy_percent"]) < 1e-12
    assert score2["row_correct"] == sum(row["beta2_correct"] for row in rows)
    patch = proof2["controlled_stream_patch"]
    assert patch["exact_added_statement_count"] == 3 and patch["restored_ast_identical"] and patch["applied_in_this_process"]
    loaded = read(BETA2 / "observed/loaded_model.json")
    assert loaded["native_argv"] == plans[2]["command"][3:] and (loaded["visual_token_num"], loaded["alpha"], loaded["beta"]) == (128, .5, 2.)
    old_score_path = OLD / "v15_mmben_E128_streamwait.official.score.json"
    old_score = read(old_score_path)
    old_raw_path = OLD / "v15_mmben_E128_streamwait.jsonl"
    assert old_score["success"] and old_score["n_rows"] == 4876 and old_score["n_groups"] == 1292
    assert old_score["prediction_sha256"] == EXPECTED_OLD_RAW
    old_raw = jsonl(old_raw_path, 4876, EXPECTED_OLD_RAW)
    old_protocol_path = OLD / "v15_mmben_E128_streamwait.control.protocol.json"
    assert sha(old_protocol_path) == old_score["control_protocol_sha256"]
    old_protocol = read(old_protocol_path)
    old_data_path = Path(old_protocol["parameters"]["question_file"])
    assert sha(old_data_path) == old_protocol["origin"]["question_sha256"]
    old_data = pd.read_table(old_data_path)
    assert len(old_data) == 4876
    old_native = Path(old_protocol["origin"]["native_entry"])
    assert sha(old_native) == old_protocol["source_and_input_sha256"][str(old_native)]
    old_helper = source_helpers(old_native)
    old_conversation_path = old_native.parents[1] / "conversation.py"
    assert sha(old_conversation_path) == old_protocol["source_and_input_sha256"][str(old_conversation_path)]
    render_old = conversation_renderer(old_conversation_path, "mmb_v11_native_conversation")
    old_encoded = image_hashes(old_data)
    old_images_path = OLD / "legacy_mmben_images.manifest.json"
    assert sha(old_images_path) == old_score["image_manifest_sha256"] == old_protocol["image_manifest_sha256"]
    old_image_manifest = read(old_images_path)
    assert {idx % 1000000: value for idx, value in old_encoded.items()} == {int(row["embedded_base"]): row["bytes_sha256"] for row in old_image_manifest["records"]}
    old_runtime_path = OLD / "v15_mmben_E128_streamwait.jsonl.runtime.jsonl"
    old_runtime = jsonl(old_runtime_path, 4876, old_score["runtime_sha256"])
    old_by_id, old_groups = {}, defaultdict(list)
    for position, ((_, item), answer, observed) in enumerate(zip(old_data.iterrows(), old_raw, old_runtime)):
        idx = int(item["index"]); prompt, options = native_prompt(item, old_helper)
        assert int(answer["question_id"]) == idx and answer["prompt"] == prompt and answer["options"] == options
        assert observed["question_id"] == idx and observed["question_position"] == position
        assert observed["prompt_sha256"] == observed["guidance_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert (observed["actual_model_object"]["alpha"], observed["actual_model_object"]["beta"], observed["actual_visual_tokens_retained"]) == (.5, 2., 128)
        assert not observed["native_generate_parameters"]["do_sample"] and observed["native_generate_parameters"]["max_new_tokens"] == 1024
        hit = letter(answer["text"]) == item["answer"]
        old_groups[idx % 1000000].append(hit)
        old_by_id[idx] = dict(prompt=prompt, rendered_full_prompt=render_old(prompt), options=options,
                             GT=item["answer"], image_sha256=old_encoded[idx], text=answer["text"], correct=hit)
    assert len(old_by_id) == 4876 and len(old_groups) == 1292 and sum(all(group) for group in old_groups.values()) == old_score["circular_correct"]
    common = [row for row in rows if row["index"] in old_by_id]
    staged = {"same_ID": len(common)}
    selected = common
    for key in ("prompt", "options", "GT", "image_sha256", "rendered_full_prompt"):
        selected = [row for row in selected if row[key] == old_by_id[row["index"]][key]]
        staged["plus_exact_" + key] = len(selected)
    strict_ids = {row["index"] for row in selected}
    new_group_ids, old_group_ids = defaultdict(set), defaultdict(set)
    for row in rows:
        new_group_ids[row["base_group"]].add(row["index"])
    for idx in old_by_id:
        old_group_ids[idx % 1000000].add(idx)
    strict_groups = [key for key, indices in new_group_ids.items()
                     if indices == old_group_ids.get(key) and indices <= strict_ids]
    strict_rows = [dict(index=row["index"], base_group=row["base_group"], GT=row["GT"],
                        image_sha256=row["image_sha256"], prompt_sha256=hashlib.sha256(row["prompt"].encode()).hexdigest(),
                        full_prompt_sha256=hashlib.sha256(row["rendered_full_prompt"].encode()).hexdigest(),
                        old_V11_beta2_text=old_by_id[row["index"]]["text"], old_V11_beta2_correct=old_by_id[row["index"]]["correct"],
                        oldEN_beta1_text=row["beta1_text"], oldEN_beta2_text=row["beta2_text"],
                        oldEN_beta1_correct=row["beta1_correct"], oldEN_beta2_correct=row["beta2_correct"])
                   for row in selected]
    group_rows = [dict(base_group=key, n_rotations=len(new_group_ids[key]),
                       old_V11_beta2_all_correct=all(old_by_id[idx]["correct"] for idx in new_group_ids[key]),
                       oldEN_beta1_all_correct=group_hits[1][key], oldEN_beta2_all_correct=group_hits[2][key])
                  for key in strict_groups]
    n_changes = sum(row["beta1_text"] != row["beta2_text"] for row in rows)
    group_order = sorted(group_hits[1])
    sources = [Path(__file__), data_path, BASE / "image_manifest.json", old_score_path, old_raw_path, old_protocol_path,
               old_data_path, old_native, old_conversation_path, old_images_path, old_runtime_path]
    for directory, path in zip((BASE, BETA1, BETA2), paths):
        sources += [path, directory / "protocol.json", directory / "independent_verification.json", directory / "state.json"]
    sources += [BETA2 / name for name in ("conversion.json", "after_pope_alpha.state.json", "observed/runtime.jsonl")]
    report = dict(success=True, CPU_only=True, completed_utc=datetime.now(timezone.utc).isoformat(),
                  full4377_beta_comparison=dict(n_rows=4377, n_groups=1176,
                    beta1_value_percent=checks[1]["circular_accuracy_percent"], beta2_value_percent=proof2["circular_accuracy_percent"],
                    beta2_minus_beta1_pp=(checks[2]["circular_correct"]-checks[1]["circular_correct"]) * 100 / 1176,
                    paper_value=proof2["paper_reported"], beta2_minus_paper_pp=proof2["circular_accuracy_percent"]-proof2["paper_reported"],
                    raw_answer_changed_rows=n_changes,
                    parsed_letter_changed_rows=sum(row["beta1_letter"] != row["beta2_letter"] for row in rows),
                    row_correctness_transitions=transitions([r["beta1_correct"] for r in rows],[r["beta2_correct"] for r in rows]),
                    circular_correctness_transitions=transitions([group_hits[1][i] for i in group_order],[group_hits[2][i] for i in group_order])),
                  V11_same_beta2_descriptive_overlap=dict(old_n=4876, old_n_groups=1292, old_value_percent=old_score["value"],
                    input_matching_stages=staged, strict_matched_rows=len(selected), strict_complete_circular_groups=len(strict_groups),
                    row_correctness_transitions=transitions([old_by_id[r["index"]]["correct"] for r in selected],[r["beta2_correct"] for r in selected]),
                    raw_answer_changed_rows=sum(old_by_id[r["index"]]["text"] != r["beta2_text"] for r in selected),
                    exact_rows=strict_rows, exact_complete_groups=group_rows),
                  native_exit_proof=exits, sources=[dict(path=str(p),sha256=sha(p)) for p in sources],
                  limits=["The4377-row beta comparison holds dataset, model, prompt, decoding, three waits and local metric fixed, but not replayed low-level scheduling.",
                          "V11 overlap requires exact ID, complete native/rendered prompt, options, GT and encoded image SHA; full circular comparison additionally requires equal complete rotation sets.",
                          "V11 whole-dataset score and oldEN whole-dataset score have different coverage; their difference is not a causal dataset estimate.",
                          "Historical versus fresh source/env/observer differences are not erased by matched input rows; neither matched subset nor a higher value is paper reproduction.",
                          "Full4377/local1176 metrics are separate from external official-server/API matching. No new generation or GPU is performed."])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "paired_rows.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows))
    report["paired_rows"] = dict(path=str(OUT / "paired_rows.jsonl"),sha256=sha(OUT / "paired_rows.jsonl"))
    (OUT / "comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    assert "torch" not in sys.modules
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            if args.check:
                complete_gate(); print(json.dumps(dict(ready=True,raw_predictions_read=False))); break
            result = compare()
            print(json.dumps({key: value for key, value in result.items() if key not in ("sources", "V11_same_beta2_descriptive_overlap")},ensure_ascii=False,indent=2))
            print(json.dumps({key:value for key,value in result["V11_same_beta2_descriptive_overlap"].items() if key not in ("exact_rows","exact_complete_groups")},ensure_ascii=False,indent=2))
            break
        except Pending as error:
            if args.check or not args.wait:
                print(json.dumps(dict(status="pending",reason=str(error),raw_predictions_read=False))); break
            time.sleep(10)
