"""Independent CPU audit of the completed, frozen 32-page OCR panel.

This entry point refuses incomplete panels before computing alignment checks.
It never computes QA coverage, reward rankings, tail selection, or exclusions.
Runtime V/DeepStack lengths are checked from the saved decoder audit metadata;
the decoder is not loaded or re-executed. Only CPU tensors reproduce masks.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess


ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / "Qwen_vl/outputs/local_reward_diagnostic"
SEEDS = [2026110801, 2026110802, 2026110803, 2026110804]
ARMS = ["topk", *[f"gumbel_{seed}" for seed in SEEDS]]
ALIGNMENT_SEED = 20261108
ALIGNMENT_ARM = ARMS[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class Checks:
    def __init__(self):
        self.failures = []
        self.checked = Counter()

    def check(self, value, check_name, **context):
        self.checked[check_name] += 1
        if not value:
            self.failures.append(dict(check=check_name, **context))
        return bool(value)


def check_hash(checks, path, expected, kind, cache):
    path = Path(path)
    if not checks.check(path.is_file(), kind + "_exists", path=str(path)):
        return None
    key = str(path.resolve())
    if key not in cache:
        cache[key] = sha(path)
    checks.check(cache[key] == expected, kind, path=key, expected=expected, actual=cache[key])
    return cache[key]


def reproduce_masks(scores):
    """Independent literal reproduction of the registered CPU random actions."""
    import torch

    values = torch.tensor(scores, dtype=torch.float32, device="cpu").flatten()
    if not values.numel() or not bool(torch.isfinite(values).all()):
        raise ValueError("mask source scores are empty or nonfinite")
    k = max(1, min(values.numel(), round(.5 * values.numel())))
    deviation = values.std() if values.numel() > 1 else values.new_tensor(0.)
    standardized = (values - values.mean()) / deviation.clamp_min(1e-6)

    def indices(x):
        return torch.argsort(x, descending=True, stable=True)[:k].sort().values.tolist()

    output = {"topk": indices(values)}
    for seed in SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        uniform = torch.rand(values.numel(), generator=generator, dtype=torch.float32)
        eps = torch.finfo(torch.float32).eps
        uniform = uniform.clamp(eps, 1 - eps)
        output[f"gumbel_{seed}"] = indices(standardized - torch.log(-torch.log(uniform)))
    return output


def alignment_sample(checks, items, index):
    """Conservation checks plus a fixed, outcome-blind human-review sample."""
    import Levenshtein
    from reward_metrics import local_rewards, normalize, teacher_fragments

    byid = {item["doc_id"]: item for item in items}
    selected = random.Random(ALIGNMENT_SEED).sample(sorted(byid), 5)
    result = []
    for doc in selected:
        teacher, student = index[(doc, "ocr", "full", None)]["text"], index[(doc, "ocr", ALIGNMENT_ARM, None)]["text"]
        target, output = normalize(teacher), normalize(student)
        item = dict(doc_id=doc, parent_id=byid[doc]["parent_id"], source_image=byid[doc]["image_path"],
                    fixed_arm=ALIGNMENT_ARM, teacher_text=teacher, compressed_text=student,
                    independent_edit_distance=Levenshtein.distance(target, output), checks=[])
        if not target:
            # An empty reference is a scientific eligibility issue, not a
            # reason to replace this preselected alignment-inspection page.
            item.update(status="no_nonempty_teacher_reference", fragments=[], edit_operations=[])
            result.append(item)
            continue
        for width in (64, 32, 128):
            normalized, spans = teacher_fragments(teacher, width)
            checks.check(normalized == target and spans[0][0] == 0 and spans[-1][1] == len(target)
                         and all(a < b for a, b in spans)
                         and all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1)),
                         "fragment_partition", doc_id=doc, width=width)
            starts = [a for a, _ in spans]
            for reverse in (False, True):
                metrics = local_rewards(teacher, student, width=width, reverse_alignment=reverse)
                ops = list(Levenshtein.editops(target[::-1], output[::-1]) if reverse
                           else Levenshtein.editops(target, output))
                independent_errors = [0] * len(spans)
                mapped_ops = []
                for operation, source, dest in ops:
                    source_original = (len(target) - source if operation == "insert" else len(target) - 1 - source) if reverse else source
                    source_for_owner = min(max(source_original, 0), len(target) - 1)
                    owner = bisect_right(starts, source_for_owner) - 1
                    independent_errors[owner] += 1
                    if width == 64 and not reverse:
                        mapped_ops.append(dict(operation=operation, source=source, destination=dest,
                                               fragment=owner, source_context=target[max(0, source-18):source+18],
                                               output_context=output[max(0, dest-18):dest+18]))
                recorded = metrics["fragment_errors"]
                total = sum(recorded)
                distance = item["independent_edit_distance"]
                checks.check(total == distance == len(ops), "edit_distance_conservation", doc_id=doc,
                             width=width, reverse=reverse, recorded_errors=total, independent_distance=distance)
                checks.check(recorded == independent_errors, "independent_fragment_assignment", doc_id=doc,
                             width=width, reverse=reverse)
                checks.check(metrics["fragment_lengths"] == [b-a for a, b in spans],
                             "fragment_lengths", doc_id=doc, width=width, reverse=reverse)
                checks.check(all(abs(loss - min(error / length, 1.)) <= 1e-12
                                 for loss, error, length in zip(metrics["fragment_losses"], recorded, metrics["fragment_lengths"])),
                             "fragment_loss_definition", doc_id=doc, width=width, reverse=reverse)
                item["checks"].append(dict(width=width, reverse_alignment=reverse, n_fragments=len(spans),
                                           fragment_error_total=total, distance_matches=total == distance))
                if width == 64 and not reverse:
                    item["fragments"] = [dict(start=a, end=b, text=target[a:b], edit_errors=recorded[i])
                                         for i, (a, b) in enumerate(spans)]
                    item["edit_operations"] = mapped_ops
        item["status"] = "automated_conservation_checked"
        result.append(item)
    return dict(seed=ALIGNMENT_SEED, fixed_arm=ALIGNMENT_ARM, selected_doc_ids=selected,
                selection="random.Random(seed).sample(sorted(all fixed primary IDs), 5); no outcome filtering",
                manual_review_status="not performed by this script; inspect these same five pages without exclusions",
                records=result)


def audit(base):
    checks, hash_cache = Checks(), {}
    manifest = json.loads((base / "data/manifest.json").read_text())
    run = json.loads((base / "run_manifest.json").read_text())
    raw = (base / "predictions.jsonl").read_text()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    items = [row for row in manifest["items"] if row["role"] == "primary"]
    checks.check(len(items) == 32, "fixed_32_pages", n=len(items))
    checks.check(len({row["doc_id"] for row in items}) == 32, "unique_doc_ids")
    checks.check(len({str(row["parent_id"]) for row in items}) == 32, "unique_parent_ids")
    checks.check(manifest.get("complete") is True, "frozen_data_manifest_complete")
    expected = {(row["doc_id"], "ocr", arm, None) for row in items for arm in ["full", *ARMS]}
    if run.get("secondary_qa"):
        expected |= {(row["doc_id"], "qa", arm, q["question_id"])
                     for row in items for arm in ["full", *ARMS] for q in row["qa"]}
    counts = Counter((r["doc_id"], r["kind"], r["arm"], r.get("question_id")) for r in records)
    missing = sorted(expected - set(counts), key=str)
    unexpected = sorted(set(counts) - expected, key=str)
    duplicate = [dict(key=list(key), count=count) for key, count in counts.items() if count != 1]
    ocr_count = sum(r["kind"] == "ocr" for r in records)
    checks.check(not duplicate, "no_duplicate_outputs", duplicates=duplicate)
    checks.check(not unexpected, "no_unregistered_outputs", unexpected=[list(key) for key in unexpected])
    checks.check(not missing and ocr_count == 192, "complete_formal_panel",
                 missing=[list(key) for key in missing], ocr_outputs=ocr_count)
    base_result = dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
                       prediction_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                       expected_ocr_outputs=192, actual_ocr_outputs=ocr_count,
                       expected_total_outputs=len(expected), actual_total_outputs=len(records))
    if missing or ocr_count != 192:
        return dict(**base_result, status="PENDING_INCOMPLETE_PANEL", pass_all=False,
                    failures=checks.failures, checks=dict(checks.checked),
                    note="No alignment sample, QA coverage or reward ranking computed on this incomplete snapshot.")
    index = {(r["doc_id"], r["kind"], r["arm"], r.get("question_id")): r for r in records}

    frozen = run.get("frozen_hashes", {})
    required = {"Qwen_vl/outputs/local_reward_diagnostic/data/manifest.json",
                "Qwen_vl/scripts/local_reward_diagnostic/run.py",
                "Qwen_vl/scripts/local_reward_diagnostic/layoutlite_runner.py",
                "Qwen_vl/scripts/local_reward_diagnostic/batch_runner.py",
                "Qwen_vl/scripts/local_reward_diagnostic/reward_metrics.py",
                "docs/local_reward_diagnostic_prereg.md",
                "Qwen_vl/third_party/LayoutLite/checkpoints/firered_600.pt"}
    checks.check(required <= set(frozen), "required_frozen_hashes", missing=sorted(required - set(frozen)))
    for name, expected_hash in frozen.items():
        path = Path(name)
        check_hash(checks, path if path.is_absolute() else ROOT / path, expected_hash, "frozen_file_sha256", hash_cache)
    checks.check(run.get("page_ids") == [row["doc_id"] for row in items], "frozen_page_order")
    for key, wanted in dict(planned_pages=32, planned_ocr_outputs=192, mask_seeds=SEEDS,
                            mask_temperature=1, keep_ratio=.5, max_new_tokens=2048,
                            qa_max_new_tokens=32, inference_only=True).items():
        checks.check(run.get(key) == wanted, "registered_run_setting", setting=key, expected=wanted, actual=run.get(key))
    model_dir = Path(run["model_dir"])
    check_hash(checks, model_dir / "model.safetensors", run["model_safetensors_sha256"], "backbone_sha256", hash_cache)
    generation = json.loads((model_dir / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos = {eos} if isinstance(eos, int) else set(eos)

    official = ROOT / "Qwen_vl/third_party/LayoutLite"
    head = subprocess.run(["git", "-C", str(official), "rev-parse", "HEAD"], capture_output=True, text=True)
    checks.check(head.returncode == 0 and head.stdout.strip() == run["layoutlite_commit"],
                 "official_source_commit", actual=head.stdout.strip(), expected=run["layoutlite_commit"])
    clean = subprocess.run(["git", "-C", str(official), "diff", "--exit-code", "HEAD", "--",
                            "layoutlite/modeling_layoutlite.py", "layoutlite/utils/utils.py"], capture_output=True, text=True)
    checks.check(clean.returncode == 0, "official_head_class_and_prompt_unmodified")

    source_hashes, duplicate_mask_counts, mask_index = {}, {}, {}
    for item in items:
        doc = item["doc_id"]
        actual = check_hash(checks, item["image_path"], item["source_sha256"], "source_image_sha256", hash_cache)
        source_hashes[doc] = actual
        parquet = item["source_parquet"]
        expected_parquet = item["source_parquet_sha256"]
        if parquet in mask_index:
            checks.check(mask_index[parquet] == expected_parquet, "consistent_source_parquet_hash", path=parquet)
        else:
            mask_index[parquet] = expected_parquet
        masks = json.loads((base / "masks" / (doc + ".json")).read_text())
        n = masks["n_vis"]
        k = round(n * .5)
        checks.check(masks["doc_id"] == doc and len(masks["official_scores"]) == n, "mask_source_alignment", doc_id=doc)
        entries = masks["masks"]
        checks.check([entry["name"] for entry in entries] == ARMS, "fixed_mask_order", doc_id=doc)
        reproduced = reproduce_masks(masks["official_scores"])
        actual_masks = {entry["name"]: entry for entry in entries}
        duplicate_mask_counts[doc] = len({tuple(entry["keep_idx"]) for entry in entries})
        for offset, arm in enumerate(ARMS):
            entry = actual_masks[arm]
            keep = entry["keep_idx"]
            checks.check(len(keep) == k == entry["K"] and keep == sorted(set(keep))
                         and all(isinstance(token, int) and 0 <= token < n for token in keep),
                         "mask_exact_K_unique_sorted", doc_id=doc, arm=arm)
            checks.check(entry.get("seed") == (None if offset == 0 else SEEDS[offset - 1]),
                         "registered_mask_seed", doc_id=doc, arm=arm)
            checks.check(keep == reproduced[arm], "mask_reproduced_from_frozen_seed_and_scores", doc_id=doc, arm=arm)
        for key, record in [(key, value) for key, value in index.items() if key[0] == doc]:
            _, kind, arm, question_id = key
            info = record["info"]
            context = dict(doc_id=doc, kind=kind, arm=arm, question_id=question_id)
            keep = list(range(n)) if arm == "full" else actual_masks[arm]["keep_idx"]
            checks.check(str(record["parent_id"]) == str(item["parent_id"]), "output_source_parent", **context)
            checks.check(info["input_n_vis"] == n and info["n_vis"] == len(keep)
                         and info["ds_lengths"] == [len(keep)] * 3 and info["keep_idx"] == keep,
                         "saved_runtime_V_DS_lengths_and_indices", **context)
            checks.check(info.get("text_positions_unchanged") is True,
                         "saved_native_text_position_gate", **context)
            if kind == "ocr":
                full_info = index[(doc, "ocr", "full", None)]["info"]
                checks.check(info["full_prompt_tokens"] == full_info["full_prompt_tokens"]
                             and info["prompt_tokens"] == info["full_prompt_tokens"] - n + len(keep)
                             and info["expected_decode_start"] == full_info["expected_decode_start"],
                             "same_original_OCR_prompt_and_native_decode_start", **context)
                if arm != "full":
                    checks.check(info.get("batch_size") == 5 and info.get("batch_row") == ARMS.index(arm)
                                 and info.get("execution_mode") == "same-page same-K batched native decoder",
                                 "formal_compressed_batch5_execution", **context)
            tokens = info["token_ids"]
            cap = 2048 if kind == "ocr" else 32
            stop = info["stop_reason"]
            checks.check(0 < len(tokens) == info["generated_tokens"] <= cap,
                         "generated_token_count", **context)
            checks.check(all(isinstance(token, int) and token >= 0 for token in tokens), "token_id_integrity", **context)
            checks.check(info["decode_positions"] == list(range(info["expected_decode_start"],
                                                                info["expected_decode_start"] + max(0, len(tokens) - 1))),
                         "native_decode_positions_and_count", **context)
            if stop == "eos":
                checks.check(tokens[-1] in eos and not any(token in eos for token in tokens[:-1])
                             and info["truncated"] is False, "first_EOS_stop_and_no_extra_tokens", **context)
            elif stop == "max_new_tokens":
                checks.check(len(tokens) == cap and not any(token in eos for token in tokens)
                             and info["truncated"] is True, "exact_generation_cap_and_truncation", **context)
            else:
                checks.check(False, "known_generation_stop_reason", actual=stop, **context)
    checks.check(len(set(source_hashes.values())) == 32, "unique_source_image_bytes")
    for path, expected_hash in mask_index.items():
        check_hash(checks, path, expected_hash, "source_parquet_sha256", hash_cache)

    gates = {}
    for name in ("identity_gate.json", "batch_gates.json"):
        path = base / name
        gate = json.loads(path.read_text()) if path.exists() else {}
        gates[name] = gate
        checks.check(gate.get("pass_all") is True, "explicit_interface_gate_pass", gate=name)
    batch = gates["batch_gates.json"]
    checks.check(batch.get("full_native_tokens_equal") is True
                 and batch.get("all_prefixes_equal") is True and batch.get("all_decode_positions_equal") is True
                 and batch.get("native_generation_tokens") == 8 and batch.get("compressed_prefix_tokens") == 16
                 and batch.get("batch_size") == 5 and len(batch.get("prefix_comparisons", [])) == 5,
                 "real_stock8_batch16_gate_details")
    outside = Path(batch.get("image", "/nonexistent/outside_page"))
    checks.check(outside.is_file() and sha(outside) not in set(source_hashes.values()), "batch_gate_outside_primary_images")
    archive = base / "serial_throughput_probe"
    checks.check(all((archive / name).is_file() for name in ("predictions.jsonl", "run_manifest.json", "invalidation.json")),
                 "serial_probe_archived_separately")

    sample = alignment_sample(checks, items, index)
    return dict(**base_result, status="PASS" if not checks.failures else "FAIL", pass_all=not checks.failures,
                checks=dict(checks.checked), failures=checks.failures,
                frozen_hashes_verified=len(frozen), unique_source_parquets=len(mask_index),
                distinct_masks_per_page=duplicate_mask_counts,
                duplicate_mask_note="Distinct masks need not equal five; registered duplicate samples are retained, never resampled.",
                alignment_inspection=sample, audit_source_sha256=sha(__file__),
                analysis_source_sha256=sha(Path(__file__).with_name("analyze.py")),
                limitations=["V/DeepStack and positions are saved runtime metadata checked against actual kept-index arrays and interface gates; this audit does not re-execute a model.",
                             "Alignment conservation checks unit edit accounting, not semantic correctness of an ambiguous minimum-edit path.",
                             "No QA coverage or reward ranking was computed by this audit; five alignment pages were chosen without outcome filtering."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    destination = args.out or args.directory / "audit.json"
    try:
        result = audit(args.directory)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result = dict(status="FAIL_AUDIT_INPUT", pass_all=False, error_type=type(error).__name__,
                      error=str(error), audit_source_sha256=sha(__file__))
    atomic_json(destination, result)
    print(json.dumps(dict(status=result["status"], pass_all=result["pass_all"], output=str(destination.resolve()),
                          checks=result.get("checks"), failures=result.get("failures", [])), indent=2))
    raise SystemExit(0 if result["pass_all"] else 2)


if __name__ == "__main__":
    main()
