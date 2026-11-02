#!/usr/bin/env python3
"""CPU audit of repeated exact image/guidance/full-prompt inputs in the as-is run."""
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
AUTHOR = OUT.parents[2] / "_reproduction/author_eadp"
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
text_sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
verified = json.loads((OUT / "independent_verification.json").read_text())
assert verified["success"] and verified["n_rows"] == 4377
raw_path = OUT / "v15_EADP_K128_oldEN_beta1_alpha05.jsonl"
assert sha(raw_path) == verified["prediction_sha256"]
answers = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
runtime = [json.loads(line) for line in (OUT / "observed/runtime.jsonl").read_text().splitlines() if line.strip()]
image_entries = {item["index"]: item for item in json.loads((OUT / "image_manifest.json").read_text())}
inputs = json.loads((OUT / "cpu_inputs_current.json").read_text())
llm_records = json.loads((OUT / "cpu_llm_tokens_current.json").read_text())["records"]
loaded = json.loads((OUT / "observed/loaded_model.json").read_text())
csv.field_size_limit(2**31 - 1)
with (OUT / "data/mmbench_dev_20230712.tsv").open(newline="") as stream:
    gt = {int(row["index"]): row["answer"] for row in csv.DictReader(stream, delimiter="\t")}
spec = importlib.util.spec_from_file_location("mmbench_audit_conversation", AUTHOR / "LLaVA/llava/conversation.py")
conversation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conversation)

rows = []
for answer, observed, llm in zip(answers, runtime, llm_records):
    idx = int(answer["question_id"])
    assert idx == llm["index"]
    guidance = observed["guidance_text"]
    assert text_sha(guidance) == observed["guidance_text_sha256"]
    assert guidance == answer["prompt"]
    image_marker = "<im_start><image><im_end>" if loaded["model_config"].get("mm_use_im_start_end") else "<image>"
    qs = image_marker + "\n" + guidance + "\nAnswer with the option's letter from the given choices directly."
    conv = conversation.conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    assert text_sha(full_prompt) == llm["prompt_sha256"]
    image = image_entries[idx]
    fingerprints = inputs["images"][image["payload_sha256"]]
    match = re.search(r"\b([A-E])\b", answer["text"].strip())
    rows.append({"index": idx, "position": observed["call"] - 1,
                 "image_payload_sha256": image["payload_sha256"],
                 "decoded_rgb_sha256": image["decoded_rgb_sha256"], "image_size": image["size"],
                 "native_fp16_image_sha256": fingerprints["float16_sha256"],
                 "guidance_sha256": text_sha(guidance), "guidance": guidance,
                 "full_prompt_sha256": text_sha(full_prompt), "full_prompt": full_prompt,
                 "LLM_token_ids_sha256": llm["token_ids_sha256"],
                 "raw_answer": answer["text"], "parsed_letter": match.group(1) if match else None,
                 "public_gt": gt[idx]})

summaries, details = {}, {}
for tier, image_key in (("exact_encoded_image", "image_payload_sha256"),
                        ("exact_decoded_RGB_and_size", "decoded_rgb_sha256"),
                        ("exact_native_FP16_image_and_size", "native_fp16_image_sha256")):
    groups = defaultdict(list)
    for row in rows:
        key = (row[image_key], tuple(row["image_size"]), row["guidance_sha256"],
               row["full_prompt_sha256"], row["LLM_token_ids_sha256"])
        groups[key].append(row)
    repeated = [members for members in groups.values() if len(members) > 1]
    raw_conflicts = [members for members in repeated if len({r["raw_answer"] for r in members}) > 1]
    letter_conflicts = [members for members in repeated if len({r["parsed_letter"] for r in members}) > 1]
    pair_denominator = sum(len(members) * (len(members) - 1) // 2 for members in repeated)
    def conflicting_pairs(field):
        return sum(sum(a[field] != b[field] for i, a in enumerate(members) for b in members[i + 1:])
                   for members in repeated)
    summaries[tier] = {"total_rows": len(rows), "unique_exact_input_keys": len(groups),
                       "repeated_input_groups": len(repeated), "rows_in_repeated_groups": sum(map(len, repeated)),
                       "extra_repeat_rows": sum(len(members) - 1 for members in repeated),
                       "duplicate_pair_denominator": pair_denominator,
                       "raw_answer_conflict_groups": len(raw_conflicts), "raw_answer_conflict_pairs": conflicting_pairs("raw_answer"),
                       "parsed_letter_conflict_groups": len(letter_conflicts), "parsed_letter_conflict_pairs": conflicting_pairs("parsed_letter"),
                       "repeated_groups_with_conflicting_public_GT": sum(len({r["public_gt"] for r in members}) > 1 for members in repeated)}
    details[tier] = [{"indices": [r["index"] for r in members],
                     "positions": [r["position"] for r in members],
                     "image_identity_sha256": members[0][image_key], "image_size": members[0]["image_size"],
                     "guidance_sha256": members[0]["guidance_sha256"], "guidance": members[0]["guidance"],
                     "full_prompt_sha256": members[0]["full_prompt_sha256"], "full_prompt": members[0]["full_prompt"],
                     "LLM_token_ids_sha256": members[0]["LLM_token_ids_sha256"],
                     "raw_answers": [r["raw_answer"] for r in members],
                     "parsed_letters": [r["parsed_letter"] for r in members], "public_GT": [r["public_gt"] for r in members],
                     "raw_answer_conflict": len({r["raw_answer"] for r in members}) > 1,
                     "parsed_letter_conflict": len({r["parsed_letter"] for r in members}) > 1}
                    for members in repeated]

sources = [raw_path, OUT / "observed/runtime.jsonl", OUT / "observed/loaded_model.json",
           OUT / "image_manifest.json", OUT / "cpu_inputs_current.json", OUT / "cpu_llm_tokens_current.json",
           OUT / "protocol.json", OUT / "independent_verification.json", OUT / "data/mmbench_dev_20230712.tsv"]
report = {"scope": "As-is author oldEN4377 K128 alpha.5 beta1; complete input equality includes image identity/size, full CLIP guidance text, rendered LLM prompt and LLM token IDs.",
          "summaries": summaries, "repeated_input_groups": details,
          "sources": [{"path": str(path), "sha256": sha(path)} for path in sources],
          "auditor_sha256": sha(__file__), "GPU_used": False,
          "interpretation": "Answer conflicts on exact repeated greedy inputs demonstrate run inconsistency, not by themselves its cause. Zero repeated keys gives no same-input repeat test; absence of observed conflicts does not prove stream correctness."}
(OUT / "duplicate_input_consistency.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summaries, ensure_ascii=False, indent=2))
