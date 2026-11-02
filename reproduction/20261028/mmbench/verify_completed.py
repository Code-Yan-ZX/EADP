#!/usr/bin/env python3
"""Independent CPU check of every native old-EN answer and circular group."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

OUT = Path(__file__).resolve().parent
protocol = json.loads((OUT / "protocol.json").read_text())
state = json.loads((OUT / "state.json").read_text())
postprocessing_path = OUT / "postprocessing_completion.json"
postprocessing = json.loads(postprocessing_path.read_text()) if postprocessing_path.exists() else None
assert state["exit_code"] == 0, "Native generation must finish successfully"
assert state["status"] == "complete" or (postprocessing and postprocessing["status"] == "complete"), "Scoring and conversion must finish first"
data = Path(protocol["dataset"]["path"])
native = Path(protocol["command"][3])
prediction = Path(protocol["command"][protocol["command"].index("--answers-file") + 1])
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
if postprocessing:
    for key in ("original_state", "first_conversion_record", "first_conversion_log", "successful_conversion_record"):
        item = postprocessing[key]
        assert sha(item["path"]) == item["sha256"]
assert sha(data) == protocol["dataset"]["sha256"]
for item in protocol["source_files"]:
    assert sha(item["path"]) == item["sha256"], item["path"]

# Extract only the exact public option/null helpers; no model or GPU code imports.
tree = ast.parse(native.read_text())
helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
           and node.name in ("is_none", "get_options")]
assert len(helpers) == 2
namespace = {"math": math}
exec(compile(ast.Module(body=helpers, type_ignores=[]), str(native) + "[CPU option helpers]", "exec"), namespace)
df = pd.read_table(data)
answers = [json.loads(line) for line in prediction.read_text().splitlines() if line.strip()]
runtime_path = OUT / "observed/runtime.jsonl"
runtime = [json.loads(line) for line in runtime_path.read_text().splitlines() if line.strip()]
images = json.loads((OUT / "image_manifest.json").read_text())
image_map = {entry["index"]: entry for entry in images}
assert len(df) == len(answers) == len(runtime) == 4377
assert len({int(answer["question_id"]) for answer in answers}) == 4377

groups, failed_rows, non_single_letter, row_correct = defaultdict(list), [], [], 0
errors = []
for position, ((_, row), answer, observation) in enumerate(zip(df.iterrows(), answers, runtime)):
    idx = int(row["index"])
    options = namespace["get_options"](row, list("ABCD"))
    prompt = row["question"]
    if not namespace["is_none"](row["hint"]):
        prompt = row["hint"] + "\n" + prompt
    for letter, option in zip("ABCD", options):
        prompt += "\n" + letter + ". " + option
    checks = {
        "question_id_and_order": int(answer["question_id"]) == idx,
        "native_prompt": answer["prompt"] == prompt,
        "options": answer["options"] == options,
        "option_char": answer["option_char"] == list("ABCD")[:len(options)],
        "round_id": answer["round_id"] == 0,
        "guidance_text": observation["guidance_text"] == prompt,
        "guidance_sha": observation["guidance_text_sha256"] == hashlib.sha256(prompt.encode()).hexdigest(),
        "runtime_order": observation["call"] == position + 1,
        "observed_K": observation["returned_visual_token_num"] == 128,
        "image_shape_dtype": observation["image_shape"] == [1, 3, 336, 336] and observation["image_dtype"] == "torch.float16",
        "image_size": observation["image_sizes"] == [image_map[idx]["size"]],
        "decoding": observation["do_sample"] is False and observation["temperature"] == 0
                    and observation["max_new_tokens"] == 1024 and observation["num_beams"] == 1,
        "public_GT_letter": row["answer"] in "ABCD" and len(row["answer"]) == 1,
    }
    for name, valid in checks.items():
        if not valid:
            errors.append({"index": idx, "check": name})
    text = answer["text"].strip()
    # Independent grouping/reduction implementation of the declared local parser.
    tokens = re.findall(r"\b[A-E]\b", text)
    letter = tokens[0] if tokens else None
    correct = letter == row["answer"]
    row_correct += correct
    groups[idx % 1000000].append(correct)
    if letter is None:
        failed_rows.append({"index": idx, "text": text})
    if not re.fullmatch(r"[A-D]", text):
        non_single_letter.append({"index": idx, "text": text})

assert len(groups) == 1176
correct_groups = sum(sum(values) == len(values) for values in groups.values())
value = correct_groups * 100 / 1176
reported = json.loads((OUT / "local_circular_score.json").read_text())
assert reported["circular_correct"] == correct_groups
assert reported["n_rows"] == 4377 and reported["n_groups"] == 1176
assert reported["row_correct"] == row_correct and reported["unparsed_count"] == len(failed_rows)
assert math.isclose(reported["value"], value, rel_tol=0, abs_tol=1e-12)
assert reported["prediction_sha256"] == sha(prediction) == state["prediction_sha256"]
loaded = json.loads((OUT / "observed/loaded_model.json").read_text())
assert loaded["native_argv"] == protocol["command"][3:]
assert loaded["alpha"] == .5 and loaded["beta"] == 1 and loaded["visual_token_num"] == 128
assert loaded["model_dtype"] == "torch.float16"
dependency_equivalence = json.loads((OUT / "cpu_dependency_input_equivalence.json").read_text())
assert dependency_equivalence["success"]
assert dependency_equivalence["all_unique_images_compared"] == 1132
assert dependency_equivalence["all_CLIP_guidance_rows_compared"] == 4377
for item in dependency_equivalence["inputs"]:
    assert sha(item["path"]) == item["sha256"]
llm_dependency_equivalence = json.loads((OUT / "cpu_llm_dependency_equivalence.json").read_text())
assert llm_dependency_equivalence["success"] and llm_dependency_equivalence["n_rows"] == 4377
for item in llm_dependency_equivalence["inputs"]:
    assert sha(item["path"]) == item["sha256"]
llm_inputs = json.loads((OUT / "cpu_llm_tokens_current.json").read_text())["records"]
assert all(observation["input_shape"] == [1, row["n_tokens"]] for observation, row in zip(runtime, llm_inputs))
references = json.loads((OUT / "comparison_reference.json").read_text())
paper = references["paper"]
assert paper["method"] == "EADP" and paper["model"] == "v15" and paper["task"] == "MMB_EN" and paper["budget_nominal"] == 128
assert sha(paper["source"]) == paper["source_sha256"]
assert sha(references["reference_index"]["path"]) == references["reference_index"]["sha256"]

conversion_path = Path(postprocessing["successful_conversion_record"]["path"]) if postprocessing else OUT / "conversion.json"
conversion = json.loads(conversion_path.read_text())
assert conversion["exit_code"] == 0 and not conversion["submitted_to_external_server"]
submission = pd.read_excel(conversion["output"], keep_default_na=False)
assert submission["index"].tolist() == df["index"].tolist()
assert submission["prediction"].tolist() == [answer["text"] for answer in answers]
assert submission["answer"].tolist() == df["answer"].tolist()
result = {
    "success": not errors, "n_rows": len(df), "all_rows_checked": True, "n_circular_groups": len(groups),
    "circular_correct": correct_groups, "circular_accuracy_percent": value,
    "paper_reported": paper["value"], "local_minus_paper": value - paper["value"],
    "prior_local_V11_beta2_streamwait": references["prior_local_V11_beta2_streamwait"]["value"],
    "comparison_boundary": references["comparison_boundary"],
    "comparison_reference_sha256": sha(OUT / "comparison_reference.json"),
    "dependency_input_equivalence_sha256": sha(OUT / "cpu_dependency_input_equivalence.json"),
    "LLM_dependency_equivalence_sha256": sha(OUT / "cpu_llm_dependency_equivalence.json"),
    "isolated_CPU_postprocessing": postprocessing,
    "group_rotation_counts": dict(Counter(map(len, groups.values()))),
    "errors": errors, "unparsed_count": len(failed_rows), "unparsed": failed_rows,
    "non_single_letter_count": len(non_single_letter), "non_single_letter": non_single_letter,
    "prediction_sha256": sha(prediction), "dataset_sha256": sha(data),
    "runtime_sha256": sha(runtime_path), "submission_sha256": sha(conversion["output"]),
    "protocol_sha256": sha(OUT / "protocol.json"), "verifier_sha256": sha(__file__),
    "metric_boundary": "Local exact-letter circular accuracy with all1176groups; external official-server matching has not been run.",
    "checked": ["All4377 ordered unique IDs", "Every prompt/options/guidance text against native source helpers",
                "All public ground-truth letters", "Every runtime K/image shape/size/dtype/decoding parameters",
                "Independent circular grouping and accuracy", "Fresh source files unchanged", "Author XLSX predictions/IDs/GT"]}
(OUT / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k not in ("non_single_letter", "unparsed", "errors", "checked")}, indent=2))
assert not errors, errors[:10]
