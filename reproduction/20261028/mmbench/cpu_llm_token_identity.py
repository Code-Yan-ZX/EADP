#!/usr/bin/env python3
"""Fingerprint all4377 native v15 MMBench LLM prompt IDs without CUDA."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import pandas as pd
import sentencepiece
import torch
from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parent
AUTHOR = OUT.parents[2] / "_reproduction/author_eadp"
sys.path.insert(0, str(AUTHOR / "LLaVA"))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--expect-author-pins", action="store_true")
args = parser.parse_args()
protobuf = importlib.metadata.version("protobuf")
if args.expect_author_pins:
    assert sentencepiece.__version__ == "0.1.99" and protobuf == "3.20.3"
    assert "vendor_author_pins" in sentencepiece.__file__
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
native = AUTHOR / "LLaVA/llava/eval/model_vqa_mmbench.py"
tree = ast.parse(native.read_text())
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("is_none", "get_options")]
namespace = {"math": math}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(native) + "[CPU original helpers]", "exec"), namespace)
model = Path("/media/disk2/YZX/doct/FastV/llava-v1.5-7b")
config = json.loads((model / "config.json").read_text())
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, local_files_only=True)
df = pd.read_table(OUT / "data/mmbench_dev_20230712.tsv")
records = []
for _, row in df.iterrows():
    question = row["question"]
    if not namespace["is_none"](row["hint"]):
        question = row["hint"] + "\n" + question
    for letter, option in zip("ABCD", namespace["get_options"](row, list("ABCD"))):
        question += "\n" + letter + ". " + option
    image = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN if config.get("mm_use_im_start_end") else DEFAULT_IMAGE_TOKEN
    qs = image + "\n" + question + "\nAnswer with the option's letter from the given choices directly."
    conversation = conv_templates["vicuna_v1"].copy()
    conversation.append_message(conversation.roles[0], qs)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    records.append({"index": int(row["index"]), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "n_tokens": ids.shape[-1], "token_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest()})
assert len(records) == 4377 and not torch.cuda.is_initialized()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
report = {"sentencepiece": sentencepiece.__version__, "sentencepiece_module": sentencepiece.__file__,
          "protobuf": protobuf, "tokenizer_class": type(tokenizer).__name__, "records": records,
          "cuda_initialized": False, "native_source_sha256": sha(native),
          "tokenizer_model_sha256": sha(model / "tokenizer.model"), "fingerprint_script_sha256": sha(__file__),
          "scope": "All4377 oldEN20230712 native MMB prompts including hints/options, image marker, single_pred_prompt suffix and vicuna_v1; native slow tokenizer_image_token IDs."}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"sentencepiece": report["sentencepiece"], "protobuf": protobuf, "n": len(records), "cuda_initialized": False}))
