#!/usr/bin/env python3
"""Fingerprint every old-EN MMB input under installed or isolated author pins."""
import argparse
import ast
import base64
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import PIL
from PIL import Image
import tokenizers
import torch
import transformers
from transformers import CLIPImageProcessor, CLIPTokenizerFast

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
AUTHOR = ROOT / "_reproduction/author_eadp"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--expect-author-pins", action="store_true")
args = parser.parse_args()
if args.expect_author_pins:
    assert PIL.__version__ == "10.4.0" and tokenizers.__version__ == "0.15.1"
    assert "vendor_author_pins" in PIL.__file__ and "vendor_author_pins" in tokenizers.__file__
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
fingerprint = lambda array: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()

utils = AUTHOR / "LLaVA/llava/mm_utils.py"
native = AUTHOR / "LLaVA/llava/eval/model_vqa_mmbench.py"
namespace = {"Image": Image, "torch": torch, "math": math}
for path, names in ((utils, ("expand2square", "process_images")),
                    (native, ("is_none", "get_options"))):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path) + "[CPU original helpers]", "exec"), namespace)
config_path = Path("/media/disk2/YZX/doct/FastV/llava-v1.5-7b/config.json")
config = SimpleNamespace(**json.loads(config_path.read_text()))
assert config.image_aspect_ratio == "pad"
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336", local_files_only=True)
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14-336", local_files_only=True)
max_position_embeddings = 77
data = OUT / "data/mmbench_dev_20230712.tsv"
df = pd.read_table(data)
assert len(df) == 4377
images, guidance = {}, []
for _, row in df.iterrows():
    payload = base64.b64decode(row["image"])
    payload_sha = hashlib.sha256(payload).hexdigest()
    if payload_sha not in images:
        image = Image.open(io.BytesIO(payload))
        image.load()
        rgb = image.convert("RGB")
        tensor = namespace["process_images"]([image], processor, config)[0].unsqueeze(0)
        assert list(tensor.shape) == [1, 3, 336, 336] and tensor.dtype == torch.float32
        images[payload_sha] = {"size": list(image.size), "mode": image.mode,
                               "decoded_rgb_sha256": fingerprint(np.asarray(rgb)),
                               "float32_sha256": fingerprint(tensor.numpy()),
                               "float16_sha256": fingerprint(tensor.half().numpy()), "shape": list(tensor.shape)}
    prompt = row["question"]
    if not namespace["is_none"](row["hint"]):
        prompt = row["hint"] + "\n" + prompt
    for letter, option in zip("ABCD", namespace["get_options"](row, list("ABCD"))):
        prompt += "\n" + letter + ". " + option
    inputs = tokenizer(text=prompt, return_tensors="pt")
    segments = (inputs.input_ids.shape[1] - 1) // max_position_embeddings + 1
    padding = max_position_embeddings * segments - inputs.input_ids.shape[1]
    padded = {k: torch.cat([v, v.new_zeros((v.shape[0], padding))], dim=1).reshape(-1, max_position_embeddings)
              for k, v in inputs.items()}
    guidance.append({"index": int(row["index"]), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "image_payload_sha256": payload_sha,
                     "original_token_shape": list(inputs.input_ids.shape),
                     "padded_shape": list(padded["input_ids"].shape),
                     "input_ids_sha256": fingerprint(padded["input_ids"].numpy()),
                     "attention_mask_sha256": fingerprint(padded["attention_mask"].numpy())})

result = {"versions": {"Pillow": PIL.__version__, "PIL_module": PIL.__file__,
                        "tokenizers": tokenizers.__version__, "tokenizers_module": tokenizers.__file__,
                        "torch": torch.__version__, "transformers": transformers.__version__},
          "n_rows": len(df), "unique_image_payloads": len(images), "all_guidance_rows": len(guidance),
          "dataset_sha256": sha(data), "model_config_sha256": sha(config_path),
          "native_source_sha256": sha(native), "image_helper_source_sha256": sha(utils),
          "fingerprint_script_sha256": sha(__file__), "images": images, "guidance": guidance,
          "scope": "CPU exact native pad image preprocessing and CLIP tokenizer+segment/padding inputs only; no model inference."}
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k not in ("images", "guidance")}, indent=2))
