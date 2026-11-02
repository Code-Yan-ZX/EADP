#!/usr/bin/env python3
"""Observe the author's unmodified native entry point after model load/generation."""
import functools
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
import time

import torch
import transformers
import llava.model.builder as builder

OUT = Path(sys.argv[1]).resolve()
NATIVE = Path(sys.argv[2]).resolve()
NATIVE_ARGV = [str(NATIVE)] + sys.argv[3:]
OUT.mkdir(parents=True, exist_ok=True)
real_load = builder.load_pretrained_model


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def observed_load(*args, **kwargs):
    result = real_load(*args, **kwargs)
    tokenizer, model, image_processor, context_len = result
    tower = model.get_vision_tower()
    details = {"native_argv": NATIVE_ARGV, "python": sys.version, "torch": torch.__version__,
               "transformers": transformers.__version__, "model_class": type(model).__name__,
               "model_class_source": inspect.getfile(type(model)), "model_dtype": str(model.dtype),
               "model_device": str(model.device), "model_config": model.config.to_dict(),
               "visual_token_num": model.visual_token_num, "alpha": model.alpha, "beta": model.beta,
               "clip_class_source": inspect.getfile(type(tower)), "clip_dtype": str(tower.dtype),
               "clip_device": str(tower.device), "image_processor": image_processor.to_dict(),
               "context_len": context_len, "observer_math_changes": False,
               "pre_generation_tensor_hashing_or_synchronization": False}
    (OUT / "loaded_model.json").write_text(json.dumps(details, indent=2, ensure_ascii=False, default=str) + "\n")
    real_generate = model.generate
    calls = 0

    @functools.wraps(real_generate)
    def observed_generate(*generate_args, **generate_kwargs):
        nonlocal calls
        start = time.monotonic()
        answer = real_generate(*generate_args, **generate_kwargs)
        calls += 1
        images = generate_kwargs.get("images")
        texts = generate_kwargs.get("texts", "")
        record = {"call": calls, "seconds": time.monotonic() - start,
                  "input_shape": list(generate_args[0].shape),
                  "image_shape": list(images.shape) if images is not None else None,
                  "image_dtype": str(images.dtype) if images is not None else None,
                  "image_sizes": generate_kwargs.get("image_sizes"),
                  "guidance_text_sha256": sha(texts), "guidance_text": texts,
                  "do_sample": generate_kwargs.get("do_sample"),
                  "temperature": generate_kwargs.get("temperature"),
                  "num_beams": generate_kwargs.get("num_beams"),
                  "max_new_tokens": generate_kwargs.get("max_new_tokens"),
                  "returned_visual_token_num": int(answer[1]),
                  "output_shape": list(answer[0].shape)}
        with (OUT / "runtime.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return answer

    model.generate = observed_generate
    return result


builder.load_pretrained_model = observed_load
sys.argv = NATIVE_ARGV
runpy.run_path(str(NATIVE), run_name="__main__")
