"""Small, predeclared runtime check of the unchanged author Qwen EADP wrapper.

This is a diagnostic, not a benchmark score. Full runs use the public launcher.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.metadata
import inspect
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHOR = ROOT / "_reproduction/author_eadp/Qwen_vl"
MODEL_NAME = "Qwen3-VL-8B-EADP-256-a0.5-b2.0"
DATASETS = {"DocVQA_VAL": 5349, "AI2D_TEST": 3088}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-authorized", action="store_true")
    opts = parser.parse_args()
    sys.path.insert(0, str(AUTHOR / "VLMEvalKit"))
    import torch
    from vlmeval.config import supported_VLM
    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP
    recipe = supported_VLM[MODEL_NAME]
    assert recipe.func is Qwen3VLChatEADP
    assert (recipe.keywords["visual_token_num"], recipe.keywords["alpha"], recipe.keywords["beta"]) == (256, .5, 2.)
    assert recipe.keywords["max_new_tokens"] == 2048
    assert recipe.keywords["use_vllm"] is False
    config_path = AUTHOR / "VLMEvalKit/vlmeval/config.py"
    sources = [config_path, AUTHOR / "model/pruner.py"]
    sources += [AUTHOR / "VLMEvalKit/vlmeval/vlm/qwen3_vl" / f for f in ("model.py", "model_fixed_res.py", "prompt.py")]
    sources += [AUTHOR / "VLMEvalKit/vlmeval" / f for f in ("inference.py", "dataset/image_base.py")]
    spec = {
        "model_name": MODEL_NAME, "recipe": recipe.keywords,
        "fixed_selection_rule": "first row and middle row of each complete canonical dataset; no selection by answers",
        "source_sha256": {str(p): sha(p) for p in sources},
        "wrapper_source": inspect.getfile(recipe.func),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "transformers", "flash-attn", "accelerate", "qwen-vl-utils",
            "Pillow", "tokenizers", "numpy", "safetensors", "timm")},
        "CUDA_initialized_before_model": torch.cuda.is_initialized(),
        "datasets": {}, "purpose": "Author-wrapper/runtime diagnostic only; no benchmark accuracy claimed.",
    }
    datasets = {}
    for name, expected in DATASETS.items():
        ds = build_dataset(name)
        assert len(ds) == expected and ds.data["index"].is_unique
        datasets[name] = ds
        positions = [0, len(ds) // 2]
        spec["datasets"][name] = {
            "n": len(ds), "positions": positions,
            "indices": [str(ds.data.iloc[p]["index"]) for p in positions],
            "data_path": ds.data_path, "data_sha256": sha(ds.data_path),
        }
    if not opts.gpu_authorized:
        assert not torch.cuda.is_initialized()
        save(HERE / "source_probe_plan.json", spec)
        print(json.dumps({"prepared": True, "CUDA_initialized": False, "datasets": spec["datasets"]}))
        return
    expected = json.loads((HERE / "source_probe_plan.json").read_text())
    assert spec == expected, "Source/versions/data/protocol changed after preparation"
    result_dir = HERE / "author_fa2_smoke"
    result_dir.mkdir(exist_ok=False)
    save(result_dir / "state.json", {"status": "loading", "pid": os.getpid(),
         "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    model = recipe()
    assert isinstance(model, Qwen3VLChatEADP)
    assert model.model.config._attn_implementation == "flash_attention_2"
    devices = {str(p.device) for p in model.model.parameters()}
    assert devices == {"cuda:0"}, devices
    runtime = {
        "torch_cuda": torch.version.cuda, "CUDA_initialized": torch.cuda.is_initialized(),
        "gpu": torch.cuda.get_device_name(0), "model_dtype": str(model.model.dtype),
        "devices": sorted(devices), "attention_backend": model.model.config._attn_implementation,
        "generate_kwargs": model.generate_kwargs,
        "module_files": {k: getattr(sys.modules[k], "__file__", None) for k in (
            "vlmeval.config", "vlmeval.vlm.qwen3_vl.model", "vlmeval.vlm.qwen3_vl.model_fixed_res", "model.pruner")},
    }
    save(result_dir / "runtime.json", runtime)
    with (result_dir / "answers.jsonl").open("w") as output:
        for name, ds in datasets.items():
            model.set_dump_image(ds.dump_image)
            for position in spec["datasets"][name]["positions"]:
                row = ds.data.iloc[position]
                custom = hasattr(model, "use_custom_prompt") and model.use_custom_prompt(name)
                message = model.build_prompt(row, dataset=name) if custom else ds.build_prompt(row)
                images = {p["value"]: sha(p["value"]) for p in message if p["type"] == "image"}
                answer = model.generate(message=message, dataset=name)
                assert isinstance(answer, str) and answer.strip()
                record = {"dataset": name, "position": position, "index": str(row["index"]),
                          "message": message, "image_sha256": images, "prediction": answer}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(json.dumps({"dataset": name, "position": position, "prediction": answer}), flush=True)
    assert spec["source_sha256"] == {str(p): sha(p) for p in sources}
    save(result_dir / "state.json", {"status": "complete", "n": 4,
         "answers_sha256": sha(result_dir / "answers.jsonl"), "source_unchanged": True,
         "purpose": spec["purpose"], "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
