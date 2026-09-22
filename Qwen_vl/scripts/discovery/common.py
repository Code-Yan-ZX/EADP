"""
Shared plumbing for EADP method-discovery diagnostics.

This module deliberately does not modify the official implementation in
``model/pruner.py``. Everything diagnostic lives here.

Timing convention (matches the EADP paper's Table 14 framing):
  * every stage is bracketed by ``torch.cuda.Event`` pairs recorded on the
    default stream, and all events are synchronised once at the end of the
    pruning call, so we measure GPU time without perturbing the pipeline;
  * the vision tower (``model.visual``) is *not* part of "pruning overhead" --
    it is identical for every method and for the unpruned baseline.
"""

from __future__ import annotations

import gc
import os
import sys

import numpy as np
import torch


# --------------------------------------------------------------------------
# Path / environment bootstrap. Must run before importing vlmeval.
# --------------------------------------------------------------------------
DISCOVERY_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(DISCOVERY_DIR)
QWEN_ROOT = os.path.dirname(SCRIPTS_DIR)
PROJECT_ROOT = os.path.dirname(QWEN_ROOT)
VLMEVALKIT = os.path.join(QWEN_ROOT, "VLMEvalKit")

for _p in (VLMEVALKIT, QWEN_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# No flash-attn wheel exists for this torch/CUDA combo; SDPA is numerically
# identical, so EADP importance scores are unaffected.
os.environ.setdefault("QWEN3_VLM_ATTN_IMPL", "sdpa")
os.environ.setdefault("LMUData", "/media/disk2/YZX/LMUData")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", "/media/disk2/YZX/hf-datasets-cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("QWEN3_VL_8B_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
# The local proxy is unreliable, and HF otherwise revalidates cached files with
# a HEAD request that can fail mid-run. Everything we need is already cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# vlmeval's config module reads this at import time.
os.environ.pop("WORLD_SIZE", None)

OUTPUT_DIR = os.path.join(QWEN_ROOT, "outputs", "discovery")

BASELINE_MODEL = "Qwen3-VL-8B-Instruct-1024"

# The vision tower / merged-token geometry for 1024x1024 Qwen3-VL:
# patch 16, merge 2  ->  64x64 patches  ->  32x32 = 1024 merged visual tokens.
N_MERGED_TOKENS = 1024
GRID_HW = 32


def eadp_model_name(tokens: int, alpha: float = 0.5, beta: float = 2.0) -> str:
    return f"Qwen3-VL-8B-EADP-{tokens}-a{alpha}-b{beta}"


def ensure_out_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


# --------------------------------------------------------------------------
# Model / dataset construction
# --------------------------------------------------------------------------
def load_model(model_name: str, max_new_tokens: int = 128):
    """Instantiate a VLMEvalKit model wrapper. Uses the official config."""
    from vlmeval.config import supported_VLM

    if model_name not in supported_VLM:
        raise KeyError(f"{model_name} is not registered in vlmeval.config")
    model = supported_VLM[model_name](max_new_tokens=max_new_tokens)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "model"):
        model.model.eval()
    if hasattr(model, "max_new_tokens"):
        model.max_new_tokens = max_new_tokens
    if hasattr(model, "generate_kwargs"):
        model.generate_kwargs["max_new_tokens"] = max_new_tokens
    return model


def free_model(model):
    try:
        del model
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def build_dataset(name: str):
    from vlmeval.dataset import build_dataset as _build

    dataset = _build(name)
    if dataset is None:
        raise RuntimeError(f"could not build dataset {name}")
    return dataset


def build_message(model, dataset, dataset_name: str, row):
    if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
        return model.build_prompt(row, dataset=dataset_name)
    return dataset.build_prompt(row)


def sample_indices(n_total: int, n_want: int, offset: int = 0) -> list:
    """Evenly spaced, deterministic sample selection (reproducible across runs)."""
    n_want = min(n_want, n_total)
    if n_want <= 0:
        return []
    idx = np.linspace(0, n_total - 1, n_want).round().astype(int)
    idx = (idx + offset) % n_total
    return sorted(set(int(i) for i in idx))


def collect_samples(model, dataset_name: str, n_samples: int, offset: int = 0):
    """Return (dataset, rows, messages) for a deterministic sample of a split."""
    dataset = build_dataset(dataset_name)
    if hasattr(model, "set_dump_image"):
        model.set_dump_image(dataset.dump_image)
    idx = sample_indices(len(dataset.data), n_samples, offset=offset)
    rows = [dataset.data.iloc[i] for i in idx]
    messages = [build_message(model, dataset, dataset_name, r) for r in rows]
    return dataset, idx, rows, messages


# --------------------------------------------------------------------------
# Low-level timing helpers
# --------------------------------------------------------------------------
class CudaTimer:
    """Bracket GPU work with CUDA events; read results after a single sync."""

    def __init__(self):
        self._events = []

    def start(self, name: str):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._events.append((name, ev, None))

    def stop(self, name: str):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        for i in range(len(self._events) - 1, -1, -1):
            n, s, e = self._events[i]
            if n == name and e is None:
                self._events[i] = (n, s, ev)
                return
        raise KeyError(f"stop() without matching start() for {name!r}")

    def finish(self) -> dict:
        torch.cuda.synchronize()
        out = {}
        for name, s, e in self._events:
            if e is None:
                continue
            out[name] = out.get(name, 0.0) + s.elapsed_time(e)
        self._events = []
        return out


def time_callable(fn, repeat: int, warmup: int = 0) -> tuple:
    """Wall-clock-free GPU timing of `fn` with warmup; returns (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.mean(times)), float(np.std(times))


def peak_memory_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# --------------------------------------------------------------------------
# Analytic FLOPs for the Qwen3-VL language-model prefill
# --------------------------------------------------------------------------
def analytic_prefill_flops_g(model_or_config, seq_len: int) -> float:
    """
    FLOPs (G) for one prefill pass of the LLM decoder over `seq_len` positions.

    Counts, per decoder layer:
      * qkv + o projections : 2 * seq * d * (d + 2*d_kv) + 2 * seq * d * d
      * MLP (SwiGLU)        : 2 * seq * d * 3 * ffn
      * attention scores    : 2 * 2 * n_heads * seq^2 * head_dim
      * attention @ V       : 2 * 2 * n_heads * seq^2 * head_dim
    plus the final LM head  : 2 * seq * d * vocab

    This is the standard "2 * params * tokens" linear-term count plus the
    quadratic attention term; it deliberately ignores norms/activations.
    """
    cfg = getattr(model_or_config, "config", model_or_config)
    text_cfg = getattr(cfg, "text_config", None) or cfg

    d = int(getattr(text_cfg, "hidden_size"))
    n_layers = int(getattr(text_cfg, "num_hidden_layers"))
    n_heads = int(getattr(text_cfg, "num_attention_heads"))
    n_kv = int(getattr(text_cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(text_cfg, "head_dim", d // n_heads))
    ffn = int(getattr(text_cfg, "intermediate_size"))
    vocab = int(getattr(text_cfg, "vocab_size"))

    lin = 2 * seq_len * d * (d + 2 * n_kv * head_dim) + 2 * seq_len * d * d
    mlp = 2 * seq_len * d * 3 * ffn
    attn_qk = 2 * 2 * n_heads * seq_len * seq_len * head_dim
    attn_av = 2 * 2 * n_heads * seq_len * seq_len * head_dim
    head = 2 * seq_len * d * vocab
    total = n_layers * (lin + mlp + attn_qk + attn_av) + head
    return total / 1e9


def generated_token_count(model, message, dataset_name: str) -> int:
    """Number of tokens the model actually emits (for latency normalisation)."""
    with torch.no_grad():
        inputs = model._processor_inputs(model._build_messages(message, dataset=dataset_name))
        out = model.model.generate(**inputs, do_sample=False, **model.generate_kwargs)
    return int(out.shape[1] - inputs["input_ids"].shape[1])
