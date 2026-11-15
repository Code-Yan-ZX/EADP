"""Small immutable-artifact helpers shared by the experiment entry points."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value, frozen=False):
    path = Path(path)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Frozen artifact differs: {path}")
        return
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if frozen:
            # Atomic publish without replacing a concurrent frozen artifact.
            try:
                os.link(name, path)
            except FileExistsError:
                if path.read_text(encoding="utf-8") != encoded:
                    raise RuntimeError(f"Concurrent frozen artifact differs: {path}")
        else:
            os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_torch_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_protocol(run_dir):
    path = Path(run_dir).resolve() / "protocol.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    for source, digest in value["source_sha256"].items():
        if sha256(source) != digest:
            raise RuntimeError(f"Source changed since protocol freeze: {source}; use a new run directory")
    if sha256(Path(run_dir) / "data/manifest.json") != value["manifest_sha256"]:
        raise RuntimeError("Data manifest changed since protocol freeze")
    for model_file, digest in value.get("model_metadata_sha256", {}).items():
        if sha256(model_file) != digest:
            raise RuntimeError(f"Frozen tokenizer/model metadata changed: {model_file}")
    for model_file, expected in value.get("model_weight_stat", {}).items():
        stat = Path(model_file).stat()
        if {"size":stat.st_size, "mtime_ns":stat.st_mtime_ns} != expected:
            raise RuntimeError(f"Frozen model weight file changed: {model_file}")
    if "environment" in value:
        import torch
        import transformers
        from PIL import __version__ as pillow_version
        for name, actual in (("torch",torch.__version__), ("transformers",transformers.__version__),
                             ("pillow",pillow_version)):
            if value["environment"].get(name) != actual:
                raise RuntimeError(f"Runtime {name} differs from protocol")
    return value
