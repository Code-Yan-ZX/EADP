"""Run the isolated author Python with a compatible libc, using host GPU drivers.

The image supplies libc and system libraries only. Python, Torch, Transformers,
FlashAttention and the EADP source all come from the frozen host paths. No global
Docker runtime, host driver, original environment, or author source is modified.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PYTHON = ROOT / "_reproduction/envs/qwen_author/bin/python"
AUTHOR = ROOT / "_reproduction/author_eadp/Qwen_vl"
MODEL_ROOT = Path("/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct")
MODEL = MODEL_ROOT / "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
IMAGE = "sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf"


def command(args, *, gpu=False, name=None, extra_env=None):
    cmd = ["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL",
           "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}"]
    if name:
        cmd += ["--name", name]
    for source in (Path("/home/dell/miniconda3"), ROOT, MODEL_ROOT, Path("/media/disk2/YZX/LMUData")):
        suffix = "" if source == ROOT else ",readonly"
        cmd += ["--mount", f"type=bind,src={source},dst={source}{suffix}"]
    env = {
        "LMUData": "/media/disk2/YZX/LMUData",
        "HF_HOME": "/tmp/eadp_hf", "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false",
        "MPLCONFIGDIR": "/tmp/eadp_matplotlib", "XDG_CACHE_HOME": "/tmp/eadp_cache",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TZ": "Asia/Shanghai",
        "QWEN3_VL_8B_MODEL_PATH": str(MODEL),
        "PYTHON": str(PYTHON), "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if gpu:
        for device in ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"):
            cmd += ["--device", device]
        for soname in ("libcuda.so.1", "libnvidia-ml.so.1", "libnvidia-ptxjitcompiler.so.1"):
            source = (Path("/usr/lib/x86_64-linux-gnu") / soname).resolve(strict=True)
            cmd += ["--mount", f"type=bind,src={source},dst=/usr/local/nvidia/lib64/{soname},readonly"]
        cmd += ["--mount", "type=bind,src=/usr/bin/nvidia-smi,dst=/usr/local/nvidia/bin/nvidia-smi,readonly"]
        env["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env.update(extra_env or {})
    for key, value in env.items():
        cmd += ["--env", f"{key}={value}"]
    return cmd + ["--workdir", str(AUTHOR), "--entrypoint", str(PYTHON), IMAGE] + list(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--name")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    opts = parser.parse_args()
    args = opts.args[1:] if opts.args and opts.args[0] == "--" else opts.args
    cmd = command(args, gpu=opts.gpu_authorized, name=opts.name)
    opts.record.parent.mkdir(parents=True, exist_ok=True)
    record = {"command": cmd, "gpu_authorized": opts.gpu_authorized, "status": "running"}
    opts.record.write_text(json.dumps(record, indent=2) + "\n")
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    record.update(status="complete" if result.returncode == 0 else "failed",
                  returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    opts.record.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("status", "returncode", "stdout", "stderr")}, indent=2))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
