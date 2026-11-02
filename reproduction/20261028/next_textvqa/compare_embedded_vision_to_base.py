"""Compare checkpoint vision parameters with the pinned base CLIP, on CPU only.

No model is constructed; safetensors and the base state dict are memory mapped.
The running reproduction worker and its frozen plan are untouched.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
PLAN = HERE / 'full_default_K128.plan.json'
PREFIX = 'model.vision_tower.vision_tower.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stat(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    plan = json.loads(PLAN.read_text())
    root = Path(plan['model_path'])
    index_path = root / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map']
    keys = sorted(k for k in index if k.startswith(PREFIX + 'vision_model.'))
    assert len(keys) == 391
    assert {index[k] for k in keys} == {'model-00003-of-00003.safetensors'}
    shard = root / index[keys[0]]
    shard_proof = next(r for r in plan['model_payloads'] if r['path'] == str(shard))
    base_proof = next(r for r in plan['clip_identity']['files'] if r['path'].endswith('/pytorch_model.bin'))
    base = Path(base_proof['path'])
    for path, proof in [(shard, shard_proof), (base, base_proof)]:
        assert stat(path) == {k: proof[k] for k in ('bytes', 'mtime_ns')}
    original_stats = {str(p): stat(p) for p in (shard, base)}
    assert not torch.cuda.is_initialized()
    # HF payloads were independently SHA-verified already; unchanged stats above
    # bind this tensor comparison to those bytes without another full-file read.
    state = torch.load(str(base), map_location='cpu', mmap=True, weights_only=True)
    rows = []
    with safe_open(shard, framework='pt', device='cpu') as mapped:
        for key in keys:
            base_key = key.removeprefix(PREFIX)
            assert base_key in state
            stored = mapped.get_tensor(key)
            checkpoint = stored.to(torch.float16)
            original = state[base_key]
            expected = original.to(torch.float16)
            assert stored.dtype == torch.bfloat16
            assert checkpoint.shape == original.shape
            left = checkpoint.contiguous().numpy().tobytes()
            right = expected.contiguous().numpy().tobytes()
            stored_bytes = stored.contiguous().view(torch.int16).numpy().tobytes()
            base_rounded = original.to(stored.dtype).contiguous().view(torch.int16).numpy().tobytes()
            bitwise_equal = left == right
            row = {'checkpoint_key': key, 'base_key': base_key,
                   'shape': list(checkpoint.shape), 'elements': checkpoint.numel(),
                   'checkpoint_stored_dtype': str(stored.dtype), 'base_dtype': str(original.dtype),
                   'comparison_dtype': str(expected.dtype), 'bitwise_equal': bitwise_equal,
                   'checkpoint_tensor_sha256': hashlib.sha256(left).hexdigest(),
                   'base_cast_fp16_tensor_sha256': hashlib.sha256(right).hexdigest(),
                   'checkpoint_stored_sha256': hashlib.sha256(stored_bytes).hexdigest(),
                   'base_cast_stored_bf16_sha256': hashlib.sha256(base_rounded).hexdigest(),
                   'stored_equals_base_cast_bf16': stored_bytes == base_rounded}
            if not bitwise_equal:
                row['unequal_values'] = int(torch.count_nonzero(checkpoint != expected))
                row['max_abs_difference_fp32'] = float((checkpoint.float() - expected.float()).abs().max())
            rows.append(row)
    assert not torch.cuda.is_initialized()
    assert original_stats == {str(p): stat(p) for p in (shard, base)}
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(),
              'CPU_only': True, 'model_constructed': False, 'GPU_initialized': False,
              'torch': torch.__version__, 'worker_or_plan_modified': False,
              'comparison': 'All391 embedded BF16 vision_model parameters cast to actual runtime FP16 vs pinned official base CLIP cast to FP16, byte for byte; also stored BF16 vs base rounded to BF16',
              'plan': {'path': str(PLAN), 'sha256': sha(PLAN)},
              'script_sha256': sha(__file__), 'index_sha256': sha(index_path),
              'checkpoint_payload': shard_proof, 'base_CLIP_payload': base_proof,
              'payload_hash_scope': 'Reuse independently verified complete-file SHA with matching current size+mtime; per-tensor bytes newly compared and hashed',
              'tensor_count': len(rows), 'element_count': sum(r['elements'] for r in rows),
              'equal_tensor_count': sum(r['bitwise_equal'] for r in rows),
              'different_tensor_count': sum(not r['bitwise_equal'] for r in rows),
              'all_bitwise_equal_after_base_fp16_cast': all(r['bitwise_equal'] for r in rows),
              'stored_equals_base_bf16_tensor_count': sum(r['stored_equals_base_cast_bf16'] for r in rows),
              'stored_differs_from_base_bf16_tensor_count': sum(not r['stored_equals_base_cast_bf16'] for r in rows),
              'elapsed_seconds': time.monotonic() - started, 'tensors': rows,
              'limits': ['Parameter identity does not establish hidden-state precision equivalence or the cause of any accuracy gap.',
                         'This compares the vision body only; separate text tower, visual projection and projector are outside this requested comparison.']}
    path = HERE / 'embedded_vision_vs_official_base.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('tensor_count', 'element_count', 'equal_tensor_count',
                                           'different_tensor_count', 'all_bitwise_equal_after_base_fp16_cast',
                                           'stored_equals_base_bf16_tensor_count', 'stored_differs_from_base_bf16_tensor_count',
                                           'elapsed_seconds', 'GPU_initialized')}), flush=True)


if __name__ == '__main__':
    main()
