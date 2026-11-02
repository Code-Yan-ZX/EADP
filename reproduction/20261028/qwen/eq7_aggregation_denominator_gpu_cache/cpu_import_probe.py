"""Import-only host probe; never asks whether CUDA devices are available."""
import importlib.metadata
import json
import os
import sys

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import torch
assert not torch.cuda.is_initialized()
print(json.dumps(dict(torch_version=torch.__version__,torch_metadata_version=importlib.metadata.version('torch'),
    torch_cuda_build=torch.version.cuda,torch_module=torch.__file__,default_dtype=str(torch.get_default_dtype()),
    autocast=torch.is_autocast_enabled(),matmul_TF32=torch.backends.cuda.matmul.allow_tf32,
    cudnn_TF32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
    cudnn_deterministic=torch.backends.cudnn.deterministic,CUDA_initialized=torch.cuda.is_initialized(),
    forbidden_imports=[n for n in sys.modules if n.startswith(('transformers','vlmeval','flash_attn'))])))
