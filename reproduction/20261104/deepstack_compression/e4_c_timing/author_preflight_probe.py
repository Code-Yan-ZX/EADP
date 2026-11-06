"""Run actual author message-builder CPU regression in the frozen container."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import test_runner
if torch.cuda.is_initialized():raise ValueError('CUDA initialized in CPU-only probe')
unittest.main(module=test_runner,defaultTest='AuthorPreflightTests',verbosity=2)
