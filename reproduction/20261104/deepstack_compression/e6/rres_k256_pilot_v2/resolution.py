"""Scoped author fixed-resolution override for exactly one processor call."""
from contextlib import contextmanager
@contextmanager
def fixed_resolution(module,value):
 if type(value) is not int or value not in (512,1024) or module.QWEN3_FIXED_RESOLUTION!=1024:raise ValueError('unexpected resolution/global state')
 original=module.QWEN3_FIXED_RESOLUTION;module.QWEN3_FIXED_RESOLUTION=value
 try:yield
 finally:module.QWEN3_FIXED_RESOLUTION=original
