"""Synthetic PID-1 worker: no torch, CUDA, model or GPU access."""
import argparse
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'e4_c_memory_runtime'))
from worker_barrier import handshake


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True, type=Path)
    p.add_argument('--nonce', required=True)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('synthetic worker requires CUDA_VISIBLE_DEVICES empty')
    handshake(a.directory, a.nonce, lambda: False, timeout=60)
    # Give the fake source time to emit a sample after release.
    time.sleep(0.8)
    with (a.directory / 'fake_exiting').open('x') as f:
        f.write('synthetic worker exiting\n')
        f.flush()
        os.fsync(f.fileno())
    time.sleep(0.3)


if __name__ == '__main__':
    main()
