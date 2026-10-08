"""CPU gate adapter for one frozen worker's JSON count-key representation.

The original validator still reads and verifies every prediction, runtime row,
native protocol and official score. Only its returned count dictionary keys
are converted with str(k), matching JSON object-key semantics. No score value,
production model, frozen file, generated artifact or queue state is changed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import hashlib
import importlib.util
from pathlib import Path
import runpy
import sys

ROOT = Path('/media/disk2/YZX/research/EADP_amp')
OUT = ROOT / 'Qwen_vl/outputs/audit_followup_20261008'
FROZEN_WORKER = OUT / 'next_gap_diagnosis_20261008/stability/next_text_full_streamwait.py'
FROZEN_WORKER_SHA256 = '80affa9c875fb25c5f78cea39a437efc81b36a9b8383a19e77166516f58cda63'
PYTHON = Path('/home/dell/miniconda3/envs/llava_pruner/bin/python')
TARGET_SHA256 = {
    ROOT / 'Qwen_vl/scripts/stage1_roundtrip_pilot/publish_repair_results.py':
        'ae9fc06bcb69dc20c2e2987badda79a95474019fa38c1ef58f9a7eed7a477d57',
    OUT / 'streamwait_continuation_20261008/schedule_streamwait_repair_continuation.py':
        '30296dc9ee8e3e139e666b1c7e79ef9783ef968588c594e3f19a7dee8eb74c0a',
    OUT / 'streamwait_continuation_20261008/schedule_legacy_streamwait_continuation.py':
        '82a7ce81194bf15b0f4d898b23f14a68ef88a8756b8c81ab25e18b94bcd77f77',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_worker_identity():
    if sha256(FROZEN_WORKER) != FROZEN_WORKER_SHA256:
        raise ValueError('Frozen full-control worker SHA differs; overlay refused')


def normalize_count_keys(report):
    """Copy the return mapping and change only actual-count dictionary keys."""
    distribution = report['actual_count_distribution']
    normalized = {str(key): count for key, count in distribution.items()}
    if len(normalized) != len(distribution):
        raise ValueError('Count-key normalization would lose a key; overlay refused')
    result = dict(report)
    result['actual_count_distribution'] = normalized
    return result


class _PinnedLoader:
    def __init__(self, original_loader):
        self.original_loader = original_loader

    def create_module(self, spec):
        return self.original_loader.create_module(spec)

    def exec_module(self, module):
        check_worker_identity()
        self.original_loader.exec_module(module)
        check_worker_identity()
        original = module.validate_and_score
        if not callable(original):
            raise ValueError('Frozen worker validator is not callable; overlay refused')

        @functools.wraps(original)
        def validate_and_score(*args, **kwargs):
            check_worker_identity()
            return normalize_count_keys(original(*args, **kwargs))

        validate_and_score._json_count_distribution_only_overlay = {
            'worker_path': str(FROZEN_WORKER),
            'worker_sha256': FROZEN_WORKER_SHA256,
            'changed_return_field': 'actual_count_distribution',
            'changed_values': False,
        }
        module.validate_and_score = validate_and_score

    def __getattr__(self, name):
        return getattr(self.original_loader, name)


@contextmanager
def installed_json_count_overlay():
    """Intercept only the exact frozen worker's explicit file-module loader."""
    check_worker_identity()
    original = importlib.util.spec_from_file_location

    def spec_from_file_location(name, location=None, *args, **kwargs):
        spec = original(name, location, *args, **kwargs)
        if spec is not None and spec.origin is not None:
            if Path(spec.origin).resolve() == FROZEN_WORKER.resolve():
                check_worker_identity()
                if spec.loader is None:
                    raise ValueError('Frozen worker has no loader; overlay refused')
                spec.loader = _PinnedLoader(spec.loader)
        return spec

    importlib.util.spec_from_file_location = spec_from_file_location
    try:
        yield
    finally:
        importlib.util.spec_from_file_location = original


def checked_target(value):
    supplied = Path(value)
    if not supplied.is_absolute():
        raise ValueError('An absolute original CPU entry path is required')
    target = supplied.resolve()
    if target not in TARGET_SHA256:
        raise ValueError('Original CPU entry is outside the three-path allowlist')
    if sha256(target) != TARGET_SHA256[target]:
        raise ValueError('Original CPU entry SHA differs; launcher refused')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True)
    parser.add_argument('original_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10) or Path(sys.executable).resolve() != PYTHON.resolve():
        raise ValueError('Use the unchanged llava_pruner Python 3.10 interpreter')
    target = checked_target(args.target)
    original_args = args.original_args
    if original_args[:1] == ['--']:
        original_args = original_args[1:]
    previous_argv = sys.argv
    sys.argv = [str(target), *original_args]
    try:
        with installed_json_count_overlay():
            runpy.run_path(str(target), run_name='__main__')
    finally:
        sys.argv = previous_argv


if __name__ == '__main__':
    main()
