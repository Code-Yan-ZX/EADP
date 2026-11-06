"""Exercise the real sampler state machine with fabricated CSV, never NVML."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'e4_c_memory'))
from host_evidence import process_alive
from sampler_io import sample_only


def fake_query(expected, directory, alive=process_alive):
    """Return fabricated target after release while process identity is live."""
    def query(argv, **kwargs):
        if argv != ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits']:
            raise ValueError('unexpected fake query')
        value = ''
        if (directory / 'release.json').exists() and not (directory / 'fake_exiting').exists() and alive(expected):
            value = f"{expected['gpu_uuid']}, {expected['host_pid']}, 1\n"
        return SimpleNamespace(stdout=value)
    return query


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    a = p.parse_args()
    d = a.directory.resolve()
    e = json.loads((d / 'expected.json').read_text())
    result = sample_only(e, d/'samples.jsonl', d/'sampler_ready.json',
                         d/'stopped.json', d/'stopped_inspect.json',
                         100_000_000, 90_000_000_000,
                         run=fake_query(e, d))
    if result.get('status') != 'completed':
        raise ValueError('fake sampler incomplete')


if __name__ == '__main__':
    main()
