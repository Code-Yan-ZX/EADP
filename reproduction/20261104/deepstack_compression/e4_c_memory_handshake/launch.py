"""Reviewed, bounded, strictly synthetic Docker handshake entry."""
import argparse
import json
import os
import secrets
import traceback
from pathlib import Path
import gate
import harness


def save_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())


def launch(run_id):
    if run_id != '001': raise ValueError('only frozen rehearsal 001')
    approval = gate.admit()  # absolutely before directory creation / Docker
    d = gate.HERE/'run_001'
    d.mkdir(exist_ok=False)
    nonce = secrets.token_hex(16)
    name, create, worker, supervisor, monitor = harness.commands(d, nonce)
    save_new(d/'launch.json', {'scope':'synthetic no-GPU handshake only',
             'gpu_authorized':False, 'model_loaded':False, 'GT_opened':False,
             'nonce':nonce, 'container_name':name, 'docker_create':create,
             'fake_sampler_worker':worker, 'sampler_supervisor':supervisor,
             'sampler_monitor':str(monitor), 'approval':approval})
    result = {'status':'failed', 'scope':'synthetic no-GPU handshake only',
              'gpu_authorized':False, 'failure':None}
    try:
        result['lifecycle'] = harness.execute(d, nonce)
        if result['lifecycle'].get('status') != 'passed':
            raise ValueError('synthetic handshake failed')
        if gate.sources() != approval['source_sha256']:
            raise ValueError('rehearsal source changed during execution')
        if gate.runtime.source_hashes() != approval['runtime']['source_sha256']:
            raise ValueError('runtime source changed during execution')
        result['status'] = 'passed'
    except BaseException as exc:
        result['failure'] = {'type':type(exc).__name__, 'message':str(exc),
                             'traceback':traceback.format_exc()}
    finally:
        save_new(d/'exit.json', result)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-id', required=True); a=p.parse_args()
    if launch(a.run_id)['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__': main()
