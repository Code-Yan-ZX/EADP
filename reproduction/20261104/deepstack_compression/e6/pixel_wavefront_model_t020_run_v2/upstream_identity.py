"""Bind the reviewed no-GPU preflight; this is NOT a GPU admission gate."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
P = BASE / 'e6/pixel_wavefront_model_t020'
R = BASE / 'e1_independent_review'
REVIEW = R / 'pixel_wavefront_model_t020_preflight_result_review.json'
REVIEWER = R / 'review_pixel_wavefront_model_t020_preflight_result.py'
PROVENANCE = BASE / 'e6/pixel_wavefront_model_t020_execution/PREFLIGHT_REVIEWER_PROVENANCE.md'
STARTED = R / 'monitor_pixel_wavefront_model_t020_preflight_result/started.json'
TERMINAL = R / 'monitor_pixel_wavefront_model_t020_preflight_result/terminal.json'
EXPECTED = {
    REVIEW: 'fd6ca3c674dbb6ec92f3f143cc60618a54ef88ee1218b952084ff8d2d6baed57',
    REVIEWER: 'eb389ad47918ab78aa58c8bb621ca12a273242e9f876dba890fbf6f6fe62242d',
    PROVENANCE: 'dddbfa14fa80376ff34016c14b65b7637fa937d4f65066b2ae8305107df5cd3e',
    STARTED: 'd76394ce25f3923e6d03acb5f5c7be8e8712bc6277bd220a8fcde9439c4b0742',
    TERMINAL: '4269546d6851b7c78721968210230d37871713dbbf6c27fc02a4f91c69f47080',
}
PROOF_FIELDS = {
    'plan_sha256': P / 'preflight_plan002.json',
    'worker_preflight_sha256': P / 'run_preflight001/worker_preflight.json',
    'launch_sha256': P / 'run_preflight001/launch.json',
    'exit_sha256': P / 'run_preflight001/exit.json',
    'started_sha256': P / 'monitor_preflight001/started.json',
    'terminal_sha256': P / 'monitor_preflight001/terminal.json',
    'reviewer_sha256': REVIEWER,
}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def validate_bundle(expected, proof_fields, review, started, terminal, reviewer, out, *, rehash_sources=True):
    # Injected paths make rejection behavior testable without an existing proof.
    for path, value in expected.items():
        if digest(path) != value:
            raise ValueError('frozen reviewed preflight bundle changed: ' + str(path))
    proof, start, end = read(review), read(started), read(terminal)
    if (proof.get('status') != 'passed' or proof.get('independent_review') is not True
        or proof.get('scope') != 'three-anchor t020 real no-GPU preflight result only'
        or proof.get('gpu_authorized') is not False or proof.get('GT_opened') is not False
        or proof.get('errors') != [] or proof.get('images') != 3
        or proof.get('weights_verified') != 4 or proof.get('metadata_aliases_verified') != 7):
        raise ValueError('review proof scope or counts')
    cmd = start.get('command', [])
    if (len(cmd) != 4 or (Path(start['cwd']) / cmd[1]).resolve() != Path(reviewer).resolve()
        or cmd[2] != '--out' or (Path(start['cwd']) / cmd[3]).resolve() != Path(out).resolve()
        or start.get('worker_pid') != start.get('process_group')
        or end.get('worker_pid') != start.get('worker_pid') or end.get('reason') != 'completed'
        or end.get('worker_returncode') != 0 or end.get('remaining_process_group') != []):
        raise ValueError('review supervision failed')
    for key, path in proof_fields.items():
        if proof.get(key) != digest(path):
            raise ValueError('preflight proof file changed: ' + key)
    sources = proof.get('source_sha256')
    if not isinstance(sources, dict) or len(sources) != 226:
        raise ValueError('preflight source inventory')
    if rehash_sources:
        for path, expected_sha in sources.items():
            if digest(path) != expected_sha:
                raise ValueError('preflight source changed: ' + path)
    return {'status': 'passed', 'no_gpu_preflight_only': True,
            'candidate_implementation_independently_reviewed': False,
            'source_count': len(sources), 'review_sha256': digest(review),
            'reviewer_sha256': digest(reviewer)}

def verify():
    return validate_bundle(EXPECTED, PROOF_FIELDS, REVIEW, STARTED, TERMINAL, REVIEWER, REVIEW)

if __name__ == '__main__':
    raise SystemExit('No executable GPU or model entry is provided by this module.')
