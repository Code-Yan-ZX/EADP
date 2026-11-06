"""Bind the reviewed no-GPU preflight; this is NOT a GPU admission gate."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
P = BASE / 'e6/pixel_wavefront_model_v2'
R = BASE / 'e1_independent_review'
REVIEW = R / 'pixel_wavefront_model_v2_preflight_result_review.json'
REVIEWER = R / 'review_pixel_wavefront_model_v2_preflight_result.py'
PROVENANCE = R / 'pixel_wavefront_model_v2_preflight_reviewer_provenance.md'
STARTED = R / 'monitor_pixel_wavefront_model_v2_preflight_result/started.json'
TERMINAL = R / 'monitor_pixel_wavefront_model_v2_preflight_result/terminal.json'
EXPECTED = {
    REVIEW: '8447413963e3a00985bc4da59784f047ff193e873dcca282abdc8ccc7dc1ea52',
    REVIEWER: '4d743dda5532beaa7479f6a9e47bd61d5258f2f832766d940413df868e464f9b',
    PROVENANCE: '91d2fb9f5b7726cab6e5e59289662f1d30261c52d2c7476586fae3f4703ce899',
    STARTED: 'c00ef79f509e8b8ee9f3eb923e1deef2fed4a8d88b625c649840cf8a0299fcdf',
    TERMINAL: '26b547f3f5de839a9b29b09fa04bd62995a546c664e15abab80571149da47fc5',
}
PROOF_FIELDS = {
    'plan_sha256': P / 'preflight_plan001.json',
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
        or proof.get('scope') != 'three-anchor real no-GPU preflight result only'
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
    if not isinstance(sources, dict) or len(sources) != 177:
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
