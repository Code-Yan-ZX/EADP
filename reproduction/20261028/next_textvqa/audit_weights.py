"""Read-only full NeXT checkpoint hashes against archived official HF LFS IDs."""
from pathlib import Path
import datetime
import hashlib
import json
import time

HERE = Path(__file__).resolve().parent
PRIOR = Path('/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/audit_followup_20261008/next_gap_diagnosis_20261008/input_identity/weight_payload_identity.json')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(64 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    source = json.loads(PRIOR.read_text())
    metadata = Path(source['official_metadata_path'])
    assert digest(metadata) == source['official_metadata_sha256']
    rows = []
    for expected in source['files']:
        path = Path(expected['file'])
        before = path.stat()
        started = time.monotonic()
        sha = digest(path)
        after = path.stat()
        stable = (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        row = {'path': str(path), 'bytes': before.st_size, 'sha256': sha, 'official_lfs_sha256': expected['official_lfs_sha256'], 'matches': sha == expected['official_lfs_sha256'] and before.st_size == expected['official_bytes'], 'stat_unchanged': stable, 'mtime_ns': before.st_mtime_ns, 'elapsed_seconds': time.monotonic() - started}
        assert row['matches'] and stable
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'official_model': source['official_repo'], 'official_revision': source['official_revision'], 'official_metadata': {'path': str(metadata), 'sha256': digest(metadata), 'source_url': source['reference_url']}, 'prior_audit': {'path': str(PRIOR), 'sha256': digest(PRIOR)}, 'files': rows, 'all_match': True, 'total_bytes_hashed': sum(r['bytes'] for r in rows), 'GPU_initialized': False, 'source_weights_modified': False}
    (HERE / 'current_weight_identity.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
