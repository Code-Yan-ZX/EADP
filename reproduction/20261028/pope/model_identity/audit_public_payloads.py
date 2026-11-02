#!/usr/bin/env python3
"""CPU-only streaming identity check against pinned public Hugging Face files."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import urllib.request

OUT = Path(__file__).resolve().parent
V15_LOCAL = Path('/media/disk2/YZX/doct/FastV/llava-v1.5-7b')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    started = time.time()
    report = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'models': {},
              'policy': 'Every existing binary payload is streamed once for SHA256; no weight tensor is deserialized, moved, changed, or downloaded. Public metadata and small configuration/tokenizer files only are downloaded. No GPU API is called.'}
    for label, repo, local in [('llava_v15', 'liuhaotian/llava-v1.5-7b', V15_LOCAL),
                               ('clip', 'openai/clip-vit-large-patch14-336', None)]:
        metadata_path = OUT / f'{label}_hf_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        revision = metadata['sha']
        if local is None:
            local = Path('/home/dell/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots') / revision
        rows = []
        for sibling in metadata['siblings']:
            name = sibling['rfilename']
            if name in {'.gitattributes', 'README.md', 'tf_model.h5'}:
                continue
            path = local / name
            row = {'filename': name, 'local_path': str(path), 'resolved_local_path': str(path.resolve()),
                   'official_size': sibling.get('size'), 'exists': path.is_file()}
            if not path.is_file():
                rows.append(row)
                continue
            row.update(bytes=path.stat().st_size, local_sha256=digest(path))
            if 'lfs' in sibling:
                row['official_lfs_sha256'] = sibling['lfs']['sha256']
                row['byte_identity_match'] = row['local_sha256'] == row['official_lfs_sha256']
                print(json.dumps({'model': label, 'file': name, 'bytes': row['bytes'], 'LFS_match': row['byte_identity_match']}), flush=True)
            else:
                url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
                data = urllib.request.urlopen(url, timeout=45).read()
                reference = OUT / ('upstream_' + label) / name
                reference.parent.mkdir(parents=True, exist_ok=True)
                reference.write_bytes(data)
                row.update(official_url=url, official_reference_path=str(reference),
                           official_sha256=hashlib.sha256(data).hexdigest(), byte_identity_match=path.read_bytes() == data)
                if name.endswith('.json'):
                    a, b = json.loads(path.read_text()), json.loads(data)
                    row['json_semantically_identical'] = a == b
                    if a != b and isinstance(a, dict) and isinstance(b, dict):
                        row['json_differences'] = {key: {'local': a.get(key), 'official': b.get(key)}
                                                   for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
            rows.append(row)
        report['models'][label] = {'repository': repo, 'public_revision': revision, 'local_root': str(local),
                                  'metadata_source_url': f'https://huggingface.co/api/models/{repo}?blobs=true',
                                  'metadata_sha256': digest(metadata_path), 'files': rows,
                                  'all_expected_runtime_files_present': all(x['exists'] for x in rows),
                                  'all_present_file_bytes_match': all(x.get('byte_identity_match', False) for x in rows if x['exists'])}
    report['elapsed_seconds'] = time.time() - started
    report['all_runtime_file_bytes_match'] = all(x['all_expected_runtime_files_present'] and x['all_present_file_bytes_match'] for x in report['models'].values())
    report['ignored_local_extra_files'] = [{'path': str(V15_LOCAL / 'configuration.json'),
        'reason': 'ModelScope pipeline metadata; not requested by transformers AutoConfig/AutoTokenizer or full pretrained state loading.'}]
    (OUT / 'model_identity_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'complete': True, 'all_runtime_file_bytes_match': report['all_runtime_file_bytes_match'],
                      'elapsed_seconds': report['elapsed_seconds'],
                      'by_model': {k: {'revision': x['public_revision'], 'all_present': x['all_expected_runtime_files_present'],
                                       'all_match': x['all_present_file_bytes_match'], 'n_files': len(x['files'])}
                                   for k, x in report['models'].items()}}), flush=True)


if __name__ == '__main__':
    main()
