"""Read-only TSV and cached-image audit against fresh author decode functions."""
import ast
import base64
import csv
import hashlib
import io
import json
import os
import os.path as osp
from pathlib import Path
import sys
import tempfile

from PIL import Image
import PIL

OUT = Path(__file__).resolve().parent
AUTHOR = Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp')
DATA = Path('/media/disk2/YZX/LMUData')
EXPECTED = {'DocVQA_VAL': ('d5ee77e1926ff10690d469c56b73eabf', 5349),
            'AI2D_TEST': ('0f593e0d1c7df9a3d69bf1f947e71975', 3088)}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    csv.field_size_limit(sys.maxsize)
    source = AUTHOR / 'Qwen_vl/VLMEvalKit/vlmeval/smp/vlm.py'
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
             node.name in ('decode_base64_to_image', 'decode_base64_to_image_file')]
    assert len(nodes) == 2
    namespace = dict(base64=base64, io=io, Image=Image, osp=osp, os=os)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    decode = namespace['decode_base64_to_image_file']
    result = {'author_decode_source': str(source), 'author_decode_sha256': file_hash(source),
              'Pillow': PIL.__version__, 'datasets': {},
              'scope': 'Regenerate each unique referenced image with unmodified author decode functions in temporary output, then compare complete cached file bytes. Original TSV/cache read-only.'}
    for dataset, (expected_md5, expected_n) in EXPECTED.items():
        path = DATA / (dataset + '.tsv')
        before = path.stat()
        md5 = file_hash(path, 'md5')
        assert md5 == expected_md5
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        assert len(rows) == expected_n and len({r['index'] for r in rows}) == expected_n
        image_map = {r['index']: r['image'] for r in rows}
        checked, mismatches = {}, []
        with tempfile.TemporaryDirectory(prefix='eadp-author-image-decode-') as temp:
            for i, row in enumerate(rows):
                payload = row['image']
                if len(payload) <= 64:
                    payload = image_map[payload]
                    assert len(payload) > 64
                if payload.startswith('['):
                    values = ast.literal_eval(payload)
                    assert len(values) == 1
                    payload = values[0]
                name = row['image_path']
                if name.startswith('['):
                    values = ast.literal_eval(name)
                    assert len(values) == 1
                    name = values[0]
                payload_sha = sha_bytes(payload.encode())
                if name in checked:
                    assert checked[name]['encoded_payload_sha256'] == payload_sha
                    checked[name]['question_count'] += 1
                    continue
                original = DATA / 'images' / dataset / name
                assert original.is_file(), str(original)
                expected = Path(temp) / name
                cache_stat = original.stat()
                decode(payload, str(expected))
                original_sha, regenerated_sha = file_hash(original), file_hash(expected)
                match = original_sha == regenerated_sha
                record = {'image_path': str(original), 'encoded_payload_sha256': payload_sha,
                          'cached_sha256': original_sha, 'author_regenerated_sha256': regenerated_sha,
                          'bytes': cache_stat.st_size, 'byte_exact': match, 'question_count': 1}
                assert original.stat().st_mtime_ns == cache_stat.st_mtime_ns
                if not match:
                    with Image.open(original) as a, Image.open(expected) as b:
                        record['cached_size'], record['expected_size'] = a.size, b.size
                        record['RGB_pixel_exact'] = a.size == b.size and a.convert('RGB').tobytes() == b.convert('RGB').tobytes()
                    mismatches.append(record)
                checked[name] = record
                expected.unlink()
                if len(checked) % 200 == 0:
                    print(dataset, 'unique checked', len(checked), 'row', i + 1, 'mismatches', len(mismatches), flush=True)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        result['datasets'][dataset] = {'tsv': str(path), 'tsv_bytes': before.st_size,
            'tsv_md5': md5, 'author_expected_md5': expected_md5,
            'rows': len(rows), 'unique_images': len(checked), 'mismatches': mismatches,
            'all_cached_images_match_author_regeneration': not mismatches, 'images': list(checked.values())}
        (OUT / 'dataset_cache_audit.json').write_text(json.dumps(result, indent=2) + '\n')
        print(dataset, 'complete', len(rows), 'questions', len(checked), 'unique images', len(mismatches), 'mismatches', flush=True)
    assert all(r['all_cached_images_match_author_regeneration'] for r in result['datasets'].values())


if __name__ == '__main__':
    main()
