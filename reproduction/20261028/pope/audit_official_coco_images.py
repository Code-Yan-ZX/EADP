#!/usr/bin/env python3
"""Check all500POPEimages against source-linked COCO ZIP, without a6GB download."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
import zlib

ROOT = Path(__file__).resolve().parent
URL = 'http://images.cocodataset.org/zips/val2014.zip'


def main():
    helper = Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/audit_official_images.py')
    spec = importlib.util.spec_from_file_location('range_reader_only', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    remote = module.RangeReader(URL)
    local = Path('/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval/pope/val2014')
    questions = [json.loads(line) for line in (ROOT / 'inputs/llava_pope_test.jsonl').read_text().splitlines()]
    names = sorted({q['image'] for q in questions})
    assert len(names) == 500
    rows, sampled = [], []
    panel_names = [names[int((i + .5) * len(names) / 16)] for i in range(16)]
    fresh = ROOT / 'fresh_coco_panel_images'
    fresh.mkdir(exist_ok=True)
    with zipfile.ZipFile(remote) as archive:
        by_name = {Path(entry.filename).name: entry for entry in archive.infolist() if not entry.is_dir()}
        for name in names:
            raw = (local / name).read_bytes()
            entry = by_name[name]
            crc = zlib.crc32(raw) & 0xffffffff
            rows.append({'image': name, 'zip_member': entry.filename, 'local_bytes': len(raw), 'official_bytes': entry.file_size,
                         'local_crc32': crc, 'official_crc32': entry.CRC, 'matches': len(raw) == entry.file_size and crc == entry.CRC})
        print(json.dumps({'phase': 'all500_crc_size_checked', 'mismatches': sum(not r['matches'] for r in rows)}), flush=True)
        for name in panel_names:
            raw = archive.read(by_name[name])
            upstream_sha = hashlib.sha256(raw).hexdigest()
            local_sha = hashlib.sha256((local / name).read_bytes()).hexdigest()
            (fresh / name).write_bytes(raw)
            sampled.append({'image': name, 'fresh_sha256': upstream_sha, 'local_sha256': local_sha, 'matches': upstream_sha == local_sha})
    report = {'source_link': 'author LLaVA/EVAL.md POPE val2014 link', 'url': URL, 'archive_size': remote.size,
        'archive_headers': remote.headers, 'used_images': len(names), 'crc_size_mismatches': [r for r in rows if not r['matches']],
        'all_used_image_crc_size_checks': rows, 'fresh_image_sha_panel_selection': '16 evenly spaced sorted-used-image filenames; fixed independently of predictions/labels',
        'fresh_image_sha_panel': sampled, 'range_requests': remote.requests, 'downloaded_bytes': sum(r['bytes'] for r in remote.requests),
        'helper_path': str(helper), 'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
        'scope': 'All500usedimagepayloads match fresh officialZIPcentral CRC32+size. Cryptographic payload identity independently checked only for16actuallydownloadedJPEGmembers. Existing localimageSHAlist covers500; no GPU or dataset mutation.'}
    (ROOT / 'official_coco_image_identity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'complete': True, 'used_images': len(names), 'crc_size_mismatches': len(report['crc_size_mismatches']),
                      'fresh_sha_panel': len(sampled), 'sha_mismatches': sum(not r['matches'] for r in sampled),
                      'downloaded_bytes': report['downloaded_bytes'], 'whole_archive_bytes': remote.size}), flush=True)


if __name__ == '__main__':
    main()
