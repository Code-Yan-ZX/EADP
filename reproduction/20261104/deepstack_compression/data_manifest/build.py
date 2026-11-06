#!/usr/bin/env python3
"""Build image-only DEV/CONFIRM/FINAL manifest; never reads predictions or answers."""
import base64
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version
from scipy.fft import dctn
import scipy

ROOT = Path('/media/disk2/YZX/LMUData')
OUT = Path(__file__).resolve().parent
TASKS = ['DocVQA_VAL', 'TextVQA_VAL', 'AI2D_TEST']
SEED = '20261104'
csv.field_size_limit(100_000_000)


def sha(x):
    return hashlib.sha256(x).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class UF:
    def __init__(self): self.parent = []
    def add(self): self.parent.append(len(self.parent)); return len(self.parent) - 1
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        a, b = self.find(x), self.find(y)
        if a != b: self.parent[max(a, b)] = min(a, b)


def describe(raw):
    with Image.open(io.BytesIO(raw)) as im:
        rgb = ImageOps.exif_transpose(im).convert('RGB')
        wh = rgb.size
        rgbsha = sha(f'{wh[0]}x{wh[1]}|RGB|'.encode() + rgb.tobytes())
        small = np.asarray(rgb.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    freq = dctn(small, type=2, norm='ortho')[:8, :8].ravel()
    med = np.median(freq[1:])
    bits = 0
    for b in (freq > med): bits = (bits << 1) | int(b)
    return wh, rgbsha, bits


def main():
    uf, images, rows = UF(), [], []
    raw_map, pixel_map, semantic_map = {}, {}, {}
    source = {}
    # Each TSV can refer to a previous image by question ID. No question/answer is consumed.
    for task in TASKS:
        path = ROOT / f'{task}.tsv'
        source[task] = {'path': str(path), 'sha256': hash_file(path)}
        refs = {}
        with path.open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                qid, value = row['index'], row['image']
                assert qid not in refs, (task, qid)
                if value.isdecimal():
                    if value not in refs: raise ValueError(f'{task}/{qid}: forward or missing image reference {value}')
                    idx = refs[value]
                else:
                    raw = base64.b64decode(value, validate=True)
                    rawsha = sha(raw)
                    idx = raw_map.get(rawsha)
                    if idx is None:
                        wh, pix, phash = describe(raw)
                        idx = uf.add()
                        images.append({'byte_sha256': rawsha, 'rgb_sha256': pix,
                                       'width': wh[0], 'height': wh[1], 'phash': f'{phash:016x}'})
                        raw_map[rawsha] = idx
                        if pix in pixel_map: uf.union(idx, pixel_map[pix])
                        else: pixel_map[pix] = idx
                refs[qid] = idx
                # Image/page identifier within a dataset, not a shared filename across tasks.
                if task == 'DocVQA_VAL':
                    doc, page = row.get('ucsf_document_id', '').strip(), row.get('ucsf_document_page_no', '').strip()
                    if any(v.lower() in ('', 'nan', 'na', 'none', 'null') for v in (doc, page)):
                        raise ValueError(f'{task}/{qid}: missing document or page identifier')
                    semantic = f'doc_page:{doc}:{page}'
                elif task == 'AI2D_TEST':
                    if not row.get('image_path', '').strip(): raise ValueError(f'{task}/{qid}: missing image_path')
                    semantic = 'ai2d_figure:' + Path(row['image_path']).stem
                else: semantic = ''
                if semantic:
                    if semantic in semantic_map: uf.union(idx, semantic_map[semantic])
                    else: semantic_map[semantic] = idx
                rows.append({'task': task, 'qid': qid, 'image_index': idx, 'source_image_path': row.get('image_path') or None})
                if len(refs) % 100 == 0: print(f'{task}: {len(refs)} questions processed', flush=True)
        print(f'{task}: {len(refs)} questions; total unique byte images {len(images)}', flush=True)

    # Exact candidate enumeration for Hamming <= 4: five disjoint bit chunks,
    # so any pair at distance <= 4 shares at least one chunk (pigeonhole principle).
    # Repeated candidates are removed; perceptual hits are conservatively grouped.
    buckets = defaultdict(list)
    vals = [int(im['phash'], 16) for im in images]
    starts = [0, 13, 26, 39, 52]
    for i, v in enumerate(vals):
        for j, start in enumerate(starts):
            width = 12 if j == 4 else 13
            buckets[(j, (v >> start) & ((1 << width) - 1))].append(i)
    candidates = set()
    for ids in buckets.values():
        for a_pos, a in enumerate(ids):
            for b in ids[a_pos+1:]: candidates.add((a,b))
    near = []
    for a,b in sorted(candidates):
        dist = (vals[a] ^ vals[b]).bit_count()
        if dist <= 4:
            near.append({'a': a, 'b': b, 'distance': dist})
            uf.union(a,b)
    grouped = defaultdict(list)
    for i in range(len(images)): grouped[uf.find(i)].append(i)
    group_id = {}
    for members in grouped.values():
        gid = sha(('group|' + '|'.join(sorted(images[i]['byte_sha256'] for i in members))).encode())
        for i in members: group_id[i] = gid
    for row in rows: row['group_id'] = group_id[row['image_index']]
    available = {r['group_id'] for r in rows}
    split = {}
    panel = []
    missing = {}
    for task in TASKS:
        groups = defaultdict(list)
        for row in rows:
            if row['task'] == task and row['group_id'] in available:
                groups[row['group_id']].append(row)
        ordered = sorted(groups, key=lambda g: (sha(f'{SEED}|{g}'.encode()), g))
        for offset, name, count in [(0, 'DEV', 128), (128, 'CONFIRM', 256)]:
            chosen = ordered[offset:offset+count]
            missing[f'{task}:{name}'] = count-len(chosen)
            for gid in chosen:
                split[gid] = name
                chosen_row = min(groups[gid], key=lambda r: (sha(f"{SEED}|{r['qid']}".encode()), r['qid']))
                panel.append({'task': task, 'qid': chosen_row['qid'], 'group_id': gid, 'split': name})
        available.difference_update(ordered[:384])
    for gid in available: split[gid] = 'FINAL'
    for row in rows: row['split'] = split[row['group_id']]
    assert all(r['split'] == p['split'] for p in panel for r in rows if r['group_id'] == p['group_id'])
    assert len({(r['task'], r['qid']) for r in rows}) == len(rows)
    def write_jsonl(name, items):
        with (OUT/name).open('w', encoding='utf-8') as f:
            for obj in items: f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + '\n')
    write_jsonl('images.jsonl', (dict(index=i, **img, group_id=group_id[i]) for i,img in enumerate(images)))
    write_jsonl('near_pairs.jsonl', near)
    write_jsonl('full_manifest.jsonl', rows)
    write_jsonl('panel_manifest.jsonl', panel)
    counts = defaultdict(int)
    for row in rows: counts[f"{row['task']}:{row['split']}:questions"] += 1
    for p in panel: counts[f"{p['task']}:{p['split']}:panel"] += 1
    meta = {'sources': source, 'builder_sha256': hash_file(Path(__file__).resolve()), 'seed': SEED, 'tasks_order': TASKS, 'counts': dict(sorted(counts.items())),
            'unique_byte_images': len(images), 'groups': len(grouped), 'near_candidate_pairs': len(candidates),
            'near_pairs_hamming_le4': len(near), 'deficits': missing, 'pillow': pillow_version, 'scipy': scipy.__version__,
            'phash_definition': 'EXIF transpose; RGB->L; Lanczos 32x32; orthonormal DCT-II; 8x8; threshold by median of 63 AC coefficients, compare all 64 bits; <=4 union',
            'group_id_definition': 'SHA256 of UTF-8 group| plus sorted byte SHA256 values joined by |',
            'limitations': ['Conservative transitive grouping can join perceptually similar but distinct images; pHash can miss semantically similar pages.', 'AI2D image variants sharing figure basename are grouped; DocVQA original document+page grouped.', 'TRAIN is not included; must audit separately against all evaluation groups before training.', 'Historical exposure remains; FINAL is held out only for this study.', 'Embedded TSV images are fingerprinted; source file pixels and runtime input must be checked before model runs.']}
    meta['outputs_sha256'] = {name:hash_file(OUT/name) for name in ['images.jsonl','near_pairs.jsonl','full_manifest.jsonl','panel_manifest.jsonl']}
    (OUT/'manifest_summary.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'counts': meta['counts'], 'deficits': missing, 'near': len(near)}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
