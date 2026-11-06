#!/usr/bin/env python3
"""Audit expected existing panel image paths, without creating/replacing images."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import build

HERE = Path(__file__).resolve().parent
def load(name):
    with (HERE/name).open(encoding='utf8') as f: return [json.loads(x) for x in f]
def sha(data): return hashlib.sha256(data).hexdigest()

def main():
    images, full, panel = map(load,['images.jsonl','full_manifest.jsonl','panel_manifest.jsonl'])
    by_qid = {(r['task'],r['qid']):r for r in full}
    rows, stats = [], Counter()
    for n,p in enumerate(panel):
        r=by_qid[(p['task'],p['qid'])]
        ref=images[r['image_index']]
        # Author's dataset dump_image names explicit image_path or index.jpg.
        # Never use another question's file for a missing index.jpg.
        relative = r['source_image_path'] if r['source_image_path'] else p['qid']+'.jpg'
        path = build.ROOT/'images'/p['task']/relative
        item = {**p,'expected_path':str(path),'tsv_byte_sha256':ref['byte_sha256'],
                'tsv_rgb_sha256':ref['rgb_sha256'],'tsv_phash':ref['phash']}
        key=p['task']+':'+p['split']
        stats[key+':total'] += 1
        try:
            raw=path.read_bytes()
            wh,pix,ph=build.describe(raw)
            distance=(ph^int(ref['phash'],16)).bit_count()
            item.update({'status':'found','cache_byte_sha256':sha(raw),'cache_rgb_sha256':pix,
                         'cache_phash':f'{ph:016x}','cache_width':wh[0],'cache_height':wh[1],
                         'byte_match':sha(raw)==ref['byte_sha256'], 'rgb_match':pix==ref['rgb_sha256'],
                         'phash_hamming_to_tsv':distance})
            stats[key+':found'] += 1
            stats[key+':byte_equal'] += int(item['byte_match'])
            stats[key+':rgb_equal'] += int(item['rgb_match'])
            stats[key+':phash_distance_gt4'] += int(distance>4)
        except FileNotFoundError:
            item['status']='missing_expected_path'; stats[key+':missing'] += 1
        except (OSError,ValueError) as ex:
            item.update({'status':'unreadable','error':str(ex)}); stats[key+':unreadable'] += 1
        rows.append(item)
        if (n+1)%100==0: print(f'audited {n+1}/{len(panel)} panel images',flush=True)
    out=HERE/'panel_cache_audit.jsonl'
    with out.open('w',encoding='utf8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
    summary={'script_sha256':sha(Path(__file__).read_bytes()),
             'builder_sha256':sha((HERE/'build.py').read_bytes()),
             'panel_sha256':sha((HERE/'panel_manifest.jsonl').read_bytes()),
             'audit_sha256':sha(out.read_bytes()),'counts':dict(sorted(stats.items())),
             'path_rule':'dataset image_path if present, otherwise question index.jpg, per fixed author image_base.py dump_image; report missing, never substitute',
             'limitations':['Only existing expected files were read; none were generated or substituted.',
                            'Expected path must be verified against the eventual runner at execution time.',
                            'pHash distance to the embedded image is a diagnostic, not proof of pixel equality.']}
    (HERE/'panel_cache_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary['counts'],ensure_ascii=False),flush=True)

if __name__=='__main__': main()
