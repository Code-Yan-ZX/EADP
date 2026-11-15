"""CPU-only exact rendering audit, without changing manifests or images."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import data_build as db


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default=str(db.OUT/'combined_manifest.json'))
    parser.add_argument('--out',default=str(db.OUT/'combined_pixel_audit.json'))
    args=parser.parse_args();manifest=json.loads(Path(args.manifest).read_text());rows=[]
    for record in manifest['records']:
        source=np.asarray(Image.open(record['source_image']).convert('RGB'))
        mask=np.zeros(source.shape[:2],dtype=bool)
        for patch in record['rendering']['patches']:
            x0,y0,x1,y1=patch['region'];mask[y0:y1,x0:x1]=True
        members={}
        for name in ['original_image','swapped_image']:
            image=np.asarray(Image.open(record[name]).convert('RGB'))
            assert image.shape==source.shape
            changed=np.any(image!=source,axis=-1)
            members[name]=dict(dimensions_identical=True,outside_patch_changed_pixels=int(changed[~mask].sum()),
                               inside_patch_changed_pixels=int(changed[mask].sum()))
            assert members[name]['outside_patch_changed_pixels']==0
        fields=record['fields']
        assert record['original_numeric_multiset']==record['swapped_numeric_multiset']
        assert db.key(fields[0]['swapped_answer'])==db.key(fields[1]['original_answer'])
        assert db.key(fields[1]['swapped_answer'])==db.key(fields[0]['original_answer'])
        assert all(db.key(f['original_answer'])==db.key(f['ocr_text']) for f in fields)
        sha=hashlib.sha256(Path(record['source_image']).read_bytes()).hexdigest()
        assert sha==record['source_sha256']
        quality=record['render_quality']
        assert quality['digit_count_ratio']<=2
        assert quality['minimum_rendered_height_retention']>=.6
        assert not any(f['value_bbox'][3]>f['value_bbox'][2] and sum(c.isdigit() for c in f['ocr_text'])<=2 for f in fields)
        rows.append(dict(pair_id=record['pair_id'],dataset=record['dataset'],source_sha256_matches=True,
                         numeric_multiset_equal=True,answers_swapped_exactly=True,render_quality_pass=True,**members))
    docs=[r['source_document_id'] for r in manifest['records']]
    shas=[r['source_sha256'] for r in manifest['records']]
    assert len(set(docs))==len(docs)
    assert len(set(shas))==len(shas)
    summary=dict(all_pass=True,pairs=len(rows),datasets=dict(Counter(r['dataset'] for r in rows)),
                 unique_documents=len(set(docs)),unique_source_bytes=len(set(shas)),exact_outside_patch_invariance=True,
                 note='Decoded RGB untouched source compared with both rerendered PNG members. All 2x documented patch regions are allowed to change; all other pixels must be exact. Geometric masks are not native preprocessing validation.',records=rows)
    db.atomic_json(args.out,summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False))

if __name__=='__main__':
    main()
