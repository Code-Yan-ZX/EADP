"""Independent CPU re-read of frozen student panel source rows and images."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/student_distill/data'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(4*1024*1024): h.update(b)
    return h.hexdigest()

def run():
    roles={'dev':1,'fit':120,'val':16,'test':32}
    data={role:json.loads((OUT/f'{role}_manifest.json').read_text()) for role in roles}
    rows=[]
    for role,n in roles.items():
        d=data[role]
        assert d['role']==role and d['complete'] and d['records']==d['items'] and len(d['items'])==n
        assert d['provenance']['selection_uses_model_outputs'] is False
        assert all(r['role']==role for r in d['items'])
        rows.extend(d['items'])
    assert len({r['item_id'] for r in rows})==len(rows)
    assert len({r['parent_id'] for r in rows})==len(rows)
    assert len({r['image_sha256'] for r in rows})==len(rows)
    assert not ({r['parent_id'] for r in rows} & set(data['fit']['provenance']['excluded_parent_documents']))
    assert not ({r['image_sha256'] for r in rows} & set(data['fit']['provenance']['prior_panel_source_image_hashes']))
    by_shard=defaultdict(list)
    for r in rows: by_shard[r['source_parquet']].append(r)
    fields=['question_id','question','answers','other_metadata','data_split','image']
    for path, group in by_shard.items():
        assert sha(path)==group[0]['source_parquet_sha256']
        table=pq.read_table(path,columns=fields)
        for r in group:
            orig=table.slice(r['source_parquet_row'],1).to_pylist()[0]
            md=orig['other_metadata']; content=orig['image']['bytes']
            assert orig['data_split']=='train' and int(orig['question_id'])==r['question_id']
            assert orig['question']==r['question'] and list(orig['answers'])==r['answers']==r['answers_original']
            assert str(md['ucsf_document_id'])==r['parent_id'] and md['image']==r['image_id']
            assert str(md['ucsf_document_page_no'])==r['source_page_number']
            assert hashlib.sha256(content).hexdigest()==r['image_sha256']==r['source_sha256']==sha(r['image_path'])
        del table
    out={'complete':True,'source_rows_verified':len(rows),'distinct_shards_verified':len(by_shard),
         'all_original_question_alias_parent_page_and_image_bytes_verified':True,
         'unique_item_parent_source_image_and_historical_exclusions_verified':True,
         'manifest_sha256':{role:sha(OUT/f'{role}_manifest.json') for role in roles},
         'audit_script_sha256':sha(__file__)}
    path=OUT/'independent_audit.json'
    if path.exists():
        if json.loads(path.read_text())!=out: raise ValueError('Conflicting immutable prior audit')
    else:
        path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': run()
