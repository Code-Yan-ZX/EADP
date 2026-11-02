"""CPU input/selection contracts. No model imports or execution."""
import hashlib
import json
from pathlib import Path

ARMS = ('source', 'question_only')
SUFFIX = '\nAnswer the question using a single word or phrase.'
SEPARATOR = '\nReference OCR token'
POSITIONS = [195,507,820,1132,1445,1757,2070,2382,2695,3007,3320,3632,3945,4257,4570,4882]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def array_sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def equal_bits(a,b):
    import numpy as np
    return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(
        np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8))


def guidance(prompt,arm):
    assert arm in ARMS and prompt.count(SEPARATOR)==1 and prompt.endswith(SUFFIX)
    return (prompt if arm=='source' else prompt.split(SEPARATOR)[0]).replace(SUFFIX,'')


def spatial_order(mask):
    import numpy as np
    assert mask.shape==(5,576) and mask.dtype==np.bool_
    selected=[t for t in range(576) if mask[0,t]]
    for r in range(48):
        for c in range(48):
            crop=1+2*(r//24)+c//24;token=24*(r%24)+c%24
            if mask[crop,token]:selected.append(crop*576+token)
    return np.asarray(selected,dtype=np.int64)


def selection(mask,ids,quotas):
    import numpy as np
    assert mask.dtype==np.bool_ and mask.shape==(5,576)
    assert ids.dtype==quotas.dtype==np.int64 and quotas.shape==(5,)
    assert ids.shape==(5,int(quotas.max())) and np.all((quotas>=1)&(quotas<=576))
    assert np.array_equal(mask.sum(axis=1),quotas)
    prefixes=[];indices=[]
    for crop in range(5):
        prefix=ids[crop,:int(quotas[crop])].tolist();active=np.flatnonzero(mask[crop]).tolist()
        assert len(set(prefix))==len(prefix) and sorted(prefix)==active
        prefixes.append(prefix);indices.append(active)
    return dict(greedy_quotas=quotas.tolist(),effective_mask_indices=indices,
                greedy_effective_prefixes=prefixes,returned_retained_tokens=int(mask.sum()))


def embeddings(embedded,projected,mask,input_ids):
    import numpy as np
    assert input_ids.ndim==2 and input_ids.shape[0]==1 and input_ids.dtype==np.int64
    markers=np.flatnonzero(input_ids[0]==-200);assert len(markers)==1
    marker=int(markers[0]);count=int(mask.sum());dimension=projected.shape[-1]
    assert projected.dtype==embedded.dtype==np.float16 and projected.shape[:2]==(5,576)
    assert embedded.shape==(1,input_ids.shape[1]-1+count,dimension)
    visual=projected.reshape(2880,dimension)[spatial_order(mask)]
    assert equal_bits(embedded[0,marker:marker+count],visual)
    nonvisual=np.concatenate([embedded[0,:marker],embedded[0,marker+count:]],axis=0)
    assert nonvisual.shape==(input_ids.shape[1]-1,dimension)
    return nonvisual,dict(marker_index=marker,n_visual=count,visual_fp16_sha256=array_sha(visual),
        nonvisual_fp16_sha256=array_sha(nonvisual),prepared_embeddings_shape=list(embedded.shape))


def require_all_sources(records):
    assert len(records)==16 and [r['position'] for r in records]==POSITIONS
    assert all(r['arm']=='source' and r['all_source_anchors_passed'] is True for r in records)


def stat(path):
    p=Path(path);s=p.stat()
    return dict(resolved=str(p.resolve()),bytes=s.st_size,device=s.st_dev,inode=s.st_ino,
                mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
