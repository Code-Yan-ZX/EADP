"""Single-image native input/selection contracts; NumPy only inside functions."""
import hashlib
import json
from pathlib import Path

ARMS = ('source','beta2')
POSITIONS = list(range(16))
SUFFIX = '\nAnswer the question using a single word or phrase.'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):value.update(block)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')


def array_sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def equal_bits(a,b):
    import numpy as np
    return a.dtype==b.dtype and a.shape==b.shape and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()


def guidance(prompt,arm):
    assert arm in ARMS and prompt.endswith(SUFFIX)
    return prompt.replace(SUFFIX,'')


def selection(mask,ids,quotas):
    import numpy as np
    assert mask.dtype==np.bool_ and mask.shape==(1,576)
    assert ids.dtype==quotas.dtype==np.int64 and ids.shape==(1,32) and quotas.tolist()==[32]
    active=np.flatnonzero(mask[0]).tolist();ordered=ids[0].tolist()
    assert len(active)==len(set(ordered))==32 and sorted(ordered)==active
    return dict(greedy_quotas=[32],effective_mask_indices=[active],greedy_effective_prefixes=[ordered],returned_retained_tokens=32)


def embeddings(embedded,projected,mask,input_ids):
    import numpy as np
    assert input_ids.ndim==2 and input_ids.shape[0]==1 and input_ids.dtype==np.int64
    marker=np.flatnonzero(input_ids[0]==-200);assert len(marker)==1;marker=int(marker[0])
    assert projected.dtype==embedded.dtype==np.float16 and projected.shape==(1,576,4096)
    assert mask.dtype==np.bool_ and mask.shape==(1,576) and mask.sum()==32
    assert embedded.shape==(1,input_ids.shape[1]-1+32,4096)
    visual=projected[0,np.flatnonzero(mask[0])]
    assert equal_bits(embedded[0,marker:marker+32],visual)
    nonvisual=np.concatenate([embedded[0,:marker],embedded[0,marker+32:]],axis=0)
    assert nonvisual.shape==(input_ids.shape[1]-1,4096)
    return nonvisual,dict(marker_index=marker,n_visual=32,visual_fp16_sha256=array_sha(visual),
        nonvisual_fp16_sha256=array_sha(nonvisual),prepared_embeddings_shape=list(embedded.shape))


def require_all_sources(records):
    assert len(records)==16 and [r['position'] for r in records]==POSITIONS
    assert all(r['arm']=='source' and r['all_source_anchors_passed'] is True for r in records)


def stat(path):
    p=Path(path);s=p.stat()
    return dict(resolved=str(p.resolve()),bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
