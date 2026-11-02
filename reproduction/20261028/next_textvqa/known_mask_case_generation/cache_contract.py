"""Lossless CPU contract for the already identified image11 mask case."""
import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
NEXT = OUT.parent
PANEL = NEXT / 'gpu_formula_facility_panel'
CACHE = PANEL / 'execution/image_11.npz'
ARMS = ['source', 'eq6_quantile_membership_only']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
array_sha = lambda x: hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def visual_order(mask):
    ids = np.arange(2880).reshape(5,576)
    # Unchanged native spatial branch: base first, local2x2 crops interleaved.
    local = ids[1:].reshape(2,2,24,24).transpose(0,2,1,3).reshape(-1)
    local_mask = mask[1:].reshape(2,2,24,24).transpose(0,2,1,3).reshape(-1)
    return np.concatenate([ids[0][mask[0]],local[local_mask]])


def load():
    registry = [json.loads(line) for line in (PANEL/'execution/per_image.jsonl').read_text().splitlines()][11]
    assert sha(CACHE) == registry['cache_sha256'] == '7c3a29c72f15c0c7e2052e09be948a696245d1e03d0710cb8af3aface44a0ce0'
    assert registry['panel']['full_question_position'] == 3632
    with np.load(CACHE,allow_pickle=False) as z:
        arrays = {name:z[name] for name in ['projected_fp16','actual_input_fp16']}
        for arm in ARMS:
            for field in ['final_mask','quotas','internal_ids']:
                arrays[arm+'_'+field] = z['K128_alpha05_beta2__'+arm+'__'+field]
    f = arrays['projected_fp16']
    assert f.shape == (5,576,4096) and f.dtype == np.float16 and np.isfinite(f).all()
    assert arrays['actual_input_fp16'].shape == (1,5,3,336,336) and arrays['actual_input_fp16'].dtype == np.float16
    descriptions = {}
    for arm in ARMS:
        mask,quota,ids = [arrays[arm+'_'+name] for name in ['final_mask','quotas','internal_ids']]
        assert mask.shape == (5,576) and mask.dtype == np.bool_ and quota.tolist() == [146,162,106,130,93]
        assert np.array_equal(mask.sum(1),quota)
        for crop in range(5):
            prefix=ids[crop,:quota[crop]]
            assert len(set(prefix.tolist())) == quota[crop]
            assert np.array_equal(np.flatnonzero(mask[crop]),np.sort(prefix))
        order=visual_order(mask);assert len(order)==637
        selected=f.reshape(2880,4096)[order]
        descriptions[arm]=dict(mask_sha256=array_sha(mask),quotas=quota.tolist(),visual_order_sha256=array_sha(order),
            selected_visual_fp16_sha256=array_sha(selected),visual_count=637)
    assert np.argwhere(arrays['source_final_mask'] != arrays[ARMS[1]+'_final_mask']).tolist() == [[3,115],[3,330]]
    a,b = [visual_order(arrays[arm+'_final_mask']) for arm in ARMS]
    assert np.flatnonzero(a!=b).tolist() == list(range(465,548))
    description=dict(cache_path=str(CACHE),cache_sha256=sha(CACHE),projected_fp16_sha256=array_sha(f),
        actual_input_fp16_sha256=array_sha(arrays['actual_input_fp16']),arms=descriptions,
        shifted_visual_sequence_positions=list(range(465,548)),panel=registry['panel'])
    return arrays,description


def verify_embeddings(embeddings, inputs_n, marker, arrays, contract):
    for arm in ARMS:
        x=embeddings[arm]
        assert x.shape == (1,inputs_n-1+637,4096) and x.dtype==np.float16
        expected=arrays['projected_fp16'].reshape(2880,4096)[visual_order(arrays[arm+'_final_mask'])]
        assert np.array_equal(x[0,marker:marker+637].view(np.uint16),expected.view(np.uint16))
        assert array_sha(x[0,marker:marker+637])==contract['arms'][arm]['selected_visual_fp16_sha256']
    a,b=[embeddings[arm] for arm in ARMS]
    assert np.array_equal(a[:,:marker].view(np.uint16),b[:,:marker].view(np.uint16))
    assert np.array_equal(a[:,marker+637:].view(np.uint16),b[:,marker+637:].view(np.uint16))
