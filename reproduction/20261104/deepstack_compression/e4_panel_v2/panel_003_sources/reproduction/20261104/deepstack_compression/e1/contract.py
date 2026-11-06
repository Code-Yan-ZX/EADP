"""Pure-CPU contracts for paired E1 inputs and journaled observations."""
from collections import Counter
from pathlib import Path
import hashlib
import json

ARMS = ('P1-D0', 'P1-D1', 'P3-D0', 'P3-D1')
TASKS = ('DocVQA_VAL', 'TextVQA_VAL', 'AI2D_TEST')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError(f'incomplete JSONL tail: {path}')
    return [json.loads(line) for line in raw.splitlines()]


def validate_panel(panel, expected=128):
    rows = [r for r in panel if r.get('split') == 'DEV']
    counts = Counter(r['task'] for r in rows)
    if set(counts) != set(TASKS) or any(counts[t] != expected for t in TASKS):
        raise ValueError(f'unexpected DEV counts: {dict(counts)}')
    ids = [(r['task'], str(r['qid'])) for r in rows]
    groups = [r['group_id'] for r in rows]
    if len(ids) != len(set(ids)) or len(groups) != len(set(groups)):
        raise ValueError('duplicate task/qid or cross-task image group in DEV')
    if any(not all(r.get(k) for k in ('task', 'qid', 'group_id', 'split')) for r in rows):
        raise ValueError('missing panel field')
    return rows


def validate_manifest(directory):
    d = Path(directory)
    meta = json.loads((d / 'manifest_summary.json').read_text())
    audit = json.loads((d / 'verification.json').read_text())
    if audit.get('status') != 'pass' or audit.get('builder_sha256') != sha(d / 'build.py'):
        raise ValueError('manifest independent verification or builder hash invalid')
    if audit.get('verifier_sha256') != sha(d / 'verify.py'):
        raise ValueError('manifest verifier hash invalid')
    for name, expected in meta['outputs_sha256'].items():
        if sha(d / name) != expected:
            raise ValueError(f'manifest file changed: {name}')
    for task in TASKS:
        source = meta['sources'][task]
        if sha(source['path']) != source['sha256']:
            raise ValueError(f'source changed: {task}')
    if any(meta['deficits'].values()):
        raise ValueError('manifest has insufficient DEV/CONFIRM images')
    return validate_panel(read_jsonl(d / 'panel_manifest.jsonl'))


def validate_anchor_exclusion(directory, old_panel):
    root=Path(directory); d=root/'anchor_exclusion'
    summary=json.loads((d/'summary.json').read_text())
    verification=json.loads((d/'verification.json').read_text())
    if summary.get('status')!='completed' or verification.get('status')!='pass': raise ValueError('anchor exclusion not verified')
    if verification.get('summary_sha256')!=sha(d/'summary.json') or verification.get('verifier_sha256')!=sha(d/'verify.py'):
        raise ValueError('anchor exclusion verification binding mismatch')
    if summary.get('builder_sha256')!=sha(d/'build.py') or summary.get('anchor_plan_sha256')!=sha(old_panel):
        raise ValueError('anchor exclusion source changed')
    if summary.get('panel_manifest_sha256')!=sha(root/'panel_manifest.jsonl') or summary.get('full_manifest_sha256')!=sha(root/'full_manifest.jsonl'):
        raise ValueError('anchor exclusion references a different split')
    for name,digest in summary['outputs_sha256'].items():
        if sha(d/name)!=digest: raise ValueError('anchor exclusion output changed: '+name)
    if summary.get('anchor_distinct_groups')!=16: raise ValueError('expected 16 old anchor groups')
    return {'summary_sha256':sha(d/'summary.json'), 'verification_sha256':sha(d/'verification.json')}


def validate_quartet(records):
    if [r.get('arm') for r in records] != list(ARMS):
        raise ValueError('arm order/inventory mismatch')
    a = records[0]
    common = ('task', 'qid', 'group_id', 'message_sha256', 'processor', 'image_start', 'original_length',
              'ordered_ids', 'spatial_ids', 'retained_ids', 'visual_sha256',
              'pruned_sha256', 'deepstack_full_sha256', 'prefill_embed_sha256',
              'attention_mask_sha256', 'original_position_ids', 'full_rope_delta')
    if any(k not in r or r[k] is None for r in records for k in common):
        raise ValueError('required input evidence missing')
    if any(any(r.get(k) != a.get(k) for k in common) for r in records[1:]):
        raise ValueError('common input or selection differs among arms')
    if any(r.get('generation_scalars') != a.get('generation_scalars') or r.get('forced') != a.get('forced') for r in records[1:]):
        raise ValueError('generation settings differ among arms')
    order, spatial = a['ordered_ids'], a['spatial_ids']
    if len(order) != 256 or len(set(order)) != 256 or any(i < 0 or i >= 1024 for i in order):
        raise ValueError('invalid selected visual IDs')
    if sorted(order) != spatial:
        raise ValueError('spatial IDs do not sort greedy IDs')
    start, original_length = a['image_start'], a['original_length']
    if start < 0 or original_length < start + 1024:
        raise ValueError('invalid original image span')
    retained = list(range(start)) + [start+i for i in spatial] + list(range(start+1024, original_length))
    if a['retained_ids'] != retained:
        raise ValueError('sequence gather does not preserve non-visual positions')
    n = len(retained)
    expected_mask = [[False]*start + [True]*256 + [False]*(n-start-256)]
    full_pos = a['original_position_ids']
    if len(full_pos) != 3 or any(len(axis) != 1 or len(axis[0]) != original_length for axis in full_pos):
        raise ValueError('invalid full-image P3 shape')
    if a['full_rope_delta'] != [[max(max(axis[0]) for axis in full_pos)+1-original_length]]:
        raise ValueError('full mRoPE delta is inconsistent with original positions')
    p1 = [[list(range(n))] for _ in range(3)]
    p3 = [[[axis[0][i] for i in retained]] for axis in full_pos]
    if len(a['deepstack_full_sha256']) != 3:
        raise ValueError('must observe three input DS streams')
    for r in records:
        if r['arm'].endswith('D0'):
            if r['injected_mask'] is not None or r['injected_deepstack'] is not None:
                raise ValueError('D0 must omit DS')
        elif r['injected_mask'] != expected_mask or r['injected_deepstack'] != r['selected_deepstack'] or len(r['selected_deepstack']) != 3:
            raise ValueError('D1 mask or selected DS mismatch')
        if not r.get('output_ids') or not isinstance(r.get('answer'), str):
            raise ValueError('missing output token IDs or answer')
    for x, y in ((records[0], records[1]), (records[2], records[3])):
        if x['prefill_position_ids'] != y['prefill_position_ids'] or x['rope_delta'] != y['rope_delta']:
            raise ValueError('DS changed position or delta')
    if records[0]['prefill_position_ids'] != p1 or records[0]['rope_delta'] != [[0]]:
        raise ValueError('P1 is not compact 1D')
    if records[2]['prefill_position_ids'] != p3:
        raise ValueError('P3 does not gather original positions')
    if records[2]['rope_delta'] != [[a['full_rope_delta'][0][0] + 768]]:
        raise ValueError('P3 decode delta is not full delta plus removed tokens')
    return True


def validate_decode_trace(trace, compact_length, delta, steps=32, layers=36):
    if len(trace) < steps + 1:
        raise ValueError('fewer than 32 decode observations')
    for step, obs in enumerate(trace[1:steps+1], start=1):
        cache_position=compact_length+step-1
        if obs['cache_position'] != [cache_position]:
            raise ValueError('decode cache_position mismatch')
        if obs['position_ids'] != [[[cache_position+delta]]] * 3:
            raise ValueError('decode mRoPE position mismatch')
        if len(obs['layer_cache_lengths']) != layers or any(n != cache_position for n in obs['layer_cache_lengths']):
            raise ValueError('actual per-layer KV cache length mismatch')
    return True


def choose_legacy_dev(records, full, old_anchors):
    old_ids={str(row['index']) for row in old_anchors}
    if len(old_anchors)!=16 or len(old_ids)!=16: raise ValueError('old anchor inventory incomplete')
    old_rows=[r for r in full if r['task']=='DocVQA_VAL' and str(r['qid']) in old_ids]
    if {str(r['qid']) for r in old_rows}!=old_ids: raise ValueError('old anchor group mapping incomplete')
    old_groups={r['group_id'] for r in old_rows}
    if len(old_groups)!=16: raise ValueError('historical anchors do not map to 16 distinct groups')
    candidates=[r for r in records if r['task']=='DocVQA_VAL' and r['group_id'] not in old_groups]
    chosen=sorted(candidates,key=lambda r:(hashlib.sha256(('20261104|'+r['group_id']).encode()).hexdigest(),r['group_id']))[:16]
    if len(chosen)!=16 or len({r['group_id'] for r in chosen})!=16: raise ValueError('insufficient independent DEV groups')
    return chosen, old_groups
