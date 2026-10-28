"""CPU-only independent post-run verification; reads all generation inputs.

Run only after state.json reports complete. The only write is a new independent
verification artifact beside this file; the worker and old results are untouched.
"""
import ast
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import zipfile
from zoneinfo import ZoneInfo

import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
PERCEPTION = ['OCR', 'artwork', 'celebrity', 'color', 'count', 'existence',
              'landmark', 'position', 'posters', 'scene']
COGNITION = ['code_reasoning', 'commonsense_reasoning',
             'numerical_calculation', 'text_translation']
ARCHIVE_SHA = 'b8125e2a7c3418e5761c12b3cfe4f1624b3c53ba44009e75b7d3f797d3d8acee'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(left, right):
    assert abs(float(left) - float(right)) < 1e-8, (left, right)


def extraction_function():
    """Load only the actual pure parser functions, avoiding model imports."""
    env = {}
    for rel, name in [
        ('Qwen_vl/VLMEvalKit/vlmeval/smp/misc.py', 'process_punctuation'),
        ('Qwen_vl/VLMEvalKit/vlmeval/dataset/utils/yorn.py', 'YOrN_Extraction'),
    ]:
        path = ROOT / rel
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env['YOrN_Extraction']


def aggregate(gt, hits):
    categories = {}
    for category, frame in gt.groupby('category'):
        pairs = defaultdict(list)
        for i, row in frame.iterrows():
            pairs[row['image_path']].append(hits[i])
        assert all(len(pair) == 2 for pair in pairs.values())
        correct = sum(hits[i] for i in frame.index)
        both = sum(all(pair) for pair in pairs.values())
        categories[category] = dict(n=len(frame), n_images=len(pairs),
            correct=correct, both_correct=both,
            score=100 * (correct / len(frame) + both / len(pairs)))
    p = sum(categories[c]['score'] for c in PERCEPTION)
    c = sum(categories[c]['score'] for c in COGNITION)
    return dict(perception=p, reasoning=c, total=p + c), categories


def main():
    state = read(OUT / 'state.json')
    assert state['status'] == 'complete' and state['completed'] == 2374
    assert state['exit_code'] == 0
    protocol = read(OUT / 'protocol.json')
    for source, expected in protocol['source_sha256'].items():
        assert digest(source) == expected, source
    loaded = read(OUT / 'loaded_runtime.json')
    assert loaded['model_name'] == 'Qwen3-VL-8B-Instruct-1024'
    assert loaded['dtype'] == 'torch.bfloat16' and loaded['attn'] == 'sdpa'
    assert loaded['cfg'] == dict(kind='uniform', lam=0.25, scope='main', scorer='rtg', K=128)
    assert loaded['legacy_flags'] == dict(deepstack=False, pos='1d')
    assert loaded['generate_kwargs']['max_new_tokens'] == 2048
    old_path = Path(protocol['old_shard'])
    base = OUT / 'run01/legacy_full/acc/L_R_MAIN0125'
    old, new = read(old_path), read(base / 'MME.json')
    expected_ids = {str(i) for i in range(2374)}
    assert set(old['records']) == set(new['records']) == expected_ids
    assert new['meta']['K'] == old['meta']['K'] == 128
    assert new['meta']['legacy'] == old['meta']['legacy'] == dict(deepstack=False, pos='1d')
    assert new['meta']['max_new_tokens'] == old['meta']['max_new_tokens'] == 2048
    assert new['meta']['pipeline'] == old['meta']['pipeline'] == 'official_legacy'
    bank_path = Path(protocol['bank_reuse'])
    assert digest(bank_path) == old['meta']['bank_sha256'] == new['meta']['bank_sha256']
    assert (OUT / 'run01/full/bank_full_rtg_MME.json.gz').resolve() == bank_path.resolve()
    with gzip.open(bank_path, 'rt') as stream:
        bank = json.load(stream)
    assert set(bank['samples']) == expected_ids and bank['meta']['K'] == 128
    for i in range(2374):
        record = bank['samples'][str(i)]
        assert record['idx'] == i and record['n_vis'] == 1024
        assert len(record['keep']) == len(set(record['keep'])) == 128
        assert record['keep'] == sorted(record['keep'])
        assert all(0 <= value < 1024 for value in record['keep'])

    gt_path = Path('/media/disk2/YZX/LMUData/MME.tsv')
    gt = pd.read_csv(gt_path, sep='\t', keep_default_na=False,
                     usecols=['index', 'category', 'image_path', 'question', 'answer'])
    pred = pd.read_csv(base / 'MME_pred.tsv', sep='\t', keep_default_na=False)
    aux = pd.read_excel(base / 'MME_pred_auxmatch.xlsx', keep_default_na=False)
    assert len(gt) == len(pred) == len(aux) == 2374
    assert gt['index'].tolist() == pred['index'].tolist() == aux['index'].tolist() == list(range(2374))
    for col in gt.columns:
        assert gt[col].tolist() == pred[col].tolist(), col
    for col in pred.columns:
        assert pred[col].tolist() == aux[col].tolist(), col

    archive = ROOT / 'LLaVA/playground/data/eval/MME/eval_tool.zip'
    assert digest(archive) == ARCHIVE_SHA
    templates = []
    with zipfile.ZipFile(archive) as stream:
        for folder in ['Your_Results', 'LaVIN']:
            mapping = {}
            for category in PERCEPTION + COGNITION:
                for line in stream.read(f'eval_tool/{folder}/{category}.txt').decode().splitlines():
                    image, question, answer, *_ = line.split('\t')
                    key = category, Path(image).stem, ' '.join(question.split())
                    assert key not in mapping
                    mapping[key] = answer.lower().strip()
            templates.append(mapping)
    assert templates[0] == templates[1] and len(templates[0]) == 2374
    seen = set()
    for _, row in gt.iterrows():
        key = row['category'], Path(row['image_path']).stem, ' '.join(row['question'].split())
        assert templates[0][key] == row['answer'].lower().strip()
        assert key not in seen
        seen.add(key)
    assert seen == set(templates[0])

    extract = extraction_function()
    old_hits, new_hits, text_changes, extracted_changes = [], [], [], []
    for i, row in gt.iterrows():
        a, b = old['records'][str(i)], new['records'][str(i)]
        assert isinstance(b['prediction'], str) and b['prediction'].strip()
        assert b['n_vis_kept'] == 128
        assert pred.iloc[i]['prediction'] == b['prediction']
        assert bool(pred.iloc[i]['truncated']) == bool(b['truncated'])
        x, y = extract(a['prediction']), extract(b['prediction'])
        assert y == aux.iloc[i]['extracted']
        old_hits.append(x == row['answer'])
        new_hits.append(y == row['answer'])
        assert new_hits[-1] == bool(aux.iloc[i]['score'])
        if a['prediction'] != b['prediction']:
            text_changes.append(i)
        if x != y:
            extracted_changes.append(i)
    summary_old, categories_old = aggregate(gt, old_hits)
    summary_new, categories_new = aggregate(gt, new_hits)
    for path, hits, summary in [(old_path.with_name('MME_score.json'), old_hits, summary_old),
                                (base / 'MME_score.json', new_hits, summary_new)]:
        saved = read(path)
        assert set(saved['per_question']) == expected_ids
        assert all(float(saved['per_question'][str(i)]) == float(hits[i]) for i in range(2374))
        for name in ['perception', 'reasoning']:
            value = saved['official'][name]
            if isinstance(value, str):
                value = ast.literal_eval(value)
            if isinstance(value, dict):
                assert len(value) == 1
                value = next(iter(value.values()))
            close(value, summary[name])
    csv_score = pd.read_csv(base / 'MME_pred_score.csv').iloc[0]
    for name in ['perception', 'reasoning']:
        close(csv_score[name], summary_new[name])
    for category in PERCEPTION + COGNITION:
        close(csv_score[category], categories_new[category]['score'])
    result = read(OUT / 'result.json')
    assert result['complete'] and result['n'] == 2374
    assert result['fresh_shard_sha256'] == digest(base / 'MME.json')
    assert result['runtime_sha256'] == digest(OUT / 'runtime.jsonl')
    for label, summary in [('old', summary_old), ('new', summary_new)]:
        for name in ['perception', 'reasoning', 'total']:
            close(result['summary'][label][name], summary[name])
    for category in PERCEPTION + COGNITION:
        close(result['categories'][category]['old'], categories_old[category]['score'])
        close(result['categories'][category]['new'], categories_new[category]['score'])
    traces = [json.loads(line) for line in (OUT / 'runtime.jsonl').read_text().splitlines()]
    assert len(traces) == 2374 and [r['index'] for r in traces] == list(range(2374))
    images, false_layer_ids = {}, []
    for r in traces:
        i = r['index']
        row, prediction = gt.iloc[i], new['records'][str(i)]
        assert r['cfg'] == dict(kind='uniform', lam=0.25, scope='main', scorer='rtg', K=128)
        assert r['legacy_flags'] == dict(deepstack=False, pos='1d')
        assert r['n_vis_kept'] == 128
        assert r['gen_tokens'] == len(r['gen_ids']) <= 2048
        assert bool(r['truncated']) == bool(prediction['truncated']) == (r['gen_tokens'] >= 2048)
        assert r['prediction_sha256'] == hashlib.sha256(prediction['prediction'].encode()).hexdigest()
        assert r['message_sha256'] == hashlib.sha256(json.dumps(r['message'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        texts = [m['value'] for m in r['message'] if m['type'] == 'text']
        assert texts == [row['question'] + ' Please answer yes or no.']
        assert len(r['input_images']) == 1
        image = r['input_images'][0]
        assert Path(image['path']).as_posix().endswith('/MME/' + row['image_path'])
        assert [m['value'] for m in r['message'] if m['type'] == 'image'] == [image['path']]
        images[image['path']] = image['sha256']
        invariants = r['invariants']
        assert all(invariants[k] for k in ['no_dup', 'in_range', 'ascending'])
        assert invariants['n_vis_full'] == 1024 and invariants['deepstack'] is False
        assert invariants['pos_mode'] == '1d'
        # decode returns the final capped token without another text forward.
        # The frozen diagnostic counters instead assume every returned token
        # has been forwarded, so both flags are expected false at the cap.
        skipped_forward = int(r['truncated'])
        assert invariants['cache_len'] == invariants['lk'] + r['gen_tokens'] - skipped_forward
        n_layers = invariants['layer_calls_expected'] // (1 + r['gen_tokens'])
        assert n_layers == 36
        assert invariants['layer_calls'] == n_layers * (1 + r['gen_tokens'] - skipped_forward)
        assert bool(invariants['cache_ok']) == (not r['truncated'])
        assert bool(invariants['layer_calls_ok']) == (not r['truncated'])
        if not invariants['layer_calls_ok']:
            false_layer_ids.append(i)
            assert r['truncated'], 'non-cap layer call mismatch'
    for path, expected in images.items():
        assert digest(path) == expected, path
    for source, expected in protocol['source_sha256'].items():
        assert digest(source) == expected, source
    rescues = sum(not a and b for a, b in zip(old_hits, new_hits))
    breaks = sum(a and not b for a, b in zip(old_hits, new_hits))
    assert result['text_changes'] == len(text_changes)
    assert result['extracted_changes'] == len(extracted_changes)
    assert result['rescues'] == rescues and result['breaks'] == breaks
    verification = dict(checks='all_pass', created_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        scope='Independent CPU verification of complete same-bank Qwen K128 MME repeat',
        n=2374, image_pairs=len(images), canonical_gt_archive_sha256=ARCHIVE_SHA,
        old_summary=summary_old, new_summary=summary_new,
        categories_old=categories_old, categories_new=categories_new,
        text_change_ids=text_changes, extracted_change_ids=extracted_changes,
        rescues=rescues, breaks=breaks, truncated_ids=[i for i, r in enumerate(traces) if r['truncated']],
        layer_call_mismatch_ids=false_layer_ids,
        original_sources_preserved=True, bank_rebuilt=False,
        code_failure_reproduced_ids=[i for i in [796, 800, 810, 812, 818] if not new_hits[i]],
        new_shard_sha256=digest(base / 'MME.json'), runtime_sha256=digest(OUT / 'runtime.jsonl'),
        limitations=['Same-bank repeat does not test independent bank construction.',
                     'Truncation is recorded; the frozen scoring protocol does not force capped answers to be wrong.',
                     'At the cap the decoder skips the final unnecessary forward; frozen cache/layer diagnostic flags expect it. Actual cache/call counts were checked against this exact cap behavior.'])
    target = OUT / 'independent_verification.json'
    assert not target.exists(), 'Independent verification output already exists'
    target.write_text(json.dumps(verification, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: verification[k] for k in ['checks', 'n', 'old_summary', 'new_summary', 'rescues', 'breaks', 'text_change_ids']}))


if __name__ == '__main__':
    main()
