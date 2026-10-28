"""Independently join exact prompts, actual image inputs, and stored answers.

The content-compatible projection is a diagnostic, not a full old-version run.
The strict projection requires identical native FP16 image input and image size.
Neither mapping uses ground truth to select a candidate or remaps answer letters.
"""
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/media/disk2/YZX/research/EADP_amp')
OUT = Path(__file__).resolve().parent
EVAL = ROOT / 'LLaVA/playground/data/eval'
OLD = EVAL / 'mmbench/mmbench_dev_20230712.tsv'
NEW = EVAL / 'mmbench/MMBench_DEV_EN_V11.tsv'
SOURCES = {
    'EADP': OUT.parent / 'mmb_eadp_immediate_20261010/v15_mmben_E128_streamwait.jsonl',
    'AnchorZip': EVAL / 'anchorzip_p3/mmben/LRMAIN025.jsonl',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prompt(row):
    hint = row.get('hint', '')
    text = (hint + '\n' if hint and hint.lower() not in {'nan', 'none'} else '') + row['question']
    for letter in 'ABCD':
        option = row.get(letter, '')
        if not option or option.lower() in {'nan', 'none'}:
            break
        text += '\n' + letter + '. ' + option
    return text


def load_tsv(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    by_id = {int(row['index']): row for row in rows}
    assert len(by_id) == len(rows)
    return by_id


def select(pairs, index, rule):
    candidates = sorted({p['v11_index'] for p in pairs if p[rule] and p['image_sizes_exact']})
    if index in candidates:
        return index
    return candidates[0] if len(candidates) == 1 else None


def main():
    csv.field_size_limit(1024 ** 3)
    old, new = load_tsv(OLD), load_tsv(NEW)
    assert len(old) == 4377 and len(new) == 4876
    pair_file = OUT / 'image_mapping/image_pairs.json'
    pairs = json.loads(pair_file.read_text())
    by_old = collections.defaultdict(list)
    for pair in pairs:
        ix, iy = pair['old_index'], pair['v11_index']
        assert prompt(old[ix]) == prompt(new[iy])
        by_old[ix].append(pair)

    verifier = OUT.parent / 'mmb_diagnosis_20261010/verify_v15_mmb_run.py'
    spec = importlib.util.spec_from_file_location('independent_mmb_verifier', verifier)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    infer = module.load_toolkit_extractor()
    predictions = {}
    for method, path in SOURCES.items():
        records = [json.loads(line) for line in path.open()]
        predictions[method] = {int(record['question_id']): record for record in records}
        assert len(records) == len(predictions[method]) == len(new)
        assert set(predictions[method]) == set(new)
        for index, record in predictions[method].items():
            assert record['model_id'] == 'llava-v1.5-7b'
            assert record['round_id'] == 0 and record['prompt'] == prompt(new[index])

    groups = collections.defaultdict(list)
    for index in old:
        groups[index % 1000000].append(index)
    rules = {
        'strict_model_input': 'strict_model_image_input_reusable',
        'same_content_diagnostic': 'near_perceptual_diagnostic_only',
    }
    maps = {name: {index: select(by_old[index], index, rule) for index in old}
            for name, rule in rules.items()}
    maps['same_ID_prompt_diagnostic'] = {
        index: index if index in new and prompt(row) == prompt(new[index]) else None
        for index, row in old.items()
    }
    ledger = []
    group_ledger = []
    extraction_disagreements = []
    for index, row in old.items():
        entry = {'old_index': index, 'old_base_id': index % 1000000, 'old_GT': row['answer']}
        for name, selected in maps.items():
            source = selected[index]
            entry[name + '_source_index'] = source
            entry[name + '_V11_GT'] = None if source is None else new[source]['answer']
            for method, prediction in predictions.items():
                value = None if source is None else prediction[source]['text'].strip()
                hit = None
                if source is not None:
                    options = {letter: row[letter] for letter in 'ABCD'
                               if row.get(letter) and row[letter].lower() not in {'nan', 'none'}}
                    extracted = infer(value, options)
                    regex = module.extract_regex(value)
                    if extracted != regex:
                        extraction_disagreements.append([index, name, method, value, extracted, regex])
                    hit = extracted == row['answer']
                entry[name + '_' + method + '_prediction'] = value
                entry[name + '_' + method + '_old_GT_correct'] = hit
        ledger.append(entry)
    assert not extraction_disagreements
    ledger_by_id = {entry['old_index']: entry for entry in ledger}

    summary = {'scope': 'LLaVA-1.5-7B, EN legacy Dev 20230712, K128; stored V11 answers projected only',
               'old_rows': len(old), 'old_groups': len(groups), 'CPU_only': True,
               'GPU_runs_started': 0, 'TeX_modified': False, 'projections': {},
               'extractor_disagreements': extraction_disagreements}
    for name, selected in maps.items():
        complete = [base for base, indices in groups.items() if all(selected[i] is not None for i in indices)]
        unchanged_gt = [base for base in complete if all(old[i]['answer'] == new[selected[i]]['answer']
                                                        for i in groups[base])]
        method_scores = {}
        for method in predictions:
            def score(bases, ground_truth):
                correct = 0
                for base in bases:
                    hits = []
                    for index in groups[base]:
                        source = selected[index]
                        predicted = predictions[method][source]['text'].strip()
                        answer = old[index]['answer'] if ground_truth == 'old' else new[source]['answer']
                        hits.append(predicted == answer)
                    correct += all(hits)
                return {'correct_groups': correct, 'n_groups': len(bases),
                        'circular_accuracy_percent': 100 * correct / len(bases) if bases else None}
            method_scores[method] = {
                'old_GT': score(complete, 'old'),
                'V11_GT_on_same_rows': score(complete, 'V11'),
                'unchanged_GT_subset': score(unchanged_gt, 'old'),
            }
        missing = [base for base in groups if base not in set(complete)]
        summary['projections'][name] = {
            'can_claim_full_old_version_run': False,
            'n_matched_rows': sum(source is not None for source in selected.values()),
            'n_complete_groups': len(complete), 'n_incomplete_groups': len(missing),
            'group_coverage_percent': 100 * len(complete) / len(groups),
            'n_complete_groups_with_GT_changes': len(complete) - len(unchanged_gt),
            'complete_group_ids': complete, 'incomplete_group_ids': missing,
            'methods': method_scores,
        }
        for base, indices in groups.items():
            sources = [selected[index] for index in indices]
            full = all(source is not None for source in sources)
            record = {'projection': name, 'old_base_id': base, 'expected_rotations': len(indices),
                      'matched_rotations': sum(source is not None for source in sources),
                      'complete': full, 'old_indices': indices, 'source_V11_indices': sources}
            for method in predictions:
                record[method + '_circular_old_GT_correct'] = (
                    all(ledger_by_id[index][name + '_' + method + '_old_GT_correct'] for index in indices)
                    if full else None)
            group_ledger.append(record)
    summary['rules'] = {
        'strict_model_input': 'Exact native prompt, equal FP16 CLIP336 padded tensor bytes and original image_sizes; no GT in matching.',
        'same_content_diagnostic': 'Exact native prompt, same image size, processed RGB8 RMSE<=3 and dHash distance<=2; recompressed pixels can differ. Diagnostic only.',
        'same_ID_prompt_diagnostic': 'Same row ID and exact native prompt only; image differences are not filtered. Diagnostic only.',
        'circular': 'Score an old group only when every old rotation has a selected answer; every rotation must match old GT.',
        'letters': 'Use original stored answer letters without permutation/remapping.',
    }
    summary['limits'] = [
        'Content-compatible images do not guarantee equal model predictions on the original encoded old images.',
        'Missing groups remain unknown; covered-subset percentages are not full1176-group scores.',
        'Old GT alignment does not establish that older labels are factually more correct.',
        'The stored EADP and historical AnchorZip executions differ in stream-wait repair.',
    ]
    summary['source_sha256'] = {str(path): sha(path) for path in [OLD, NEW, pair_file, verifier, Path(__file__), *SOURCES.values()]}
    (OUT / 'reuse_scores.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    comparison = []
    for name, result in summary['projections'].items():
        for method, scores in result['methods'].items():
            comparison.append({'projection': name, 'method': method,
                               'old_GT_correct_groups': scores['old_GT']['correct_groups'],
                               'n_complete_old_groups': result['n_complete_groups'],
                               'n_total_old_groups': len(groups),
                               'old_GT_circular_accuracy_percent': scores['old_GT']['circular_accuracy_percent'],
                               'full_old_version_score': False})
    with (OUT / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    for filename, records in [('old_rows_rescored.csv', ledger), ('old_groups_rescored.csv', group_ledger)]:
        with (OUT / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps({name: {key: value for key, value in result.items()
                            if not key.endswith('_ids')} for name, result in summary['projections'].items()},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
