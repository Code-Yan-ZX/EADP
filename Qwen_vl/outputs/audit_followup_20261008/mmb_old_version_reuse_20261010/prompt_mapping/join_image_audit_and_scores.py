"""Join CPU native-prompt/image identity audits and project unchanged answers.

The strict bucket requires exact FP16 model image inputs and image_sizes.
Perceptual buckets are sensitivity diagnostics, not exact legacy reproduction.
All rotations of each legacy group must be covered. GT never gates matching.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path('/media/disk2/YZX/research/EADP_amp')
AUDIT = ROOT / 'Qwen_vl/outputs/audit_followup_20261008'
DEST = AUDIT / 'mmb_old_version_reuse_20261010/prompt_mapping'
IMAGE = DEST.parent / 'image_mapping'
EVAL = ROOT / 'LLaVA/playground/data/eval'
OLD = EVAL / 'mmbench/mmbench_dev_20230712.tsv'
V11 = EVAL / 'mmbench/MMBench_DEV_EN_V11.tsv'
PRED = {
    'EADP': AUDIT / 'mmb_eadp_immediate_20261010/v15_mmben_E128_streamwait.jsonl',
    'AnchorZip': EVAL / 'anchorzip_p3/mmben/LRMAIN025.jsonl',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def present(x):
    return x is not None and str(x).lower() not in ('', 'nan', 'none')


def prompt(row):
    text = (row['hint'] + '\n' if present(row.get('hint')) else '') + row['question']
    for option in 'ABCD':
        if not present(row.get(option)):
            break
        text += '\n' + option + '. ' + row[option]
    return text


def table(path):
    csv.field_size_limit(10000000)
    return {int(row['index']): row for row in csv.DictReader(path.open(), delimiter='\t')}


def select(old_index, candidates):
    if old_index in candidates:
        return old_index
    if len(candidates) == 1:
        return candidates[0]
    return None


def main():
    old, new = table(OLD), table(V11)
    image_path = IMAGE / 'old_rows_image_mapping.json'
    prompt_path = DEST / 'old_rows_mapping.json'
    image_rows = {r['old_index']: r for r in json.loads(image_path.read_text())}
    prompt_rows = {r['old_index']: r for r in json.loads(prompt_path.read_text())}
    assert set(old) == set(image_rows) == set(prompt_rows)
    for index in old:
        assert image_rows[index]['candidate_v11_indices'] == prompt_rows[index]['candidate_v11_indices']
    predictions = {}
    for method, path in PRED.items():
        rows = [json.loads(line) for line in path.open()]
        predictions[method] = {int(row['question_id']): row for row in rows}
        assert len(rows) == len(predictions[method]) == len(new)
        assert set(predictions[method]) == set(new)
        for index, row in predictions[method].items():
            assert row['prompt'] == prompt(new[index])
            assert row['text'].strip() in ('A', 'B', 'C', 'D')
    old_groups = collections.defaultdict(list)
    for index in old:
        old_groups[index % 1000000].append(index)
    policies = {
        'exact_model_inputs': 'Native prompt exact + FP16 pad/CLIP336 image tensor exact + image_sizes exact; all old rotations covered.',
        'near_images_same_ID_diagnostic': 'Same row ID and native prompt exact; final RGB8 RMSE<=3 and dHash<=2. Includes exact bucket; approximate diagnostic only.',
        'near_images_any_ID_diagnostic': 'Native prompt exact; final RGB8 RMSE<=3 and dHash<=2; prefer same ID otherwise one unique image-near candidate. Includes exact bucket; approximate diagnostic only.',
        'same_ID_prompt_only_diagnostic': 'Same row ID and native prompt exact, images not gated. Approximate projection only.',
    }
    mappings = {name: {} for name in policies}
    for index in old:
        ir, pr = image_rows[index], prompt_rows[index]
        mappings['exact_model_inputs'][index] = select(index, ir['strict_image_eligible_v11_indices'])
        near = ir['near_perceptual_candidate_v11_indices']
        mappings['near_images_same_ID_diagnostic'][index] = index if index in near else None
        mappings['near_images_any_ID_diagnostic'][index] = select(index, near)
        mappings['same_ID_prompt_only_diagnostic'][index] = index if index in pr['candidate_v11_indices'] else None
    result = {'schema': 'CPU_native_input_and_legacy_GT_projection_v1', 'old_rows': len(old), 'old_groups': len(old_groups),
              'GT_policy': 'Evaluate unchanged predicted letters against old GT and V11 GT separately. Old GT defines the old evaluation convention and is not asserted to be more accurate.',
              'can_claim_full_old_version_score': False,
              'source_sha256': {str(p): sha(p) for p in [OLD, V11, *PRED.values(), image_path, prompt_path, IMAGE / 'summary.json', Path(__file__)]},
              'buckets': {}}
    group_details = []
    for name, mapping in mappings.items():
        complete = []
        for group, indices in sorted(old_groups.items()):
            if any(mapping[index] is None for index in indices):
                continue
            selected = [mapping[index] for index in sorted(indices)]
            source_groups = sorted({index % 1000000 for index in selected})
            assert len(source_groups) == 1
            for old_i, new_i in zip(sorted(indices), selected):
                assert prompt(old[old_i]) == prompt(new[new_i])
            entry = {'bucket': name, 'old_base_id': group, 'old_indices': sorted(indices),
                     'selected_v11_indices': selected, 'source_v11_base_ids': source_groups,
                     'uses_cross_ID_mapping': any(o != n for o, n in zip(sorted(indices), selected)),
                     'GT_changed': any(old[o]['answer'] != new[n]['answer'] for o, n in zip(sorted(indices), selected)),
                     'methods': {}}
            for method, pred in predictions.items():
                entry['methods'][method] = {
                    'old_GT_circular_correct': all(pred[n]['text'].strip() == old[o]['answer'] for o, n in zip(sorted(indices), selected)),
                    'V11_GT_circular_correct': all(pred[n]['text'].strip() == new[n]['answer'] for n in selected),
                    'unchanged_predicted_letters': [pred[n]['text'].strip() for n in selected],
                    'old_GT_letters': [old[o]['answer'] for o in sorted(indices)],
                    'V11_GT_letters': [new[n]['answer'] for n in selected],
                }
            complete.append(entry)
            group_details.append(entry)
        bucket = {'policy': policies[name], 'exact_reuse': name == 'exact_model_inputs',
                  'n_complete_old_groups': len(complete), 'n_complete_old_rows': sum(len(x['old_indices']) for x in complete),
                  'group_coverage_percent_of_1176': 100 * len(complete) / len(old_groups),
                  'n_missing_old_groups': len(old_groups) - len(complete),
                  'n_GT_changed_groups': sum(x['GT_changed'] for x in complete),
                  'n_cross_ID_groups': sum(x['uses_cross_ID_mapping'] for x in complete),
                  'old_group_ids': [x['old_base_id'] for x in complete], 'methods': {}}
        for method in PRED:
            co = sum(x['methods'][method]['old_GT_circular_correct'] for x in complete)
            cn = sum(x['methods'][method]['V11_GT_circular_correct'] for x in complete)
            rescue = sum(x['methods'][method]['old_GT_circular_correct'] and not x['methods'][method]['V11_GT_circular_correct'] for x in complete)
            broken = sum(not x['methods'][method]['old_GT_circular_correct'] and x['methods'][method]['V11_GT_circular_correct'] for x in complete)
            bucket['methods'][method] = {'old_GT_correct': co, 'old_GT_accuracy_percent': 100 * co / len(complete) if complete else None,
                                         'V11_GT_correct_same_mapped_rows': cn, 'V11_GT_accuracy_percent_same_mapped_rows': 100 * cn / len(complete) if complete else None,
                                         'switch_to_old_GT_rescued_groups': rescue, 'switch_to_old_GT_broken_groups': broken,
                                         'old_GT_minus_V11_GT_pp': 100 * (co - cn) / len(complete) if complete else None}
        result['buckets'][name] = bucket
    result['overlap_note'] = 'The exact bucket is contained in both near buckets and in the same-ID prompt-only bucket. Bucket denominators are separately filtered; their counts must not be summed.'
    (DEST / 'combined_image_prompt_scores.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (DEST / 'combined_image_prompt_group_details.json').write_text(json.dumps(group_details, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: {x: y for x, y in v.items() if x != 'old_group_ids'} for k, v in result['buckets'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
