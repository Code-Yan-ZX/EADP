"""CPU-only projection of complete V11 predictions onto identical old prompts.

The native guidance prompt is the exact hint/question/ordered A-D text. GT is
excluded from the match key, because GT is not a model input. Predicted letters
are never rotated or remapped. Images are verified by a separate audit: this
script does not claim that textual matching proves complete input identity.
"""
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path('/media/disk2/YZX/research/EADP_amp')
AUDIT = ROOT / 'Qwen_vl/outputs/audit_followup_20261008'
EVAL = ROOT / 'LLaVA/playground/data/eval'
OLD = EVAL / 'mmbench/mmbench_dev_20230712.tsv'
NEW = EVAL / 'mmbench/MMBench_DEV_EN_V11.tsv'
DEST = AUDIT / 'mmb_old_version_reuse_20261010/prompt_mapping'
PRED = {
    'EADP': AUDIT / 'mmb_eadp_immediate_20261010/v15_mmben_E128_streamwait.jsonl',
    'AnchorZip': EVAL / 'anchorzip_p3/mmben/LRMAIN025.jsonl',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def present(value):
    return value is not None and str(value).lower() not in ('', 'nan', 'none')


def prompt(row):
    result = (row['hint'] + '\n' if present(row.get('hint')) else '') + row['question']
    for option in ('A', 'B', 'C', 'D'):
        # Native get_options stops at the first absent option.
        if not present(row.get(option)):
            break
        result += '\n' + option + '. ' + row[option]
    return result


def load(path):
    csv.field_size_limit(10000000)
    result = {int(row['index']): row for row in csv.DictReader(path.open(), delimiter='\t')}
    assert len(result) == sum(1 for _ in csv.DictReader(path.open(), delimiter='\t'))
    return result


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    old, new = load(OLD), load(NEW)
    old_groups, new_groups = collections.defaultdict(list), collections.defaultdict(list)
    lookup = collections.defaultdict(list)
    for index, row in old.items():
        old_groups[index % 1000000].append(index)
    for index, row in new.items():
        new_groups[index % 1000000].append(index)
        lookup[prompt(row)].append(index)
    predictions = {}
    for method, path in PRED.items():
        rows = [json.loads(line) for line in path.open()]
        predictions[method] = {int(row['question_id']): row for row in rows}
        assert len(rows) == len(predictions[method]) == len(new) == 4876
        assert set(predictions[method]) == set(new)
        for index, row in predictions[method].items():
            assert row['prompt'] == prompt(new[index])
            assert re.fullmatch(r'[A-D]', row['text'].strip())

    mapping, gt_only_rows = [], []
    counts = collections.Counter()
    for index, row in old.items():
        text = prompt(row)
        candidates = sorted(lookup.get(text, []))
        if index in candidates:
            selected, status = index, 'same_ID_exact_prompt'
        elif len(candidates) == 1:
            selected, status = candidates[0], 'cross_ID_unique_exact_prompt'
        elif candidates:
            selected, status = None, 'ambiguous_exact_prompt_candidates'
        else:
            selected, status = None, 'no_exact_prompt_match'
        counts[status] += 1
        entry = {
            'old_index': index,
            'old_base_id': index % 1000000,
            'old_GT': row['answer'],
            'native_prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'candidate_v11_indices': candidates,
            'same_ID_candidate_matches': index in candidates,
            'selected_v11_index_or_null': selected,
            'selected_v11_base_id_or_null': None if selected is None else selected % 1000000,
            'match_status': status,
            'image_identity_verified_by_this_script': False,
            'methods': {},
        }
        if selected is not None:
            entry['selected_V11_GT'] = new[selected]['answer']
            entry['GT_changed'] = row['answer'] != new[selected]['answer']
            for method, pred in predictions.items():
                answer = pred[selected]['text'].strip()
                entry['methods'][method] = {
                    'unchanged_predicted_letter': answer,
                    'correct_against_old_GT': answer == row['answer'],
                    'correct_against_V11_GT': answer == new[selected]['answer'],
                }
            if index == selected and entry['GT_changed']:
                gt_only_rows.append(entry)
        mapping.append(entry)
    by_old = {row['old_index']: row for row in mapping}
    group_report = []
    complete, same_ID_complete, cross_ID_complete = [], [], []
    old_only = set(old_groups) - set(new_groups)
    old_only_recovered = []
    for group, indices in old_groups.items():
        entries = [by_old[index] for index in sorted(indices)]
        all_mapped = all(entry['selected_v11_index_or_null'] is not None for entry in entries)
        source_groups = sorted({entry['selected_v11_base_id_or_null'] for entry in entries
                                if entry['selected_v11_base_id_or_null'] is not None})
        report = {
            'old_base_id': group,
            'old_rotation_indices': sorted(indices),
            'n_expected_rotations': len(indices),
            'n_mapped_rotations': sum(entry['selected_v11_index_or_null'] is not None for entry in entries),
            'complete_prompt_mapping': all_mapped,
            'originally_old_only_ID': group in old_only,
            'source_v11_base_ids': source_groups,
            'uses_cross_ID_mapping': any(entry['match_status'] == 'cross_ID_unique_exact_prompt' for entry in entries),
            'has_GT_changes': any(entry.get('GT_changed', False) for entry in entries),
            'rows': entries,
            'scores': {},
        }
        if all_mapped:
            assert len(source_groups) == 1, 'Mixed source-group mappings require explicit review'
            complete.append(group)
            if report['uses_cross_ID_mapping']:
                cross_ID_complete.append(group)
            else:
                same_ID_complete.append(group)
            if group in old_only:
                old_only_recovered.append(group)
            for method in PRED:
                report['scores'][method] = {
                    'old_GT_circular_correct': all(row['methods'][method]['correct_against_old_GT'] for row in entries),
                    'V11_GT_same_projected_rows_circular_correct': all(row['methods'][method]['correct_against_V11_GT'] for row in entries),
                }
        group_report.append(report)
    group_by_id = {row['old_base_id']: row for row in group_report}

    def subset_score(group_ids, method):
        if not group_ids:
            return {'n_groups': 0, 'old_GT_accuracy_percent': None}
        correct_old = sum(group_by_id[group]['scores'][method]['old_GT_circular_correct'] for group in group_ids)
        correct_new = sum(group_by_id[group]['scores'][method]['V11_GT_same_projected_rows_circular_correct'] for group in group_ids)
        return {
            'n_groups': len(group_ids),
            'old_GT_correct': correct_old,
            'old_GT_accuracy_percent': 100 * correct_old / len(group_ids),
            'V11_GT_correct_on_identical_projected_rows': correct_new,
            'V11_GT_accuracy_percent_on_identical_projected_rows': 100 * correct_new / len(group_ids),
            'old_GT_minus_V11_GT_pp': 100 * (correct_old - correct_new) / len(group_ids),
        }

    gt_affected_complete = [group for group in complete if group_by_id[group]['has_GT_changes']]
    score_summary = {}
    for method in PRED:
        row_changes = {
            'n_same_ID_exact_prompt_GT_changed_rows': len(gt_only_rows),
            'old_GT_correct_rows': sum(row['methods'][method]['correct_against_old_GT'] for row in gt_only_rows),
            'V11_GT_correct_rows': sum(row['methods'][method]['correct_against_V11_GT'] for row in gt_only_rows),
            'rescues_switching_to_old_GT': sum(not row['methods'][method]['correct_against_V11_GT'] and row['methods'][method]['correct_against_old_GT'] for row in gt_only_rows),
            'breaks_switching_to_old_GT': sum(row['methods'][method]['correct_against_V11_GT'] and not row['methods'][method]['correct_against_old_GT'] for row in gt_only_rows),
        }
        score_summary[method] = {
            'all_complete_prompt_mapped_old_groups': subset_score(complete, method),
            'same_ID_complete_old_groups': subset_score(same_ID_complete, method),
            'cross_ID_complete_old_groups': subset_score(cross_ID_complete, method),
            'old_only_ID_recovered_complete_groups': subset_score(old_only_recovered, method),
            'GT_changed_complete_groups': subset_score(gt_affected_complete, method),
            'GT_only_row_effect': row_changes,
        }
    summary = {
        'schema': 'cross-ID-exact-native-prompt-old-version-projection-v1',
        'matching_policy': 'Exact native prompt including hint/question/ordered options, GT excluded. Prefer same row ID only if exact prompt matches; otherwise require exactly one candidate. Ambiguous candidates are not selected. No option or predicted-letter remapping.',
        'evaluation_policy': 'Use original old-version GT on unchanged predicted letters; do not claim old GT is more correct. Only complete old rotation groups receive circular scores.',
        'source_sha256': {str(path): sha(path) for path in [OLD, NEW, *PRED.values(), Path(__file__)]},
        'old_rows': len(old),
        'old_groups': len(old_groups),
        'V11_rows': len(new),
        'V11_groups': len(new_groups),
        'row_mapping_counts': dict(counts),
        'old_complete_prompt_mapped_groups': len(complete),
        'old_unmapped_or_ambiguous_groups': len(old_groups) - len(complete),
        'complete_group_coverage_percent': 100 * len(complete) / len(old_groups),
        'old_only_ID_groups': len(old_only),
        'old_only_ID_complete_groups_recovered_cross_ID': len(old_only_recovered),
        'old_only_ID_recovered_group_ids': sorted(old_only_recovered),
        'unmapped_or_ambiguous_old_group_ids': sorted(set(old_groups) - set(complete)),
        'complete_groups_with_GT_changes': len(gt_affected_complete),
        'method_scores': score_summary,
        'can_claim_complete_legacy_run': False,
        'image_boundary': 'This is a prompt-only projection. Images must independently match or be explicitly accepted for reuse; most same-ID legacy/V11 images are recompressed. The subset scores cannot substitute for full1176-group old-version performance.',
        'future_join': 'Image audit can join each old_rows_mapping.json item by old_index and selected_v11_index_or_null.',
    }
    for filename, content in [
        ('old_rows_mapping.json', mapping),
        ('old_groups_scores.json', group_report),
        ('GT_only_changed_rows.json', gt_only_rows),
        ('summary.json', summary),
    ]:
        (DEST / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n')
    with (DEST / 'old_rows_mapping.csv').open('w', newline='') as stream:
        fields = ['old_index', 'old_base_id', 'old_GT', 'selected_v11_index_or_null', 'selected_V11_GT', 'GT_changed', 'match_status', 'candidate_v11_indices']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(mapping)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
