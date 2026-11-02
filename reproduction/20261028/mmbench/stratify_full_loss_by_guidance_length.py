"""CPU post-hoc grouping of completed FULL/pruned losses by CLIP prompt length."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
OUT = BASE / 'full_vs_pruned_complete'
PAIRED_SHA = '10a488d7026235a2fefd63ada2c7bd2c4cf9534043429370573899408a0bbecd'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('completed_FULL_pair_gates', BASE / 'compare_full_vs_pruned.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.gates()  # Terminal metadata, frozen hashes and actual process exits.
    report_path = OUT / 'comparison.json'
    assert sha(report_path) == PAIRED_SHA
    original = read(report_path)
    assert original['success'] and original['n_rows'] == 4377 and original['n_groups'] == 1176
    for entry in original['sources']:
        assert sha(entry['path']) == entry['sha256'], entry['path']
    for key in ('paired_rows', 'paired_groups'):
        entry = original[key]
        assert sha(entry['path']) == entry['sha256']
    rows = [json.loads(line) for line in Path(original['paired_rows']['path']).read_text().splitlines()]
    groups = [json.loads(line) for line in Path(original['paired_groups']['path']).read_text().splitlines()]
    assert len(rows) == 4377 and len(groups) == 1176

    current_path, pinned_path = BASE / 'cpu_inputs_current.json', BASE / 'cpu_inputs_author_pins.json'
    equivalence_path = BASE / 'cpu_dependency_input_equivalence.json'
    equivalence = read(equivalence_path)
    assert equivalence['success'] and equivalence['all_CLIP_guidance_rows_compared'] == 4377
    for entry in equivalence['inputs']:
        assert sha(entry['path']) == entry['sha256']
    current, pinned = read(current_path), read(pinned_path)
    assert current['guidance'] == pinned['guidance']
    assert current['dataset_sha256'] == pinned['dataset_sha256'] == '52d45f29876e56138d3d4feb169dc9a7323f6ed2ee3f439ae7e4af94192b3324'
    assert current['n_rows'] == current['all_guidance_rows'] == len(current['guidance']) == 4377
    assert len({entry['index'] for entry in current['guidance']}) == 4377
    group_lengths = {}
    row_lengths = []
    for row, entry in zip(rows, current['guidance']):
        assert row['index'] == entry['index'] and row['prompt_sha256'] == entry['prompt_sha256']
        assert entry['original_token_shape'][0] == 1
        length = entry['original_token_shape'][1]
        assert length > 0 and entry['padded_shape'] == [(length + 76) // 77, 77]
        row_lengths.append(length)
        group_lengths.setdefault(row['base_group'], []).append(length)
    assert sum(length > 77 for length in row_lengths) == 703 and max(row_lengths) == 582
    assert set(group_lengths) == {entry['base_group'] for entry in groups}
    strata = {}
    for name, is_long in [('any_rotation_gt77', True), ('all_rotations_le77', False)]:
        selected = [entry for entry in groups if (max(group_lengths[entry['base_group']]) > 77) == is_long]
        ids = [entry['base_group'] for entry in selected]
        result = dict(n_groups=len(ids), n_rows=sum(len(group_lengths[idx]) for idx in ids),
                      n_rows_gt77=sum(length > 77 for idx in ids for length in group_lengths[idx]),
                      group_ids=ids, runs={}, paired={})
        for run in ('FULL', 'K128_beta1', 'K128_beta2'):
            correct = sum(entry['runs'][run] for entry in selected)
            result['runs'][run] = dict(correct_groups=correct, accuracy_percent=100 * correct / len(ids))
        for run in ('K128_beta1', 'K128_beta2'):
            counts = gate.transitions([entry['runs'][run] for entry in selected], [entry['runs']['FULL'] for entry in selected])
            net = counts['wrong_to_correct'] - counts['correct_to_wrong']
            result['paired'][run + '_to_FULL'] = dict(circular_group_transitions=counts,
                net_groups_restored_by_FULL=net,
                FULL_minus_pruned_pp_within_stratum=100 * net / len(ids),
                contribution_to_overall_FULL_minus_pruned_pp=100 * net / 1176)
        strata[name] = result

    assert sum(item['n_groups'] for item in strata.values()) == 1176
    assert sum(item['n_rows'] for item in strata.values()) == 4377
    for run in ('FULL', 'K128_beta1', 'K128_beta2'):
        assert sum(item['runs'][run]['correct_groups'] for item in strata.values()) == original['scores'][run]['public_correct_groups']
    for run in ('K128_beta1', 'K128_beta2'):
        pair = run + '_to_FULL'
        for transition, expected in original['paired'][pair]['circular_group_transitions'].items():
            assert sum(item['paired'][pair]['circular_group_transitions'][transition] for item in strata.values()) == expected
    sources = [report_path, current_path, pinned_path, equivalence_path, Path(__file__),
               Path(original['paired_rows']['path']), Path(original['paired_groups']['path'])]
    result = dict(success=True, CPU_only=True, model_loaded=False, GPU_or_API_used=False,
        n_rows=4377, n_groups=1176, guidance_rows_gt77=703, max_guidance_tokens=582,
        fixed_stratum_rule='A circular group is long if any native rotation guidance prompt has more than77 CLIP tokens, including tokenizer special tokens.',
        direction='Pruned -> FULL: wrong_to_correct is restoration by FULL; correct_to_wrong is degradation by FULL.',
        strata=strata, sources=[dict(path=str(p), sha256=sha(p)) for p in sources],
        limitations=['Post-hoc descriptive analysis authorized after complete scores; not preregistered or a modified score protocol.',
                    'Uses existing actual CPU CLIP tokenizer/segmentation records matched to every native prompt hash. No new tokenization, model execution, API, truncation or segmentation variant is performed.',
                    'Strata describe local pruning losses under these complete controls. Paper per-question predictions are unavailable, so these associations do not assign the paper gap to long guidance.',
                    'The existing primary score and all denominators remain unchanged; no score-direction selection or new GPU run is triggered.'])
    output = OUT / 'guidance_length_stratification.json'
    with output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    assert 'torch' not in sys.modules
    print(json.dumps({key: result[key] for key in ('success', 'n_rows', 'n_groups', 'guidance_rows_gt77', 'max_guidance_tokens')}))
    print(json.dumps({name: {key: value for key, value in item.items() if key != 'group_ids'} for name, item in strata.items()}))


if __name__ == '__main__':
    main()
