"""Read completed CPU arrays only; separate common-history gains from masks."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPECTED = 'feb116dec1014cd71fbcd74287d291b342d20b78ac14c6e74269aad5373d6afe'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import numpy as np
    path = HERE / 'comparison.json'
    assert sha(path) == EXPECTED
    report = json.loads(path.read_text())
    assert report['status'] == 'complete' and report['n_questions'] == 16
    results = []
    for record in report['records']:
        artifact = record['CPU_output_arrays']
        assert sha(artifact['path']) == artifact['sha256']
        assert sha(record['cache']['path']) == record['cache']['sha256']
        with np.load(artifact['path'], allow_pickle=False) as arrays:
            source = arrays['source__gains'][:, 0]
            eq7 = arrays['eq7_exact_denominator__gains'][:, 0]
        first = record['source_vs_eq7']['first_ordered_divergence']
        stop = 256 if first is None else first + 1
        valid = (source[:stop] > -1e8) & (eq7[:stop] > -1e8)
        assert np.all(valid.any(axis=1))
        maximum = float(np.abs(source[:stop] - eq7[:stop])[valid].max())
        common = dict(steps=stop, max_abs_gain_delta_on_unselected_candidates=maximum,
            excludes_selected_mask_sentinel=True, same_selection_history_before_every_compared_step=True)
        divergence = None
        if first is not None:
            a, b = record['source_ordered_ids'][first], record['eq7_ordered_ids'][first]
            divergence = dict(zero_based_step=first, source_pick=a, eq7_pick=b,
                source_gains_at_two_picks=[float(source[first,a]),float(source[first,b])],
                eq7_gains_at_two_picks=[float(eq7[first,a]),float(eq7[first,b])],
                source_pair_margin=float(source[first,a]-source[first,b]),
                eq7_pair_margin=float(eq7[first,b]-eq7[first,a]))
        with np.load(record['cache']['path'], allow_pickle=False) as original:
            visual = original['visual_features_BF16_bits']
            native_final = original['native_pruned_BF16_bits']
            source_final = visual[np.sort(record['source_ordered_ids'])]
            eq7_final = visual[np.sort(record['eq7_ordered_ids'])]
            assert np.array_equal(source_final,native_final) and np.array_equal(eq7_final,native_final)
            final = dict(source_equals_native_BF16_bytes=True, eq7_CPU_equals_native_BF16_bytes=True,
                native_sha256=hashlib.sha256(native_final.tobytes()).hexdigest(),
                shape=list(native_final.shape), scope='Cached projected BF16 sequence sorted by spatial IDs, not an LLM generation')
        results.append(dict(position=record['position'], index=record['index'],
            common_history_gain_comparison=common, first_divergence=divergence,
            whole_trace_max_delta_including_mask_sentinels=record['source_vs_eq7']['gain_trace']['max_abs_delta'],
            final_projected_feature_sequence=final))
    result = dict(status='complete',n_questions=16,comparison_sha256=EXPECTED,
        script_sha256=sha(__file__),CPU_only=True,model_loaded=False,GPU_started=False,
        source_GPU_anchor_not_bitwise=True,all_facility_results_are_CPU_conditional_references=True,
        maximum_common_history_valid_gain_delta=max(x['common_history_gain_comparison']['max_abs_gain_delta_on_unselected_candidates'] for x in results),
        records=results,limitations=[
            'The1e9 whole-trace difference is caused by selected-mask sentinel locations after differing selection histories; it is not a numerical overflow.',
            'Common-history differences exclude candidates already masked as selected and end at the first divergent argmax inclusive.',
            'Cached sorted BF16 feature identity does not measure downstream logits, natural termination, accuracy, or another GPU variant.'])
    assert sha(path)==EXPECTED and 'torch' not in sys.modules
    with (HERE / 'gain_and_final_sequence_boundary.json').open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','limitations']}))


if __name__ == '__main__':
    main()
