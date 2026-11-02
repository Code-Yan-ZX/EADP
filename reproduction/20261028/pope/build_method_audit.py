#!/usr/bin/env python3
"""Assemble a bounded paper/source audit from completed CPU/GPU evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
AUTHOR = Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp')
OLD = Path('/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/audit_followup_20261008')
PAPER = Path('/media/disk2/YZX/research/EADP_v2/Qwen_vl/outputs/audit_followup_20261008/submission_tables_20261009/paper_tables_source.html')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    probe = json.loads((ROOT / 'cpu_dense_token_audit.json').read_text())
    full = json.loads((ROOT / 'full_control_comparison.json').read_text())
    previous = json.loads((OLD / 'next_runtime_repair_20261009/panel64/result.json').read_text())
    protocol = json.loads((OLD / 'next_runtime_repair_20261009/panel64/protocol.json').read_text())
    assert previous['complete'] and previous['n'] == 64
    assert sha(OLD / 'next_runtime_repair_20261009/panel64/records.jsonl') == previous['records_sha256']
    assert sha(OLD / 'next_runtime_repair_20261009/panel64/protocol.json') == previous['protocol_sha256']
    config = json.loads(Path('/media/disk2/YZX/doct/FastV/llava-v1.5-7b/config.json').read_text())
    panel = probe['panel_rows']
    masses = [r['padding_aggregation_weight_mass'] for r in panel]
    changes = [r['comparisons']['nonpadding_nonEOS']['importance_comparison']['0.5'] for r in panel]
    evidence_paths = [PAPER, OLD / 'llava_paper_configuration_ambiguities.json',
        OLD / 'next_runtime_repair_20261009/panel64/result.json', OLD / 'next_runtime_repair_20261009/panel64/protocol.json',
        OLD / 'next_runtime_repair_20261009/panel64/records.jsonl',
        OLD / 'next_gap_diagnosis_20261008/paper_config/paper_config_audit.json',
        ROOT / 'cpu_dense_token_audit.py', ROOT / 'cpu_dense_token_audit.json', ROOT / 'cpu_formula_boundary_checks.json',
        ROOT / 'full_control_comparison.json', ROOT / 'historical_alpha05_beta2_repeat_consistency.json',
        AUTHOR / 'LLaVA/llava/model/llava_arch.py', AUTHOR / 'LLaVA/llava/model/multimodal_encoder/clip_encoder.py',
        AUTHOR / 'LLaVA/llava/model/builder.py', AUTHOR / 'LLaVA/scripts/v1_5/eval/pope.sh',
        AUTHOR / 'LLaVA/llava/eval/model_vqa_loader.py']
    audit = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'author_commit': full['source_commit'],
        'paper_url': 'https://arxiv.org/html/2607.02484v1',
        'scope': 'CPU-only method audit after two complete POPE runs. No third model, source changes, Git mutations, or score-selected parameters.',
        'completed_controls': {
            'same_alpha1_beta1': {'source_as_is_full_F1': full['source_as_is']['official_macro_f1_percent'],
                'wait_only_full_F1': full['wait_only']['official_macro_f1_percent'],
                'same_input_conflicts_before': 231, 'same_input_conflicts_after': 0, 'repeated_groups': 2334},
            'historical_alpha05_beta2_wait_full_F1': full['historical_alpha05_beta2_waited_f1'],
            'paper_K32_F1': 86.7, 'attribution_limit': 'Historical .5/2 and current 1/1 differ in two parameters; their score difference does not isolate alpha or beta. Both are stable on all2334 observed repeated inputs after waits.'},
        'explicit_paper_and_source_matches': [
            {'item': 'negative cross-modal cosine', 'paper': 'Appendix9 explicitly requires the negative cosine despite positive signs in main Eq2/3.', 'source': 'llava_arch.py:190-192', 'status': 'matched; reversing the sign would contradict the appendix'},
            {'item': 'entropy temperature', 'paper': 'Appendix12 explicitly sets tau=.01.', 'source': 'llava_arch.py:211,586 multiplies logits by100.', 'status': 'matched'},
            {'item': 'dense weighting', 'paper': 'Eq7 negative entropy softmax; Appendix12 gamma=.01.', 'source': 'llava_arch.py:274-279,589 uses negative_entropy with denominator .01+1e-6.', 'status': 'matched structure; tiny numeric epsilon difference, no evidence of benchmark gap cause'},
            {'item': 'spatial prior', 'paper': 'Eq11 and Appendix9 kernel3, sigma1, reflection padding before power.', 'source': 'llava_arch.py:289-340,604-607', 'status': 'matched'},
            {'item': 'facility objective and nonnegative rebound', 'paper': 'Eq13/14 and Appendix9 Eq24 use weighted marginal coverage and .5*(cos+1).', 'source': 'llava_arch.py:168-177,445-494', 'status': 'matched for v15 single patch; no 5x budget ambiguity applies to POPE v15'}],
        'configuration_identity_not_closed': {
            'alpha_means_paper_mu': True, 'source_POPE_shell': {'alpha': 1., 'beta': 1.},
            'source_loader_and_README_example': {'alpha': .5, 'beta': 1.},
            'source_active_q': .2,
            'paper_explicit': {'mu_domain': '[0,1]', 'q_domain': '(0,1]', 'tau': .01, 'gamma': .01,
                'Eq12_description': 'beta>1 for sharpening; Sec5.6 also discusses robustness plateau beta>=1.'},
            'paper_ablation_only': {'mu_peak_average': .5, 'q_peak_average': .3, 'beta_grid': [0., .5, 2., 5., 10.]},
            'mechanistic_implication': 'alpha1 exactly removes finite dense guidance from fusion; beta1 is the identity power. Thus the released POPE shell is a boundary variant with no dense contribution and no nonlinear sharpening. These argv are source-supported, but not proven to generate paperTable1.',
            'rejected_inference': 'Neither a best ablation cell nor a repeated table value pins the main-table q/mu/beta. No beta10, q.5 or q.3 is promoted as the paper recipe.'},
        'specific_discrepancies': [
            {'id': 'dense_candidate_domain', 'paper': 'Eq3 explicitly defines dense similarities on non-EOS CLIP text tokens; padding positions are not prompt tokens.',
             'source': 'clip_encoder.py:109-132 zero-pads to77, passes mask only to text encoder, projects every last_hidden_state position. llava_arch.py:192,222,586 filters all77 positions with no mask.',
             'status': 'verified implementation versus described candidate-domain difference',
             'all_actual_prompts': {'unique': 79, 'rows': 8910, 'lengths_including_BOS_EOS': probe['unpadded_length_distribution_unique_prompts'],
                 'source_candidates_per_prompt': 77, 'source_keep_at_q02': 15},
             'fixed16_CPU_real_feature_probe': {'selected_padding_of_240': probe['summary']['total_selected_padding'],
                 'selected_EOS_of_240': probe['summary']['total_selected_EOS'],
                 'padding_weight_mass_mean': statistics.mean(masses), 'padding_weight_mass_max': max(masses),
                 'images_with_padding_weight_gt_1percent': sum(m > .01 for m in masses),
                 'alpha05_beta2_importance_changed_images': sum(not x['importance_exact_equal'] for x in changes),
                 'alpha05_beta2_importance_max_abs_delta': max(x['importance_max_abs_difference'] for x in changes),
                 'alpha1_beta1_all16_importance_bit_equal_after_masking': probe['summary']['all_alpha1_importance_exact_equal']},
             'interpretation': 'Actual domain discrepancy, but selected-token count overstates causal weight. Applying q after candidate masking also reduces retained count from15 to1-2, so this comparison does not isolate padding weight alone. This cannot explain current1/1 gap because dense is multiplied by0. CPU real-feature differences are not generated-answer or benchmark effects.',
             'prior_negative_control': {'model_task': 'NeXT TextVQA fixed64, not POPE full', 'result': previous['summary'],
                 'protocol_arms': protocol['arms'], 'conclusion': 'Valid-content masking, bounded chunks and both already failed to recover panel accuracy. Do not rediscover these as proven fixes or blindly rerun the same panel.'}},
            {'id': 'normalization_numerator_epsilon', 'paper': 'Eq10 and Algorithm1 add epsilon only to denominator.',
             'source': 'llava_arch.py:598-599 adds1e-6 to both numerator and denominator.',
             'status': 'verified arithmetic discrepancy; synthetic boundary check only',
             'testable_effect': 'Constant fused scores normalize to all1 in source versus all0 in paper formula; nonconstant scores differ by a uniform epsilon/(range+epsilon) before smoothing/power. No actual degenerate POPE map or answer effect established.'},
            {'id': 'quantile_threshold_vs_fixed_topk', 'paper': 'Eq6 retains h_i<=q-quantile, including threshold ties.',
             'source': 'llava_arch.py:222-226 takes max(1,floor(L*q)) smallest entropies.',
             'status': 'verified formulation/implementation edge difference; sample quantile convention unspecified',
             'testable_effect': 'For9 distinct entropy values andq.2, common quantile threshold retains2 versus source1. Ties can increase the difference. No current answer-level consequence established.'}],
        'dismissed_FP16_NaN_hypothesis': {
            'synthetic_fact': 'Half-precision1e-12 rounds to0 and0*log(0) can produceNaN.',
            'native_v15_construction': 'config unfreeze_mm_vision_tower=False delays CLIP load until after LLM.from_pretrained; native builder defaultdevice_map=auto skips tower.to(float16); standalone CLIP loaders omittorch_dtype.',
            'config_unfreeze_mm_vision_tower': config.get('unfreeze_mm_vision_tower'),
            'CPU_mirrored_parameter_dtypes': probe['cpu_native_loader_parameter_dtypes'],
            'CPU_cross_similarity_dtype': sorted({r['cross_similarity_dtype'] for r in panel}),
            'CPU_entropy_nan_count': probe['summary']['total_entropy_nans'],
            'conclusion': 'Not supported as a cause of completed native v15 runs. CPU mirror establishes source-default dtype behavior; finished GPU intermediates were not directly instrumented.'},
        'bounded_next_actions': [
            'Keep1/1 source andwait fullnegative results; no q/mask modification can alter finitealpha1fusion.',
            'If a future slot is approved, the distinct source-supported generic/README .5/1 recipe is a single declared configuration control, not a parameter search or certified paperargv. Do not launch automatically.',
            'Before any numerical algorithm variant, inspect an instrumented existing-recipe run for actual epsilon/span and normalization-degenerate maps; change only one specified arithmetic operation. Such instrumentation is not presently scheduled.',
            'No current evidence justifies a negative-cos sign change, guessed q, switched allocation, or renewed masking/chunk panel as a reproduction repair.'],
        'evidence_registry': {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in evidence_paths}}
    (ROOT / 'paper_method_source_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    text = f'''# POPE paper/source method audit, 2026-10-28

The paper gap remains unresolved. Source-as-is K32/alpha1/beta1 gives 77.3936863461 F1; adding only the three caller-stream dependencies gives 82.2202369991, versus paper86.7. Observed identical-input conflicts fall231/2334 to0/2334. Historical .5/2 wait is84.0735996644 with0/2334 conflicts; changing two parameters together does not identify their separate effects.

The released POPE shell uses alpha1/beta1. Alpha is paper mu, so dense guidance has zero finite contribution, and beta1 leaves smoothing unsharpened. The generic loader and README instead default .5/1. The paper explicitly fixes tau=.01/gamma=.01 and spatial3/sigma1/reflect; those match. Its main-table mu/q/beta argv remain unavailable. Best ablation cells cannot supply that identity.

There is a concrete dense-token domain mismatch: paper Eq3 uses non-EOS tokens, while source filters all77 sequence outputs, including BOS/EOS and mask0 zero-padding. All79 distinct POPE prompts have only10–12 nonpadded positions, but source retains15 candidates. In a deterministic16-image CPU feature diagnostic,185/240 selected positions are padding and5 areEOS. Padding aggregation mass averages{statistics.mean(masses):.6%}, peaks{max(masses):.6%}, and exceeds1% for only1 image. Masking nonpadding/nonEOS changes .5/2 importance on7/16 images (maximum absolute change .004930757), but all16 alpha1/beta1 importance maps remain exactly equal. Applying q after masking also lowers retained count from15 to1–2; this does not isolate padding weight alone. This discrepancy therefore cannot explain the present1/1 paper gap.

Prior NeXT fixed64 generation already gives wait56.40625, valid_dense54.375, bounded_chunks56.40625, and both55.9375. These are negative mechanism controls, not full-benchmark scores. They prevent treating masking/chunking as a recovered recipe or repeating that panel without a new hypothesis.

Two smaller exact formula differences remain testable: source Eq10 adds epsilon to the numerator as well as denominator, and Eq6 uses fixed floor-topk rather than a threshold quantile with ties. Synthetic checks show the boundaries, including constant-map1 versus0. No actual degenerate POPE map, answer change, or gap attribution has been established.

The FP16 entropy-NaN concern is unsupported for native v15 defaults: unfreeze_mm_vision_tower is false, the delayed standalone CLIP loaders useFP32, and default device_map=auto skips a later FP16 tower cast. The CPU native-construction mirror confirms FP32 vision/projection/text/cross-sim and no entropyNaNs across16 real images. This is not direct introspection of completed GPU tensors. Appendix9 explicitly requires negative cross-modal cosine, so the source sign matches the paper.

Artifacts: `paper_method_source_audit.json` records paper/source anchors, all evidence hashes and bounded next actions; `cpu_dense_token_audit.json` stores actual IDs/masks and per-image entropies/weights; `cpu_formula_boundary_checks.json` contains declared synthetic checks. CPU images preserve native FP32→FP16→FP32 input rounding. The endpoint-spaced panel rule was fixed in the script before computation and differs from the previous midpoint-spaced image-archive SHA panel. An initial unrounded CPU diagnostic is retained with its limitation labeled. No LLM, GT labels, generation, GPU context, source edits, or third POPE model was used for this audit.
'''
    (ROOT / 'paper_method_source_audit.md').write_text(text)
    print(json.dumps({'audit': str(ROOT / 'paper_method_source_audit.json'), 'sha256': sha(ROOT / 'paper_method_source_audit.json'),
        'selected_padding': probe['summary']['total_selected_padding'], 'padding_weight_mean': statistics.mean(masses),
        'padding_weight_max': max(masses), 'prior_panel_verified': True}))


if __name__ == '__main__':
    main()
