"""Assemble the independent CPU protocol audit, without modifying experiments."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import re
import subprocess

ROOT = Path('/media/disk2/YZX/research/EADP_amp')
OUT = Path(__file__).resolve().parent
OFF = Path('/media/disk2/YZX/research/audit_base_gap_20261003/eadp_official/LLaVA')
WRAP = ROOT / 'Qwen_vl/scripts/stage1_roundtrip_pilot'
EV = ROOT / 'LLaVA/playground/data/eval'
COMMIT = 'e1a08801461c181871b9ff1cb0803b8e9966f083'
BASE_URL = f'https://github.com/SJTU-DeepVisionLab/EADP/blob/{COMMIT}/LLaVA/'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def evidence(p, patterns=()):
    p = Path(p)
    r = {'path': str(p), 'sha256': sha(p)}
    if patterns:
        r['matching_line_numbers'] = {pattern: [i for i,line in enumerate(p.read_text().splitlines(),1) if pattern in line] for pattern in patterns}
    if p.is_relative_to(OFF):
        r['primary_url'] = BASE_URL + p.relative_to(OFF).as_posix()
    return r


def main():
    cpu = json.load((OUT / 'cpu_protocol_identity_and_scores.json').open())
    assert cpu['all_cpu_identity_checks_pass']
    pope = json.load((OUT / 'pope_original_primary_GT_verification.json').open())
    scripts = {}
    defaults = []
    for model in ('v1_5','v1_6'):
        for task in ('vqav2','gqa','vizwiz','vizwiz_val','sqa','textvqa','pope','mme','mmbench','mmbench_cn','mmvet'):
            p = OFF / f'scripts/{model}/eval/{task}.sh'
            text = p.read_text()
            alpha = float(re.search(r'^ALPHA=\$\{3:-([\d.]+)\}', text, re.M)[1])
            beta = float(re.search(r'^BETA=\$\{2:-([\d.]+)\}', text, re.M)[1])
            entry = re.search(r'python -m (llava\.eval\.\w+)', text)[1]
            scripts[f'{model}/{task}'] = evidence(p, ('ALPHA=', 'BETA=', '--lang', '--conv-mode', 'SPLIT='))
            decoder = 128 if entry.endswith('model_vqa_loader') else 1024
            local_beta = 1.0 if task == 'mmvet' else 2.0
            defaults.append({'model': model, 'task': task, 'official_released_alpha_default':alpha,
                             'official_released_beta_default':beta, 'official_entry':entry,
                             'official_released_max_new_tokens':decoder, 'official_conv_mode':'vicuna_v1',
                             'local_current_alpha':0.5, 'local_current_beta':local_beta,
                             'alpha_matched':alpha == 0.5, 'beta_matched':beta == local_beta,
                             'paper_actual_per_task_argv':'UNPUBLISHED/UNVERIFIED',
                             'scope_note':'vizwiz_val is relevant to starred validation results; vizwiz is the separate historical test entry. FULL bypasses pruning, so alpha/beta differences do not by themselves change FULL selection.'})
    assert len(defaults) == 22
    sources = {
        'official_scripts':scripts,
        'official_mmbench_native':evidence(OFF / 'llava/eval/model_vqa_mmbench.py', ('max_new_tokens=', "args.lang == 'cn'", 'cur_prompt = question', '--lang', '--all-rounds')),
        'local_mmbench_native':evidence(ROOT / 'LLaVA/llava/eval/model_vqa_mmbench.py', ('max_new_tokens=', "args.lang == 'cn'", 'cur_prompt = question', '--lang', '_img_map', '--all-rounds')),
        'official_loader':evidence(OFF / 'llava/eval/model_vqa_loader.py', ('question.replace', '--max_new_tokens', 'texts=question')),
        'local_model_vqa_wrapper':evidence(WRAP / 'llava_eval_arm_model_vqa.py', ('generation_text_settings', 'return cur_prompt, 1024', 'official_textvqa')),
        'local_loader_wrapper':evidence(WRAP / 'llava_eval_arm_loader.py', ('question.replace', '--max_new_tokens')),
        'local_science_wrapper':evidence(WRAP / 'llava_eval_arm_science.py', ('single_pred_prompt', 'question.replace', 'max_new_tokens=')),
        'official_science_native':evidence(OFF / 'llava/eval/model_vqa_science.py', ('single_pred_prompt', 'question.replace', 'max_new_tokens=')),
        'released_vizwiz_scorer':evidence(OFF / 'scripts/eval_vizwiz.py', ('match_count', 'min(1.0', 'processor(ans)', 'continue')),
        'released_vizwiz_converter':evidence(OFF / 'scripts/convert_vizwiz_val_to_llava.py', ('prompt_suffix', "'question_id'")),
        'official_benchmark_vizwiz_API':evidence(Path('/media/disk2/YZX/research/audit_rescore_20261008/scripts_snapshot/vizwiz_official_vqaEval.py'), ('otherGTAns', 'avgGTAcc')),
        'local_official_live_scorer':evidence(WRAP / 'official_score.py', ('VizWizNormalizer', 'three-category mean F1', 'checked_predictions')),
        'local_official_math':evidence(WRAP / 'rescore_official_20261008.py', ('def vizwiz_official_score', 'other =', 'def gqa_official_score')),
        'local_mmbench_circular_scorer':evidence(WRAP / 'mmben_circular_score.py', ('extract_letter', '1000000', 'all(x == 1')),
        'local_vlmevalkit_matcher':evidence(ROOT / 'Qwen_vl/VLMEvalKit/vlmeval/utils/matching_util.py', ('def can_infer_option', 'def can_infer_text', 'def can_infer(')),
        'local_v15_driver':evidence(WRAP / 'llava_lane_a_v15.sh', ('gen_mmb ()', '--single-pred-prompt', 'gen_mmb mmbcn')),
        'local_next_driver':evidence(WRAP / 'llava_lane_b_next.sh', ('gen_mmb ()', '--single-pred-prompt', 'gen_mmb next_mmbcn')),
        'mme_canonical_scorer':evidence(WRAP / 'mme_canonical_score.py', ('GT_ARCHIVE_SHA256', 'PERCEPTION', 'parse_prediction', '100 * (acc + acc_plus)')),
        'official_POPE_scorer':evidence(OFF / 'llava/eval/eval_pope.py', ('zip(pred_list, label_list)', 'avg_f1')),
        'cpu_collector':evidence(OUT / 'collect_protocol_cpu.py'),
        'cpu_report_builder':evidence(Path(__file__)),
        'cpu_identity_and_scores':evidence(OUT / 'cpu_protocol_identity_and_scores.json'),
        'mmbcn_prompt_identity':evidence(OUT / 'mmbcn_prompt_identity.json'),
        'pope_primary_GT_verification':evidence(OUT / 'pope_original_primary_GT_verification.json'),
    }
    sources['official_benchmark_vizwiz_API']['primary_url'] = 'https://vizwiz.org/wp-content/uploads/2020/06/API.zip'
    sources['local_vlmevalkit_matcher']['primary_reference_url'] = 'https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/utils/matching_util.py'
    findings = []
    def add(id,status,title,detail,**kwargs):
        findings.append(dict(id=id,status=status,title=title,detail=detail,**kwargs))
    add('MMBCN_LANG','confirmed_project_protocol_error','Chinese MMBench used English option-answer instruction',
        'Both official EADP model-family scripts explicitly pass --lang cn; both historical local lane scripts omit it and native default is en. All 4876 LLM prompts change under cn, while EADP guidance and saved prompt fields are unchanged. Existing stream-only controls deliberately retain en and do not fix this separate mismatch.',
        evidence_keys=['official_scripts.v1_5/mmbench_cn','official_scripts.v1_6/mmbench_cn','local_mmbench_native','local_v15_driver','local_next_driver'],
        affected_complete_old_arms=cpu['mmbench']['CN']['completed_predictions'],
        cpu_prompt_identity=cpu['mmbench']['CN'],
        independent_correction_scope={'existing_arms':6,'questions_each':4876,'predictions':29256,'change_only':{'lang':'cn'},'retain_other_fields':['V11 input bytes','alpha=.5','beta=2','max_new_tokens=1024','vicuna_v1','single_pred_prompt=True','all_rounds=False','wait-only patch'], 'scheduled_by_this_audit':False},
        accuracy_effect='Unmeasured; no assumed improvement. Guidance suffix identity means this is primarily an LLM prompt correction, not a different pruning guidance string.')
    add('MMB_SPLIT_VERSION','confirmed_protocol_difference_effect_unmeasured','Current V11 differs from released LLaVA/EADP 2023 datasets',
        'Official scripts name EN20230712/CN20231003; current generation uses official V11 TSVs. EN local legacy file has4377 rows/1176 base groups versus V114876/1292. Shared base IDs do not establish question, GT, or pixel identity. Actual paper input bytes and matching-server configuration remain unavailable.',
        evidence_keys=['official_scripts.v1_5/mmbench','official_scripts.v1_6/mmbench','official_scripts.v1_5/mmbench_cn','official_scripts.v1_6/mmbench_cn'],
        en_exact_comparison=cpu['mmbench']['EN']['legacy_split_comparison'],
        image_identity_limit='RGB hashes include dimensions. Exact pixels agree for7 shared IDs; dimensions agree for950. dHash agreement875/950 and low distances in74 others are approximate visual evidence, not proof of identity or wrong-image evidence.',
        cn_old_split_content_identity='Not verified: old CN20231003 TSV was not available locally. Do not transfer EN counts to CN.',
        required_control='A separate clearly labeled released-2023-split EADP/AnchorZip comparison at identical K and corrected CN language, with exact frozen source/input/image manifest and the same scoring protocol. Existing V11 controls remain valuable within-version comparisons; do not compare V11 directly to paper as a proven same-dataset reproduction.')
    add('VIZWIZ_GENERATION','confirmed_protocol_difference_effect_unmeasured','VizWiz uses model_vqa instead of released loader',
        'All4319 validation questions and image filenames exactly match the released conversion. LLM question text is unchanged. Local model_vqa guidance retains the single-word answer instruction; official EADP loader removes that instruction. Decoder limit is1024 locally versus128 in the released loader. The Unanswerable instruction remains in both guidance strings.',
        evidence_keys=['official_scripts.v1_5/vizwiz_val','official_scripts.v1_6/vizwiz_val','official_loader','local_model_vqa_wrapper','released_vizwiz_converter'],
        n=4319,n_changed_guidance=4319,independent_parameter_differences={'released_val_alpha':0.0,'local_alpha':0.5,'released_beta':1.0,'local_beta':2.0},
        accuracy_effect='Unmeasured. A larger decoding bound only changes generations that would otherwise reach it; this audit does not assume truncation or effect.',
        required_controls=['With wait-only and all other fields fixed, change only guidance suffix removal; save full runtime counts and prediction identity.', 'Separately change only max_new_tokens1024→128; do not combine that with guidance, alpha or beta when assigning causality.', 'If evaluating the public default recipe, preregister one released-alpha/beta recipe separately and retain all outcomes; its result is not proof of paper argv.'])
    add('VIZWIZ_SCORING','published_code_vs_benchmark_metric_difference','Official local score differs from EADP released validation scorer',
        'Published EADP eval_vizwiz.py uses normalized-GT min(match_count/3), while the benchmark API averages scores leaving each of10 annotator answers out. Current local official scorer follows the benchmark API and is correct. The auxiliary published-code score is computed on the same complete4319 raw answers, without editing them or replacing official scores. Paper scorer identity is unverified.',
        evidence_keys=['released_vizwiz_scorer','official_benchmark_vizwiz_API','local_official_live_scorer','local_official_math'],
        formulas={'released_code':'min(1, count(processed_GT == processed_prediction)/3)',
                  'benchmark_official':'mean_i min(1, count_{j!=i}(GT_j == benchmark_processed_prediction)/3)',
                  'ten_annotators_example':'With3 matching annotations: naive1.0 versus leave-one-out0.9; with1 or2:1/3 versus0.3,2/3 versus0.6.'},
        complete_same_prediction_comparison=cpu['vizwiz']['complete_prediction_scores'],
        paper_split_boundary='Paper starred VizWiz numbers are validation; historical unstarred test numbers are a separate split. Printed FULL val references v15=55.6 and NeXT=60.9 are not local generated FULL results.',
        required_control='CPU only: retain benchmark-official score as headline; show auxiliary released-code score when describing cross-code reproduction residual. No new GPU run is required for this metric difference.')
    add('PUBLIC_ALPHA_BETA_DEFAULTS','released_defaults_differ_paper_argv_unverified','Per-task alpha and beta defaults differ from local frozen recipe',
        'Both families: POPE default alpha1.0 vs local.5; VizWiz validation defaultalpha0 vs local.5; other listed task defaultsalpha.5 match. Released beta1 differs from localbeta2 for scored pruning tasks. Local MMVetbeta1 matches. The VizWiz test script usesalpha.5, which must not be substituted for its validation-script default. Shell defaults and ablation table cells do not identify actual paper main-table argv.',
        all_task_defaults=defaults,evidence_keys=['official_scripts'],
        required_control='Do not alter stream-only queues. One explicitly named public-default sensitivity control per selected task is admissible if separately authorized; single-factor comparisons are needed to attribute effects to alpha or beta. No parameter sweep or selecting the highest-scoring budget.')
    add('MMB_DECODER_1024','excluded_as_project_error','MMBench1024 matches both official EADP and upstream LLaVA native code',
        'Official model_vqa_mmbench.py hardcodes1024, as does the local native entry; upstream LLaVA main also hardcodes1024. A generic loader128 default is irrelevant to this MMBench entry.',
        evidence_keys=['official_mmbench_native','local_mmbench_native'],
        upstream_primary_url='https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_mmbench.py')
    add('MMB_CIRCULAR_COVERAGE','excluded_current_coverage_or_parser_error','All V11 rotations are generated and grouped correctly',
        'Each of12 historical complete files contains4876 ordered unique indices and matching native saved prompts, with round_id0. Grouping1292 base IDs includes all3584 rotation rows. For all rows, the local letter parser equals VLMEvalKit no-judge can_infer and all options/GT text are consistent across provided permutations.28 groups retainD and permuteABC; that is present in official TSV and is not an omitted generated rotation. No extra --all-rounds should be added to already-expanded V11 data.',
        complete_arms={k:v['completed_predictions'] for k,v in cpu['mmbench'].items()},
        group_sizes={k:v['group_size_distribution'] for k,v in cpu['mmbench'].items()},
        published_matching_server_identity='Unverified; CPU agreement with toolkit no-judge extraction does not establish identity of a historical external GPT matching service.')
    add('SQA_AUTHOR_INPUT_IDENTITY','author_input_provenance_unverified','Shipped CQM-A is exact; author script CQM-I is unavailable',
        'EADP released scripts name CQM-I; paper does not explicitly identify that file format. Actual author file bytes are unavailable. Current2017 image subset exactly equals shipped LLaVA eval.zip CQM-A image entries. Existing format-control results establish sensitivity to prompt formatting, not identity of author CQM-I. Current option-answer instruction, guidance removal, vicuna_v1 and decoder1024 match released science entry.',
        cpu_identity=cpu['scienceqa'],evidence_keys=['official_scripts.v1_5/sqa','official_scripts.v1_6/sqa','official_science_native','local_science_wrapper'],
        required_control='No further guessed reconstruction. If actual author CQM-I becomes available, first verify question/image/answer identities and run a separate frozen-format comparison. Existing CQM-A scores must be labeled shipped-LLaVA protocol.')
    add('POPE_8910','excluded_missing_random90_project_error','8910 exactly matches the official LLaVA eval archive and pinned POPE release',
        'Official original POPE release linked by LLaVA containsrandom2910/popular3000/adversarial3000; archive question bytes match current8910 exactly. Original category order and all identity-paired labels match the retained questions. Current randomGT3000 contains extra90, which are outside this user-approved as-asked input; current identity pairing is correct. Original paper exact sample bytes are still not independently available.',
        cpu_identity=cpu['pope'],original_primary_GT=pope,
        metric='Unweighted mean of three category F1 scores; not pooled F1 or one category accuracy.',
        evidence_keys=['official_POPE_scorer','local_official_live_scorer'],
        required_control='No additional90-question generation or another already-completePOPE32 synchronization replay for this count issue.')
    add('MME_GT_METRIC','already_fixed_cpu_scoring_error','MME canonical2374 labels and perception score are already corrected',
        '162 historical reconstructed GT labels differed from the official evaluator archive. Canonical scorer uses verified archive2374 keys, exactly two questions per image, official first-four-character answer parsing, and sum of category100*(accuracy+both-answers accuracy). LLaVA paper table uses perception raw points separately from cognition; it is not a percentage. These label errors need CPU rescoring, independently of currently queued stream-wait generation repairs.',
        cpu_question_identity=cpu['mme'],canonical_artifact=evidence(ROOT / 'Qwen_vl/outputs/audit_followup_20261008/mme_official_gt_rescore.json'),
        evidence_keys=['mme_canonical_scorer'],
        benchmark_primary_url='https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation',
        required_control='No additional generation because of labels alone; do not replace perception with perception+cognition or divide raw points by an unrelated maximum.')
    add('GQA_INPUT_SCORE','excluded_new_input_split_or_prompt_error','GQA testdev input matches shipped primary question file and GT',
        'All12578 input bytes match LLaVA eval.zip; all GT keys, question strings after removing the answer instruction, and image filename stems match the official testdev_balanced GT. Current complete raw scoring uses lowercase exact match. No new split or prompt defect found; beta default provenance remains a separate issue.',cpu_identity=cpu['gqa'])
    add('TEXTVQA_INPUT_GUIDANCE','already_repaired_protocol_difference','Current TextVQA official mode has repaired guidance and decoder',
        'Official OCR5000 input bytes match LLaVA eval.zip. Earlier model_vqa kept the answer instruction in pruning guidance and used1024. Current official_textvqa mode removes that suffix while preserving the LLM prompt and sets128. Current stream controls address a separate execution issue and must not be labeled as recovering the paper2.204pp residual before full results.',cpu_identity=cpu['textvqa'],
        evidence_keys=['official_loader','local_model_vqa_wrapper'])
    add('MMVET_UNSCORED','excluded_unavailable_final_metric','MMVet cannot establish method or paper accuracy without its official grader',
        'Question bytes match shipped eval.zip. Local decoder1024 and MMVetalpha.5/beta1 match released entry. Official GPT grading output is unavailable; do not substitute an invented exact-match score or add unscored arms to an average.',cpu_identity=cpu['mmvet'],
        evidence_keys=['official_scripts.v1_5/mmvet','official_scripts.v1_6/mmvet'])
    report = {
        'schema':'additional-reproduction-protocol-audit-v1',
        'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Read-only LLaVA benchmark protocol audit beyond CUDA waits; no environment, queue, production source, parameter, prediction or frozen worker mutation.',
        'git_branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        'live_git_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'handoff_lines_read':906,
        'official_EADP_commit':COMMIT,
        'paper_primary_url':'https://arxiv.org/html/2607.02484v1',
        'upstream_LLaVA_evaluation_primary_url':'https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md',
        'scientific_limits':['Static recipe differences do not prove an accuracy increase or identify paper main-table argv.',
                             'Within-version method comparisons require complete same-input same-budget EADP and AnchorZip arms; most historical MMBench/VizWiz arms have no EADP mate.',
                             'JPEG/pixel equality and approximate dHash visual similarity are different facts; low dHash distance is not exact input identity.',
                             'Current stream-only queues intentionally preserve their frozen historical language, max tokens and alpha/beta. Additional corrections must use independent provenance and outputs.'],
        'sources':sources,'findings':findings,
        'counts_by_status':dict(collections.Counter(f['status'] for f in findings)),
        'cpu_checks_pass':True,'gpu_launches':0,'live_files_modified':0,
    }
    for cat, info in pope.items():
        assert info['n_original_gt'] == info['n_asked']
        assert info['asked_label_difference_original_vs_current'] == 0
        assert info['asked_original_order_keys_exact']
    path = OUT / 'protocol_differences.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'report':str(path),'sha256':sha(path),'n_findings':len(findings),'counts_by_status':report['counts_by_status']}))


if __name__ == '__main__':
    main()
