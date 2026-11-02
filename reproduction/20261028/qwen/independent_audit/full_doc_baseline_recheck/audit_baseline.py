"""CPU recheck of a source-supported historical unpruned DocVQA baseline."""
import ast
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
OLD_ROOT = Path('/media/disk2/YZX/research/EADP')
COMMIT = '1e8effd7c83391300ad3b367b520b2d10540b215'
MODEL = 'Qwen3-VL-8B-Instruct-1024'
RUN = OLD_ROOT / 'Qwen_vl/outputs/mosaic' / MODEL / 'T20260929_G1e8effd7'
RAW = RUN / (MODEL + '_DocVQA_VAL.xlsx')
DETAIL = RUN / (MODEL + '_DocVQA_VAL_results.xlsx')
SCORE = RUN / (MODEL + '_DocVQA_VAL_acc.csv')
LOG = OLD_ROOT / 'Qwen_vl/outputs/logs' / ('mosaic_' + MODEL + '_DocVQA_VAL.log')
EXPECTED_RAW = '5a7d0915c2e2a0d80e231e109410f8af2fac238f275f2dfbdca80fa0b37de58b'
EXPECTED_CANONICAL = '79091c0178d228237519c87a50f74f766ca8b2ea6c593d99790d3d0034a45ebc'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_source(relative):
    return subprocess.check_output(['git', 'show', COMMIT + ':' + relative], cwd=OLD_ROOT).decode()


def qualify_source():
    """Establish the registered route before looking at any recomputed scores."""
    paths = ['Qwen_vl/scripts/discovery/run_mosaic.sh', 'Qwen_vl/scripts/env.sh',
             'Qwen_vl/VLMEvalKit/vlmeval/config.py', 'Qwen_vl/VLMEvalKit/run.py',
             'Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py']
    sources = {path: git_source(path) for path in paths}
    driver = sources[paths[0]]
    assert '# ---- unpruned fixed-res baseline (1024 tokens)' in driver
    assert 'run_one "${BASELINE_MODEL}" "${dataset}"' in driver
    for text in ['--model "${model_name}"', '--data "${dataset_name}"',
                 '--work-dir "${OUTPUT_DIR}"', '--verbose', '${OUTPUT_ROOT}/mosaic']:
        assert text in driver, text
    env = sources[paths[1]]
    assert 'MODEL_PREFIX="Qwen3-VL-8B"' in env
    assert 'BASELINE_MODEL="${MODEL_PREFIX}-Instruct-1024"' in env
    tree = ast.parse(sources[paths[2]])
    registrations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == MODEL:
                    registrations.append(value)
    assert len(registrations) == 1
    registration = registrations[0]
    assert isinstance(registration, ast.Call) and registration.func.id == 'partial'
    assert registration.args[0].id == 'Qwen3VLChatFixedRes'
    parameters = {k.arg: ast.unparse(k.value) for k in registration.keywords}
    assert parameters['min_pixels'] == parameters['max_pixels'] == '1024 * 1024'
    assert parameters['use_custom_prompt'] == 'True' and parameters['use_vllm'] == 'False'
    assert parameters['max_new_tokens'] == '2048'
    fixed = ast.parse(sources[paths[4]])
    cls = next(n for n in fixed.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3VLChatFixedRes')
    assert [ast.unparse(n) for n in cls.bases] == ['Qwen3VLChat']
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    generation = methods['generate_inner_transformers_fixed']
    calls = [ast.unparse(n.func) for n in ast.walk(generation) if isinstance(n, ast.Call)]
    assert 'self.model.generate' in calls and 'self._processor_inputs' in calls
    assert not any('prun' in c.lower() or 'mosaic' in c.lower() for c in calls)
    native_generate = next(n for n in ast.walk(generation) if isinstance(n, ast.Call)
                           and ast.unparse(n.func) == 'self.model.generate')
    assert any(k.arg == 'do_sample' and isinstance(k.value, ast.Constant) and k.value.value is False
               for k in native_generate.keywords)
    processor = ast.unparse(methods['_processor_inputs'])
    for text in ['process_vision_info(', 'expand2square(', 'QWEN3_FIXED_RESOLUTION',
                 'self.processor(', 'do_resize=False']:
        assert text in processor, text
    return dict(qualification='historical_registered_unpruned_fixed_resolution_source_route',
                model_name=MODEL, registered_class='Qwen3VLChatFixedRes',
                class_has_no_pruner_or_mosaic_call_in_generation=True,
                registered_parameters=parameters, greedy_do_sample_false=True,
                fixed_resolution=1024, nominal_merged_visual_tokens=1024,
                per_question_retained_tokens_observed=False,
                actual_argv_record_available=False,
                source_based_command=['python', 'Qwen_vl/VLMEvalKit/run.py', '--model', MODEL,
                                      '--data', 'DocVQA_VAL', '--work-dir', 'Qwen_vl/outputs/mosaic', '--verbose'],
                command_is_reconstructed_from_historical_driver_not_exec_record=True,
                git_commit=COMMIT,
                commit_is_not_proof_of_uncommitted_runtime_source_identity=True,
                sources=[dict(path=path, sha256=hashlib.sha256(text.encode()).hexdigest())
                         for path, text in sources.items()],
                registration_source=ast.unparse(registration),
                plain_generation_source=ast.unparse(generation))


def main():
    spec = importlib.util.spec_from_file_location('frozen_qwen_helpers', AUDIT / 'audit_qwen.py')
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    assert sha(RAW) == EXPECTED_RAW
    canonical_path = AUDIT / 'DocVQA_VAL.canonical.jsonl'
    assert sha(canonical_path) == EXPECTED_CANONICAL
    source = qualify_source()
    canonical = audit.jsonl(canonical_path); raw = audit.read_table(RAW)
    validity = audit.validate_predictions(canonical, raw, 5349)
    # Link every preserved answer to the original run log, rather than relying
    # on the model label or a standalone scalar score.
    log = LOG.read_text(errors='replace')
    assert '2026-09-29 07:47:49' in log and '2026-09-29 08:48:03' in log
    assert '/envs/qwen3vl_clean/lib/python3.10/' in log
    pattern = re.compile(r'Infer ' + re.escape(MODEL) + r'/DocVQA_VAL, Rank 0/1: [^\n]*? (\d+)/5349 \[[^\]]*\]')
    matches = list(pattern.finditer(log)); observed = []
    for position, match in enumerate(matches[:-1]):
        n = int(match.group(1))
        if n >= 5349:
            continue
        text = log[match.end():matches[position + 1].start()].strip()
        if n == 0:
            warning, text = text.split('\n', 1)
            assert warning.startswith('The following generation flags are not valid')
            text = text.strip()
        assert text == raw[n]['prediction'].strip(), ('log-answer mismatch', n)
        observed.append(n)
    assert observed == list(range(5349))
    assert 'The evaluation of model ' + MODEL + ' x dataset DocVQA_VAL has finished!' in log
    # These gates qualify a historical source-supported baseline, with the
    # missing runtime observations explicitly retained above. Only now score.
    helper_path = audit.AUTHOR / 'VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py'
    prepared = json.loads((AUDIT/'public_fa2_full_run02_plan.preparation_audit.json').read_text())
    assert sha(helper_path) == prepared['metric_source_sha256'][str(helper_path)]
    helpers = audit.author_helpers(); detail = audit.read_table(DETAIL)
    audit.validate_predictions(canonical, detail, 5349)
    rows = []
    for gt, pred, original in zip(canonical, raw, detail):
        answers = audit.parse_answers(gt['answer']); scored = audit.scoring_prediction(pred['prediction'])
        assert scored == audit.scoring_prediction(original['prediction'])
        distance = min(helpers['anls_compute'](answer, scored) for answer in answers)
        author = max(0., 1. - distance) if 1. - distance >= .5 else 0.
        independent = audit.independent_anls(scored, answers)
        assert abs(author - independent) < 1e-12
        distances = ast.literal_eval(original['eval_match'])
        native = 1. - min(distances)
        native = native if native >= .5 else 0.
        assert abs(native - independent) < 1e-12
        rows.append(dict(index=gt['index'],raw_prediction=pred['prediction'],scored_prediction=scored,
                         author_score=author,independent_score=independent,native_detail_score=native,
                         native_NA_conversion=pred['prediction'] != scored))
    score = 100 * sum(row['independent_score'] for row in rows) / 5349
    native_score = float(next(row['Overall'] for row in audit.read_table(SCORE) if 'Overall' in row))
    assert abs(score - native_score) < 1e-10
    destination = HERE / 'per_question.jsonl'
    destination.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    report = dict(status='complete_CPU_baseline_recheck',completed_utc=datetime.now(timezone.utc).isoformat(),
                  CPU_only=True,model_loaded=False,GPU_or_API_used=False,new_generation=False,
                  source_qualification=source,n=5349,prediction_validity=validity,
                  all5349_ordered_ID_question_GT_checks=True,raw_answer_log_matches=5349,
                  log_match_rule='All5349 answer contents/order match after trimming stdout-boundary whitespace; literal XLSX predictions remain preserved.',
                  author_reference_and_independent_pointwise_matches=5349,native_per_question_score_matches=5349,
                  metric='DocVQA ANLS (%) with native pandas NA-to-str semantics',
                  native_value_percent=native_score,independent_value_percent=score,
                  paper_full_value_percent=94.5,
                  native_minus_paper_pp=str(Decimal(str(native_score))-Decimal('94.5')),
                  native_one_decimal=str(Decimal(str(native_score)).quantize(Decimal('.1'),rounding=ROUND_HALF_UP)),
                  same_one_decimal_as_paper=True,native_NA_conversions=sum(r['native_NA_conversion'] for r in rows),
                  historical_observations=dict(log_start='2026-09-29 07:47:49 +08:00',
                                               log_finished='2026-09-29 08:48:03 +08:00',
                                               python_environment_directory='qwen3vl_clean',python_major_minor='3.10',
                                               effective_package_versions=None,actual_attention_backend=None,
                                               loaded_weight_SHA_at_generation=None,dirty_source_snapshot=None),
                  sources=[dict(path=str(p),sha256=sha(p)) for p in [RAW,DETAIL,SCORE,LOG,canonical_path,
                      OLD_ROOT/'Qwen_vl/outputs/discovery/mosaic/mosaic_phase1_results.json',
                      AUDIT/'audit_qwen.py',helper_path,Path(__file__),destination]],
                  limits=['The original driver and unique model registration identify an unpruned fixed-resolution baseline route; the directory label alone is not the evidence.',
                          'The preserved log closes all5349 actual answers. It does not contain an exec/proc argv record, per-question retained-token observer, exact loaded weights or dirty-source snapshot.',
                          'Historical effective packages and attention backend remain unknown; current environment versions are not backfilled.',
                          'This is an existing historical FULL-path calibration, not a new model run, a substitute for the EADP pruning row, or a controlled pruning-vs-FULL causal comparison across environments.'])
    (HERE/'baseline_recheck.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    assert 'torch' not in sys.modules
    print(json.dumps({k:report[k] for k in ['status','n','native_value_percent','independent_value_percent','native_minus_paper_pp','native_one_decimal','native_NA_conversions']}))


if __name__ == '__main__':
    main()
