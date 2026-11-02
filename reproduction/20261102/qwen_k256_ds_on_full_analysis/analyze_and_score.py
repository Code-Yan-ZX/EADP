"""CPU-only terminal analysis of one complete EADP DeepStack-on run.

The complete/exit gate is deliberately checked before opening observations or
ground truth. This script never loads a model. Run with CUDA_VISIBLE_DEVICES=''.
"""
import ast
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

ROOT = Path('/media/disk2/YZX/research/EADP_v2')
Q = ROOT / 'reproduction/20261028/qwen'
RUN = ROOT / 'reproduction/20261102/qwen_k256_ds_on_full_perf'
HERE = Path(__file__).resolve().parent
MANIFEST = Q / 'doc_full_two_routes/input_manifest.jsonl'
PAIR = Q / 'doc_full_k256_ds_pair'
OLD = Q / 'gpu_formula_facility_panel_run02/run_plan.json'
CANONICAL = Q / 'independent_audit/DocVQA_VAL.canonical.jsonl'
AUDIT = Q / 'independent_audit/audit_qwen.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def gate(run, expected_n):
    """Do not access generation records until both exit proofs are validated."""
    run = Path(run)
    proof_path, complete_path = run / 'root_exit_proof.json', run / 'complete.json'
    proof, complete = read(proof_path), read(complete_path)
    require(proof.get('root_owned_exec_actual_exit_code') == 0, 'root exec not proved successful')
    require(proof.get('container_exit_code') == 0 and proof.get('original_container_exited') is True,
            'container exit not proved successful')
    require(proof.get('root_session_id') is not None, 'root session missing')
    require(proof.get('complete_sha256') == sha(complete_path), 'complete hash mismatch')
    filename = proof.get('process_exit_file')
    require(isinstance(filename, str) and Path(filename).name == filename and filename.startswith('process_exit_'),
            'invalid process exit filename')
    exit_path = run / filename
    require(proof.get('process_exit_sha256') == sha(exit_path), 'process exit hash mismatch')
    require(read(exit_path).get('container_exit_code') == 0, 'process exit is not successful')
    require(complete.get('n') == expected_n, 'not a full dataset')
    require(proof.get('observations_sha256') == complete.get('observations_sha256'), 'proof/complete disagreement')
    observation_path = run / 'observations.jsonl'
    require(sha(observation_path) == complete['observations_sha256'], 'observations hash mismatch')
    inputs = read(run / 'inputs.json')
    require(inputs == complete.get('inputs'), 'complete inputs differ')
    return {'root_exit_proof_sha256': sha(proof_path), 'complete_sha256': sha(complete_path),
            'process_exit_sha256': sha(exit_path), 'observations_sha256': sha(observation_path),
            'inputs': inputs}


def quantile(values, q):
    a = sorted(values)
    pos = (len(a) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return a[low] + (a[high] - a[low]) * (pos - low)


def distribution(values):
    require(bool(values) and all(math.isfinite(x) for x in values), 'nonfinite or empty metric')
    return dict(mean=statistics.mean(values), p50=quantile(values, .5), p90=quantile(values, .9),
                p95=quantile(values, .95), p99=quantile(values, .99), min=min(values), max=max(values))


def summarize(rows, manifest, prior_predictions, prior_observations):
    n = len(manifest)
    require(n > 0 and len(rows) == len(prior_predictions) == len(prior_observations) == n, 'row counts differ')
    text_match = token_match = selection_match = eos = truncated = 0
    for i, (r, m, p, o) in enumerate(zip(rows, manifest, prior_predictions, prior_observations)):
        require(all(x['position'] == i for x in (r, m, p, o)), 'position mismatch')
        require(len({str(x['index']) for x in (r, m, p, o)}) == 1, 'index mismatch')
        require(r['injected'] == 1 and len(r['ordered_ids']) == len(set(r['ordered_ids'])) == 256,
                'missing injection or invalid selection')
        require(isinstance(r['response'], str) and bool(r['output_ids']) and
                r['output_tokens'] == len(r['output_ids']), 'invalid output')
        require(r['seconds'] > 0 and r['prefill_gpu_ms'] >= 0 and
                0 < r['peak_allocated_bytes'] <= r['peak_reserved_bytes'], 'invalid measurements')
        t = r['response'] == p['response']; ids = r['output_ids'] == o['output_ids']
        sel = r['ordered_ids'] == o['ordered_ids']
        require(r['prior_response_match'] == t and r['prior_output_ids_match'] == ids and
                r['prior_selection_match'] == sel, 'stored comparison flag mismatch')
        require(sel, 'actual prior selection changed')
        text_match += t; token_match += ids; selection_match += sel
        eos += r['eos_present']; truncated += r['at_max_tokens'] and not r['eos_present']
    values = lambda name, div=1: distribution([r[name] / div for r in rows])
    elapsed = sum(r['seconds'] for r in rows)
    tokens = sum(r['output_tokens'] for r in rows)
    return dict(n=n, seconds=values('seconds'), prefill_gpu_ms=values('prefill_gpu_ms'),
                peak_allocated_MiB=values('peak_allocated_bytes', 1024**2),
                peak_reserved_MiB=values('peak_reserved_bytes', 1024**2),
                output_tokens=values('output_tokens'), total_measured_seconds=elapsed,
                total_output_tokens=tokens, questions_per_s=n / elapsed, output_tokens_per_s=tokens / elapsed,
                eos_present=eos, no_eos=n-eos, max_tokens_without_eos=truncated,
                prior_response_matches=text_match, prior_output_id_matches=token_match,
                prior_selection_matches=selection_match,
                limitations=['Natural output lengths vary and affect elapsed time.',
                             'Wall time includes image read, processor, visual tower, selector, generation, decode and light hooks; excludes model loading and validation.',
                             'First language model CUDA event includes DeepStack injection; excludes visual tower and selection.',
                             'Peak memory is whole-inference PyTorch allocated/reserved including the model; excludes allocations outside PyTorch.',
                             'Single DeepStack-on arm: cannot infer a fair on/off cost difference from old instrumented full runs.'])


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'set CUDA_VISIBLE_DEVICES to empty for CPU-only scoring')
    evidence = gate(RUN, 5349)
    identity = evidence['inputs']
    for path, key in [(MANIFEST, 'manifest_sha256'), (OLD, 'old_plan_sha256'),
                      (PAIR / 'execution/ds_on.ordered.predictions.jsonl', 'prior_prediction_sha256'),
                      (PAIR / 'execution/ds_on.ordered.observations.jsonl', 'prior_observation_sha256'),
                      (RUN / 'benchmark.py', 'code_sha256')]:
        require(sha(path) == identity[key], 'input hash changed: ' + str(path))
    manifest = jsonl(MANIFEST)
    observations = jsonl(RUN / 'observations.jsonl')
    previous_predictions = jsonl(PAIR / 'execution/ds_on.ordered.predictions.jsonl')
    previous_observations = jsonl(PAIR / 'execution/ds_on.ordered.observations.jsonl')
    report = summarize(observations, manifest, previous_predictions, previous_observations)
    for path, digest in read(PAIR / 'plan.json')['score_only_sha256'].items():
        require(sha(path) == digest, 'scoring source changed: ' + path)
    canonical = jsonl(CANONICAL)
    require(len(canonical) == 5349, 'canonical length')
    for i, (gt, m) in enumerate(zip(canonical, manifest)):
        require(str(gt['index']) == str(m['index']) and gt['question'] == m['question'] and m['position'] == i,
                'canonical mismatch')

    # The released metric reads the predictions back from XLSX, which may turn
    # literal strings such as "NA" into NaN; keep this conversion observable.
    out = HERE / 'results'
    out.mkdir(exist_ok=False)
    import pandas as pd
    old = read(OLD)
    sys.path[:0] = [old['author_root'], str(Path(old['author_root']) / 'VLMEvalKit')]
    from vlmeval.dataset.image_vqa import ImageVQADataset
    spec = importlib.util.spec_from_file_location('qualified_audit_qwen', AUDIT)
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    import torch
    require(not torch.cuda.is_initialized(), 'CUDA initialized')
    table = pd.DataFrame(canonical)
    table['prediction'] = [r['response'] for r in observations]
    xlsx = out / 'new.xlsx'; table.to_excel(xlsx, index=False)
    dataset = ImageVQADataset.__new__(ImageVQADataset); dataset.dataset_name = 'DocVQA_VAL'
    aggregate = dataset.evaluate(str(xlsx))
    detail = out / 'new_results.xlsx'; native = pd.read_excel(detail).to_dict(orient='records')
    require(len(native) == len(canonical), 'native scored row count')
    previous_scores = jsonl(PAIR / 'score/ds_on.per_question.jsonl')
    require(sha(PAIR / 'score/ds_on.per_question.jsonl') == read(PAIR / 'score/score_report.json')['scores']['ds_on']['per_question_sha256'], 'prior score hash changed')
    require(len(previous_scores) == len(canonical), 'old score row count')
    per = []
    for gt, row, orig, prev in zip(canonical, observations, native, previous_scores):
        require(audit.normalized_id(orig['index']) == str(gt['index']) == str(prev['index']), 'score row ID mismatch')
        converted = audit.scoring_prediction(row['response'])
        independent = audit.independent_anls(converted, audit.parse_answers(gt['answer']))
        distances = ast.literal_eval(str(orig['eval_match']))
        official = 1. - min(distances); official = official if official >= .5 else 0.
        require(abs(independent - official) < 1e-12, 'pointwise official vs independent mismatch')
        diff = independent - prev['independent_score']
        per.append(dict(index=str(gt['index']), prediction=row['response'], scored_prediction=converted,
                        native_NA_conversion=row['response'] != converted, official_score=official,
                        independent_score=independent, prior_ds_on_score=prev['independent_score'], difference=diff))
    official_mean = float(aggregate['Overall'].iloc[0])
    independent_mean = 100 * sum(r['independent_score'] for r in per) / len(per)
    require(abs(official_mean-independent_mean) < 1e-10, 'aggregate mismatch')
    require(not torch.cuda.is_initialized(), 'CUDA initialized during scoring')
    per_path = out / 'per_question.jsonl'
    with per_path.open('x', encoding='utf-8') as f:
        for r in per: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    report['scores'] = dict(official=official_mean, independent=independent_mean,
                            full_denominator=len(per), native_NA_conversions=sum(r['native_NA_conversion'] for r in per),
                            prior_score_improved=sum(r['difference'] > 0 for r in per),
                            prior_score_declined=sum(r['difference'] < 0 for r in per),
                            prior_score_unchanged=sum(r['difference'] == 0 for r in per),
                            per_question_sha256=sha(per_path), native_input_sha256=sha(xlsx),
                            native_detail_sha256=sha(detail), all_pointwise_equal=True)
    report['evidence'] = evidence
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
