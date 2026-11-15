"""CPU-only, complete-denominator paired QA analysis for attention calibration.

Input: {complete: true, config: {manifest_sha256, planned_arms, baseline_arm?},
        records: [{item_id, parent_id, question, answers,
                   predictions: {ARM: {text, truncated, token_ids?}}}]}.
All frozen primary questions, including completed wrong/empty answers and
truncations, must be present. Missing/failed runs remain pending for completion.
"""
from __future__ import annotations

import argparse
import ast
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re

import numpy as np

QWEN = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = QWEN / 'outputs/attention_calibration/data/manifest.json'
SEED = 20261109
DRAWS = 10000
_PRIMARY_NAMESPACE = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def primary_functions():
    """Extract exactly the historical pure functions without importing torch."""
    global _PRIMARY_NAMESPACE
    if _PRIMARY_NAMESPACE is None:
        path = QWEN / 'scripts/cf_mining/cf_mine.py'
        names = {'normalize', 'correct'}
        functions = [node for node in ast.parse(path.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in functions} != names:
            raise ValueError('Missing historical numeric exact functions')
        namespace = {'re': re, 'Decimal': Decimal, 'InvalidOperation': InvalidOperation}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        _PRIMARY_NAMESPACE = namespace
    return _PRIMARY_NAMESPACE


def normalize(value):
    return primary_functions()['normalize'](value)


def score_exact(prediction, answers):
    return any(primary_functions()['correct'](prediction, answer) for answer in answers)


def scorers():
    official = QWEN / 'VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py'
    source = official.read_text()
    names = {'levenshtein_distance', 'anls_compute', 'hit_calculate'}
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError('Missing checked-in official ANLS functions')
    namespace = {'np': np, 'listinstr': lambda patterns, text: any(p in text for p in patterns)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(official), 'exec'), namespace)

    def score_anls(prediction, answers):
        distances = [namespace['anls_compute'](answer, prediction) for answer in answers]
        return float(namespace['hit_calculate']([{'match': distances}], 'DocVQA')[0])

    return score_anls, dict(
        primary_exact=dict(source=str(QWEN / 'scripts/cf_mining/cf_mine.py'),
                           sha256=sha256(QWEN / 'scripts/cf_mining/cf_mine.py'),
                           extracted_functions=['correct', 'normalize'],
                           semantics='Use original CF normalize/correct exactly: strip outer quotes; lowercase; remove whitespace/currency/comma; normalize Unicode minus and Decimal; preserve percent suffix/sign. Leading zeros are not preserved; report literal exact sensitivity.'),
        secondary_anls=dict(source=str(official), sha256=sha256(official),
                            functions=sorted(names),
                            semantics='Official original-string normalized distance followed by best-alias DocVQA hit/similarity threshold; no numeric canonicalization'),
        literal_strict_exact='Generated text equals an original alias byte-for-byte, without stripping or normalization')


def summarize(vectors, rows, arms, contrasts, binary=True):
    if not rows:
        return dict(n=0, documents=0, arms={}, paired={}, note='Empty sensitivity subgroup; not estimable')
    parents = sorted({row['parent_id'] for row in rows})
    indices = [[i for i, row in enumerate(rows) if row['parent_id'] == parent] for parent in parents]
    draws = np.random.default_rng(SEED).integers(0, len(parents), size=(DRAWS, len(parents)))
    by_doc = {arm: np.asarray([np.mean(np.asarray(vectors[arm], dtype=float)[ii]) for ii in indices])
              for arm in arms}

    def interval(vector):
        return np.quantile(vector[draws].mean(axis=1), [.025, .975]).tolist()

    result = dict(n=len(rows), documents=len(parents), arms={}, paired={})
    for arm in arms:
        raw = np.asarray(vectors[arm], dtype=float)
        entry = dict(n=len(rows), documents=len(parents), mean=float(by_doc[arm].mean()),
                     micro_mean=float(raw.mean()), bootstrap_95ci=interval(by_doc[arm]))
        if binary:
            entry.update(correct=int(raw.sum()), accuracy=float(by_doc[arm].mean()))
        result['arms'][arm] = entry
    for before, after in contrasts:
        delta = np.asarray(vectors[after], dtype=float) - np.asarray(vectors[before], dtype=float)
        doc_delta = by_doc[after] - by_doc[before]
        entry = dict(before=before, after=after, n=len(rows), documents=len(parents),
                     delta_mean=float(doc_delta.mean()), micro_delta_mean=float(delta.mean()),
                     bootstrap_95ci=interval(doc_delta), improved=int((delta > 0).sum()),
                     worsened=int((delta < 0).sum()), tied=int((delta == 0).sum()))
        if binary:
            entry.update(delta_accuracy=float(doc_delta.mean()), rescue=int((delta == 1).sum()),
                         breakage=int((delta == -1).sum()))
        if bool((delta == 0).all()):
            entry['interval_note'] = 'Identical observed outcomes; a degenerate interval does not establish algorithmic equivalence.'
        result['paired'][after + '_minus_' + before] = entry
    return result


def analyze(payload, manifest=DEFAULT_MANIFEST):
    manifest = Path(manifest)
    frozen = json.loads(manifest.read_text())
    if frozen.get('complete') is not True or frozen.get('role') != 'primary':
        raise ValueError('Require complete primary manifest; development pages cannot enter primary analysis')
    config = payload['config']
    if payload.get('complete') is not True:
        raise ValueError('Primary analysis requires a completed run')
    if config.get('manifest_sha256') != sha256(manifest):
        raise ValueError('Runner manifest hash differs from frozen manifest')
    arms = config.get('planned_arms', config.get('arms'))
    if not isinstance(arms, list) or not arms or not all(isinstance(a, str) for a in arms) or len(set(arms)) != len(arms):
        raise ValueError('config.planned_arms must enumerate unique frozen arm names')
    baseline = config.get('baseline_arm', 'baseline')
    if baseline not in arms:
        raise ValueError('config.baseline_arm must name an evaluated arm')
    contrasts = config.get('contrasts', [[baseline, arm] for arm in arms if arm != baseline])
    if any(len(pair) != 2 or pair[0] not in arms or pair[1] not in arms or pair[0] == pair[1]
           for pair in contrasts):
        raise ValueError('Invalid frozen pairwise contrast')
    keys = [after + '_minus_' + before for before, after in contrasts]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate pairwise contrasts')
    rows = payload['records']
    items = frozen['items']
    if len(rows) != len(items) or config.get('sample_count', len(items)) != len(items):
        raise ValueError('Must complete every frozen primary question, including wrong answers and truncations')
    by_id = {item['item_id']: item for item in items}
    if len(by_id) != len(items) or len({row['item_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate frozen or predicted item ID')
    if {row['item_id'] for row in rows} != set(by_id):
        raise ValueError('Predictions do not cover exactly the frozen questions')
    if len({row['parent_id'] for row in rows}) != len(rows):
        raise ValueError('This one-QA-per-parent panel contains duplicated source parents')
    # Use frozen order so replaying the same outputs has the same bootstrap.
    prediction_by_id = {row['item_id']: row for row in rows}
    rows = [prediction_by_id[item['item_id']] for item in items]
    anls, scoring = scorers()
    scores = {metric: {arm: [] for arm in arms} for metric in ('exact', 'anls', 'literal')}
    detail = []
    for row in rows:
        item = by_id[row['item_id']]
        for field in ('parent_id', 'question', 'answers'):
            if row[field] != item[field]:
                raise ValueError(f'Frozen annotation changed: {row["item_id"]} {field}')
        answers = row['answers']
        if not answers or not all(isinstance(a, str) for a in answers):
            raise ValueError('Missing original answer aliases')
        record = dict(item_id=row['item_id'], parent_id=row['parent_id'],
                      question=row['question'], answers=answers,
                      image_sha256=item['image_sha256'], predictions={})
        if set(row['predictions']) != set(arms):
            raise ValueError('Missing or undeclared arm in predictions')
        for arm in arms:
            prediction = row['predictions'][arm]
            if prediction.get('error'):
                raise ValueError(f'Operationally failed run remains pending: {row["item_id"]} {arm}')
            text = prediction['text']
            if not isinstance(text, str) or not isinstance(prediction['truncated'], bool):
                raise ValueError('Prediction needs string text and explicit boolean truncated')
            exact = bool(score_exact(text, answers))
            if 'correct' in prediction and bool(prediction['correct']) != exact:
                raise ValueError('Runner/scorer numeric exact disagreement')
            literal = any(text == answer for answer in answers)
            secondary = anls(text, answers)
            for metric, value in [('exact', exact), ('anls', secondary), ('literal', literal)]:
                scores[metric][arm].append(value)
            record['predictions'][arm] = dict(text=text, exact=exact, anls=secondary,
                                              literal_strict_exact=literal,
                                              numeric_literal_disagreement=exact != literal,
                                              truncated=prediction['truncated'],
                                              error=prediction.get('error'),
                                              eos=prediction.get('eos'),
                                              final_token_id=prediction.get('token_ids', [None])[-1] if prediction.get('token_ids') else None,
                                              token_count=len(prediction.get('token_ids', [])))
        detail.append(record)
    primary = summarize(scores['exact'], rows, arms, contrasts)
    secondary = summarize(scores['anls'], rows, arms, contrasts, binary=False)
    literal = summarize(scores['literal'], rows, arms, contrasts)
    for arm in arms:
        primary['arms'][arm].update(
            truncated=sum(r['predictions'][arm]['truncated'] for r in rows),
            operational_errors=sum(bool(r['predictions'][arm].get('error')) for r in rows),
            secondary_anls=secondary['arms'][arm]['mean'],
            literal_strict_exact=literal['arms'][arm]['correct'])
    untruncated = [i for i, row in enumerate(rows)
                   if all(not row['predictions'][arm]['truncated'] for arm in arms)]
    sensitivity = dict(n=len(untruncated), included_item_ids=[rows[i]['item_id'] for i in untruncated],
                       excluded_item_ids=[rows[i]['item_id'] for i in range(len(rows)) if i not in untruncated])
    for metric, binary in [('exact', True), ('anls', False), ('literal', True)]:
        sensitivity[metric] = summarize(
            {arm: [scores[metric][arm][i] for i in untruncated] for arm in arms},
            [rows[i] for i in untruncated], arms, contrasts, binary)
    return dict(schema_version=1, complete=True, n=len(rows), documents=primary['documents'],
                denominator='Every preselected primary source document/QA after all runs complete; no correctness or truncation filtering; missing/operationally failed runs remain pending',
                arms=primary['arms'], paired=primary['paired'], secondary_anls=secondary,
                literal_strict_exact=literal, all_untruncated_sensitivity=sensitivity,
                records=detail, scoring=scoring,
                data_manifest=str(manifest.resolve()), data_manifest_sha256=sha256(manifest),
                bootstrap=dict(seed=SEED, draws=DRAWS, unit='original source parent document',
                               method='paired percentile document bootstrap; same document draws for all arms/contrasts; document macro mean'),
                timing_note='Correctness cannot establish efficiency; report independently measured timing artifacts',
                limitations='Exploratory DocVQA TRAIN numeric pilot. Small paired sample; cannot establish benchmark generalization, equivalence, or rejection of a whole method family.')


def self_check():
    anls, _ = scorers()
    assert anls('1997', ['1996']) == .75 and not score_exact('1997', ['1996'])
    assert anls('ab', ['ac']) == .5 and anls('abcd', ['axyz']) == 0.
    assert anls('wrong', ['wrong', 'correct']) == 1.
    assert score_exact(' $ 1,200.00 ', ['1200'])
    assert score_exact('0012', ['12']) and not score_exact('12', ['12%'])
    assert not score_exact('−12', ['12']) and not score_exact('value: 12', ['12'])
    arms = ['baseline', 'calibrated']
    summary = summarize({'baseline': [0, 1], 'calibrated': [1, 0]},
                        [{'parent_id': 'a'}, {'parent_id': 'b'}], arms, [('baseline', 'calibrated')])
    pair = summary['paired']['calibrated_minus_baseline']
    assert pair['rescue'] == pair['breakage'] == 1 and pair['delta_accuracy'] == 0.
    clustered = summarize({'baseline': [0, 0, 1], 'calibrated': [1, 1, 1]},
                          [{'parent_id': 'a'}, {'parent_id': 'a'}, {'parent_id': 'b'}], arms,
                          [('baseline', 'calibrated')])
    assert clustered['documents'] == 2
    assert clustered['paired']['calibrated_minus_baseline']['delta_accuracy'] == .5
    assert clustered['paired']['calibrated_minus_baseline']['micro_delta_mean'] == 2 / 3
    return dict(complete=True, official_anls_and_numeric_exact_checks='pass',
                paired_rescue_break_and_source_document_bootstrap='pass')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=QWEN / 'outputs/attention_calibration/predictions.json')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--out', type=Path, default=QWEN / 'outputs/attention_calibration/analysis.json')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return
    payload = json.loads(args.input.read_text())
    result = analyze(payload, args.manifest)
    result['predictions_path'] = str(args.input.resolve())
    result['predictions_sha256'] = sha256(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + '.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(args.out)
    print(json.dumps(dict(n=result['n'], arms=result['arms'], paired=result['paired']), indent=2))


if __name__ == '__main__':
    main()
