"""CPU-only paired QA and four-stream teacher audit for the multistream oracle.

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
import itertools
import json
from pathlib import Path
import re
import tempfile

import numpy as np

QWEN = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = QWEN / 'outputs/multistream_oracle/data/manifest.json'
SEED = 20261109
DRAWS = 10000
_PRIMARY_NAMESPACE = None
SCORE_NAMES = ('mag_v', 'mag_joint', 'signed_v', 'signed_joint')
FAMILIES = ('mag', 'signed')
N, K = 1024, 256


def required_contrasts(arms):
    """The final protocol's 27 before/after comparisons, when all 15 arms run."""
    result = []

    def add(before, after):
        if before in arms and after in arms:
            result.append([before, after])

    for family in FAMILIES:
        for stage in ('on', 'off'):
            add(f'{family}_v_{stage}', f'{family}_joint_{stage}')
    for before, after in [('full_no_ds', 'full'), ('b2_no_ds', 'b2'), ('random_off', 'random_on')]:
        add(before, after)
    for after in ('mag_v_on', 'mag_joint_on', 'signed_v_on', 'signed_joint_on'):
        for before in ('b2', 'rres512', 'random_on'):
            add(before, after)
    for after in ('mag_v_off', 'mag_joint_off', 'signed_v_off', 'signed_joint_off'):
        for before in ('b2_no_ds', 'random_off'):
            add(before, after)
    return result


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


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return dict(n=0, mean=None, median=None, quantiles=None)
    return dict(n=int(array.size), mean=float(array.mean()), median=float(np.median(array)),
                min=float(array.min()), max=float(array.max()),
                quantiles={str(q): float(np.quantile(array, q)) for q in (0, .1, .25, .5, .75, .9, .99, 1)})


def score_statistics(values):
    array = np.asarray(values, dtype=np.float64)
    square_sum = float(np.square(array).sum())
    return dict(**distribution(array), finite=int(np.isfinite(array).sum()),
                zero=int((array == 0).sum()), negative=int((array < 0).sum()),
                positive=int((array > 0).sum()), total=float(array.sum()),
                absolute_total=float(np.abs(array).sum()),
                positive_total=float(array[array > 0].sum()), negative_total=float(array[array < 0].sum()),
                participation=None if square_sum == 0 else float(np.abs(array).sum() ** 2 / square_sum))


def stable_topk(scores):
    array = np.asarray(scores)
    return np.lexsort((np.arange(len(array)), -array))[:K]


def validate_teacher(row, config):
    """Audit saved float32 maps without recomputing a model or using gradients."""
    teacher = row.get('teacher')
    if not isinstance(teacher, dict) or teacher.get('error'):
        raise ValueError('Missing/failed teacher remains pending')
    loss = float(teacher['loss'])
    if not np.isfinite(loss) or loss < 0:
        raise ValueError('Teacher answer NLL must be finite and nonnegative')
    if teacher['target_text'] != row['answers'][0]:
        raise ValueError('Teacher target must be the unmodified first original answer alias')
    target_ids = teacher['target_ids']
    if not target_ids or not all(isinstance(value, int) and value >= 0 for value in target_ids):
        raise ValueError('Teacher needs nonempty integer answer token IDs')
    if set(target_ids) & set(config.get('eos_ids', [151645])):
        raise ValueError('Teacher target unexpectedly contains an EOS token')
    stream = {kind: np.asarray(teacher['stream_scores'][kind], dtype=np.float32)
              for kind in ('mag', 'signed')}
    for kind, values in stream.items():
        if values.shape != (4, N) or not np.isfinite(values).all():
            raise ValueError(f'Invalid {kind} four-stream score array')
    if np.any(stream['mag'] < 0):
        raise ValueError('Coordinate-wise absolute magnitude must remain nonnegative')
    eps = np.finfo(np.float32).eps
    if np.any(np.abs(stream['signed']) > stream['mag'] + 16 * eps * np.maximum(stream['mag'], 1e-30)):
        raise ValueError('Absolute signed sum cannot exceed its coordinate-wise absolute magnitude')
    scores = {name: np.asarray(teacher['scores'][name], dtype=np.float32) for name in SCORE_NAMES}
    bounds, maximum_errors = {}, {}
    for name, values in scores.items():
        if values.shape != (N,) or not np.isfinite(values).all():
            raise ValueError(f'Invalid {name} score map')
        family = 'mag' if name.startswith('mag_') else 'signed'
        if name.endswith('_v'):
            reference = stream[family][0]
            if not np.array_equal(values, reference):
                raise ValueError(f'{name} is not exactly stream zero from the shared teacher')
            maximum_errors[name], bounds[name] = 0., 0.
        else:
            reference = stream[family].astype(np.float64).sum(axis=0)
            # Only float32 four-term summation rounding is tolerated. The bound
            # scales with the sum of absolute terms, including cancellations.
            bound = 4 * eps * np.abs(stream[family].astype(np.float64)).sum(axis=0) + 1e-30
            error = np.abs(values.astype(np.float64) - reference)
            if np.any(error > bound):
                raise ValueError(f'{name} is not the unnormalized signed/absolute four-stream sum')
            maximum_errors[name], bounds[name] = float(error.max()), float(bound.max())
    keep_raw = dict(teacher['keep'])
    for name, indices in row.get('keep', {}).items():
        if name in keep_raw and keep_raw[name] != indices:
            raise ValueError('Conflicting saved keep sets')
        keep_raw[name] = indices
    required_keep = set(SCORE_NAMES)
    if 'b2' in config.get('planned_arms', []):
        required_keep.add('b2')
    if 'random_on' in config.get('planned_arms', []):
        required_keep.add('random')
    if not required_keep.issubset(keep_raw):
        raise ValueError('Missing teacher/B2/random keep-set diagnostic')
    sets = {}
    for name, indices in keep_raw.items():
        if not isinstance(indices, list) or len(indices) != K or not all(isinstance(v, int) for v in indices):
            raise ValueError(f'Invalid K256 index list: {name}')
        if indices != sorted(set(indices)) or indices[0] < 0 or indices[-1] >= N:
            raise ValueError(f'{name} keep indices must be unique/raster-ordered/in-range')
        sets[name] = set(indices)
    for name in SCORE_NAMES:
        expected = stable_topk(scores[name])
        if keep_raw[name] != sorted(expected.tolist()):
            raise ValueError(f'{name} does not match stable descending score Top-K')
        if 'rank_order' in teacher and name in teacher['rank_order']:
            if teacher['rank_order'][name] != expected.tolist():
                raise ValueError(f'{name} score-order trace is not stable descending Top-K')
        for stage in ('on', 'off'):
            prediction = row.get('predictions', {}).get(name + '_' + stage, {})
            recorded_keep = prediction.get('keep_indices', prediction.get('keep_idx'))
            if recorded_keep is not None and recorded_keep != keep_raw[name]:
                raise ValueError('On/off prediction changed the common spatial keep-set')
    overlaps = {}
    for left, right in itertools.combinations(sorted(sets), 2):
        common = len(sets[left] & sets[right])
        overlaps[right + '_vs_' + left] = dict(left=left, right=right, common_tokens=common,
                                              union_tokens=2 * K - common,
                                              jaccard=common / (2 * K - common))
    stream_summary = []
    magnitude_total = float(stream['mag'].astype(np.float64).sum())
    for index, label in enumerate(('V', 'D1', 'D2', 'D3')):
        mag_stats = score_statistics(stream['mag'][index])
        stream_summary.append(dict(stream=label, magnitude=mag_stats,
                                   signed=score_statistics(stream['signed'][index]),
                                   relative_magnitude_mass=None if magnitude_total == 0 else mag_stats['total'] / magnitude_total))
    return dict(item_id=row['item_id'], parent_id=row['parent_id'], loss=loss,
                target_text=teacher['target_text'], target_ids=target_ids,
                target_tokens=len(target_ids), score_validation_pass=True,
                maximum_float32_sum_errors=maximum_errors, maximum_sum_error_bounds=bounds,
                scores={name: score_statistics(values) for name, values in scores.items()},
                streams=stream_summary, overlaps=overlaps,
                keep={name: keep_raw[name] for name in sorted(keep_raw)})


def interactions(vectors, rows):
    if not rows:
        return dict(n=0, documents=0, families={}, note='Empty auxiliary subgroup')
    parents = sorted({r['parent_id'] for r in rows})
    groups = [[i for i, row in enumerate(rows) if row['parent_id'] == parent] for parent in parents]
    draws = np.random.default_rng(SEED).integers(0, len(parents), size=(DRAWS, len(parents)))

    def estimate(values):
        array = np.asarray(values, dtype=float)
        document_values = np.asarray([array[group].mean() for group in groups])
        return dict(mean=float(document_values.mean()), micro_mean=float(array.mean()),
                    bootstrap_95ci=np.quantile(document_values[draws].mean(axis=1), [.025, .975]).tolist(),
                    positive_documents=int((document_values > 0).sum()),
                    negative_documents=int((document_values < 0).sum()),
                    zero_documents=int((document_values == 0).sum()))

    result = dict(n=len(rows), documents=len(parents), families={})
    for family in FAMILIES:
        delta_on = np.asarray(vectors[family + '_joint_on'], float) - np.asarray(vectors[family + '_v_on'], float)
        delta_off = np.asarray(vectors[family + '_joint_off'], float) - np.asarray(vectors[family + '_v_off'], float)
        joint_ds = np.asarray(vectors[family + '_joint_on'], float) - np.asarray(vectors[family + '_joint_off'], float)
        v_ds = np.asarray(vectors[family + '_v_on'], float) - np.asarray(vectors[family + '_v_off'], float)
        difference = delta_on - delta_off
        if not np.allclose(difference, joint_ds - v_ds, atol=1e-14, rtol=0):
            raise ValueError('Interaction algebra failed')
        result['families'][family] = dict(
            formula='(joint_on - v_on) - (joint_off - v_off)',
            delta_on=estimate(delta_on), delta_off=estimate(delta_off),
            joint_ds_on_minus_off=estimate(joint_ds), v_ds_on_minus_off=estimate(v_ds),
            interaction=estimate(difference), per_question_interactions=difference.tolist())
    return result


def teacher_summary(diagnostics, rows):
    parents = sorted({r['parent_id'] for r in rows})
    by_parent = {d['parent_id']: d for d in diagnostics}
    draws = np.random.default_rng(SEED).integers(0, len(parents), size=(DRAWS, len(parents)))
    overlap_keys = set(diagnostics[0]['overlaps'])
    if any(set(d['overlaps']) != overlap_keys for d in diagnostics):
        raise ValueError('Saved keep-set diagnostic names differ across primary documents')
    overlap_summary = {}
    for key in sorted(overlap_keys):
        entry = {}
        for metric in ('common_tokens', 'jaccard'):
            values = np.asarray([by_parent[parent]['overlaps'][key][metric] for parent in parents], float)
            entry[metric] = dict(**distribution(values),
                                 bootstrap_95ci=np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist())
        overlap_summary[key] = entry
    streams = {}
    for index, label in enumerate(('V', 'D1', 'D2', 'D3')):
        streams[label] = dict(
            magnitude_total=distribution([d['streams'][index]['magnitude']['total'] for d in diagnostics]),
            signed_total=distribution([d['streams'][index]['signed']['total'] for d in diagnostics]),
            absolute_signed_total=distribution([d['streams'][index]['signed']['absolute_total'] for d in diagnostics]),
            relative_magnitude_mass=distribution([d['streams'][index]['relative_magnitude_mass'] for d in diagnostics
                                                  if d['streams'][index]['relative_magnitude_mass'] is not None]))
    return dict(all_four_score_maps_verified=True, score_count_per_document=4,
                source_streams=4, input_tokens=N, keep_tokens=K,
                stable_topk_and_signed_scores_preserved=True,
                loss=distribution([d['loss'] for d in diagnostics]), streams=streams, overlaps=overlap_summary,
                per_document=diagnostics,
                note='Gradient scores, participation and overlap are diagnostics, not evidence of deployable utility or unique causal information.')


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
    baseline = config.get('baseline_arm', 'b2')
    if baseline not in arms:
        raise ValueError('config.baseline_arm must name an evaluated arm')
    if 'contrasts' not in config:
        raise ValueError('Runner must explicitly freeze the complete preregistered contrasts')
    contrasts = config['contrasts']
    if any(len(pair) != 2 or pair[0] not in arms or pair[1] not in arms or pair[0] == pair[1]
           for pair in contrasts):
        raise ValueError('Invalid frozen pairwise contrast')
    keys = [after + '_minus_' + before for before, after in contrasts]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate pairwise contrasts')
    if not set(map(tuple, required_contrasts(arms))).issubset(set(map(tuple, contrasts))):
        raise ValueError('Missing preregistered explicit pairwise contrast')
    for family in FAMILIES:
        if not {f'{family}_{kind}_{stage}' for kind in ('v', 'joint') for stage in ('on', 'off')}.issubset(arms):
            raise ValueError('Both score families require DS-on/off joint/V arms')
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
    detail, diagnostics = [], []
    for row in rows:
        item = by_id[row['item_id']]
        for field in ('parent_id', 'question', 'answers'):
            if row[field] != item[field]:
                raise ValueError(f'Frozen annotation changed: {row["item_id"]} {field}')
        if 'image_sha256' in row and row['image_sha256'] != item['image_sha256']:
            raise ValueError('Frozen source image hash differs in prediction record')
        answers = row['answers']
        if not answers or not all(isinstance(a, str) for a in answers):
            raise ValueError('Missing original answer aliases')
        diagnostics.append(validate_teacher(row, config))
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
        sensitivity[metric + '_interactions'] = interactions(
            {arm: [scores[metric][arm][i] for i in untruncated] for arm in arms},
            [rows[i] for i in untruncated])
    interaction_results = {metric: interactions(scores[metric], rows) for metric in ('exact', 'anls', 'literal')}
    strata = {}
    full_arm = config.get('full_arm', 'full')
    if full_arm not in arms:
        raise ValueError('Need the declared full native DS-on reference for auxiliary strata')
    for label, indices in [('full_correct', [i for i in range(len(rows)) if scores['exact'][full_arm][i]]),
                           ('full_wrong', [i for i in range(len(rows)) if not scores['exact'][full_arm][i]])]:
        subgroup = [rows[i] for i in indices]
        stratum = dict(auxiliary_only=True, n=len(indices), item_ids=[rows[i]['item_id'] for i in indices],
                       teacher_loss=distribution([diagnostics[i]['loss'] for i in indices]))
        for metric, binary in [('exact', True), ('anls', False), ('literal', True)]:
            vectors = {arm: [scores[metric][arm][i] for i in indices] for arm in arms}
            stratum[metric] = summarize(vectors, subgroup, arms, contrasts, binary)
            stratum[metric + '_interactions'] = interactions(vectors, subgroup)
        strata[label] = stratum
    return dict(schema_version=1, complete=True, n=len(rows), documents=primary['documents'],
                denominator='Every preselected primary source document/QA after all runs complete; no correctness or truncation filtering; missing/operationally failed runs remain pending',
                arms=primary['arms'], paired=primary['paired'], secondary_anls=secondary,
                literal_strict_exact=literal, all_untruncated_sensitivity=sensitivity,
                ds_interactions=interaction_results, auxiliary_full_correctness_strata=strata,
                teacher_diagnostics=teacher_summary(diagnostics, rows),
                records=detail, scoring=scoring,
                data_manifest=str(manifest.resolve()), data_manifest_sha256=sha256(manifest),
                bootstrap=dict(seed=SEED, draws=DRAWS, unit='original source parent document',
                               method='paired percentile document bootstrap; same document draws for all arms/contrasts; document macro mean'),
                timing_note='Gold-answer backward teachers are offline diagnostics; generation timing without the teacher cannot establish deployable TTFT',
                limitations='Label-aware, answer-conditioned DocVQA TRAIN gradient diagnostic. Shared DS-on teacher defines all four scores; DS-off changes execution distribution. No trained student or deployable accuracy claim; first-order full-sequence scores are not exact hard-deletion utilities or a rigorous upper bound.')


def synthetic_teacher(target='12'):
    base = np.arange(1, N + 1, dtype=np.float32)
    magnitude = np.stack([base, 2 * base[::-1], np.ones(N, np.float32), np.ones(N, np.float32)])
    signed = np.stack([-.5 * base, .1 * base[::-1], np.zeros(N, np.float32), -np.ones(N, np.float32)])
    maps = dict(mag_v=magnitude[0], mag_joint=magnitude.sum(axis=0),
                signed_v=signed[0], signed_joint=signed.sum(axis=0))
    return dict(loss=.5, target_text=target, target_ids=[16, 17],
                stream_scores=dict(mag=magnitude.tolist(), signed=signed.tolist()),
                scores={name: values.tolist() for name, values in maps.items()},
                keep={**{name: sorted(stable_topk(values).tolist()) for name, values in maps.items()},
                      'b2': list(range(K)), 'random': list(range(N - K, N))})


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
    zero = score_statistics([0., 0.])
    assert zero['participation'] is None
    assert stable_topk(-np.ones(N, np.float32)).tolist() == list(range(K))
    teacher = synthetic_teacher()
    row = dict(item_id='synthetic_0', parent_id='synthetic_parent_0', answers=['12'], teacher=teacher)
    diagnostic = validate_teacher(row, {})
    assert diagnostic['score_validation_pass'] and len(diagnostic['overlaps']) == 15
    assert abs(sum(s['relative_magnitude_mass'] for s in diagnostic['streams']) - 1) < 1e-12
    import copy
    for label, mutate in [('corrupt_joint', lambda t: t['scores']['mag_joint'].__setitem__(0, -1)),
                          ('corrupt_keep', lambda t: t['keep']['mag_v'].__setitem__(0, 0)),
                          ('wrong_target', lambda t: t.__setitem__('target_text', '13')),
                          ('missing_gradient_stream', lambda t: t['stream_scores']['mag'].pop())]:
        bad = copy.deepcopy(row)
        mutate(bad['teacher'])
        try:
            validate_teacher(bad, {})
        except ValueError:
            pass
        else:
            raise AssertionError(label)
    vectors = {}
    for family in FAMILIES:
        vectors.update({family + '_v_on': [0, 1], family + '_joint_on': [1, 0],
                        family + '_v_off': [0, 1], family + '_joint_off': [1, 0]})
    inter = interactions(vectors, [{'parent_id': 'a'}, {'parent_id': 'b'}])
    for family in FAMILIES:
        assert inter['families'][family]['interaction']['bootstrap_95ci'] == [0., 0.]
        assert inter['families'][family]['delta_on']['bootstrap_95ci'] == [-1., 1.]
    arms = ['full', 'full_no_ds', 'rres512', 'b2', 'b2_no_ds',
            'mag_v_on', 'mag_v_off', 'mag_joint_on', 'mag_joint_off',
            'signed_v_on', 'signed_v_off', 'signed_joint_on', 'signed_joint_off',
            'random_on', 'random_off']
    contrasts = required_contrasts(arms)
    assert len(contrasts) == len(set(map(tuple, contrasts))) == 27
    items, predictions = [], []
    for index in range(32):
        item = dict(item_id=f'synthetic_{index}', parent_id=f'parent_{index}', question='Synthetic value?',
                    answers=['12'], image_sha256='synthetic-only-no-real-image')
        items.append(item)
        outputs = {arm: dict(text='12', truncated=False, eos=True, token_ids=[16, 17]) for arm in arms}
        if index == 0:
            outputs['mag_joint_on']['text'] = '13'
            outputs['mag_v_on']['truncated'] = True
        if index == 1:
            outputs['full']['text'] = '13'
        predictions.append(dict(**item, teacher=synthetic_teacher(), predictions=outputs))
    with tempfile.TemporaryDirectory(prefix='multistream_analysis_selfcheck_') as directory:
        path = Path(directory) / 'manifest.json'
        path.write_text(json.dumps(dict(complete=True, role='primary', items=items)))
        payload = dict(complete=True,
                       config=dict(manifest_sha256=sha256(path), planned_arms=arms,
                                   baseline_arm='b2', contrasts=contrasts, eos_ids=[151645]),
                       records=predictions)
        result = analyze(payload, path)
        assert result['n'] == 32 and len(result['paired']) == 27
        assert result['paired']['mag_joint_on_minus_mag_v_on']['breakage'] == 1
        assert result['ds_interactions']['exact']['families']['mag']['interaction']['mean'] == -1 / 32
        assert result['all_untruncated_sensitivity']['n'] == 31
        assert result['auxiliary_full_correctness_strata']['full_wrong']['n'] == 1
        assert result['teacher_diagnostics']['all_four_score_maps_verified']
        for label, mutate in [('missing_question', lambda x: x['records'].pop()),
                              ('duplicate_question', lambda x: x['records'].__setitem__(0, x['records'][1])),
                              ('missing_contrast', lambda x: x['config']['contrasts'].pop()),
                              ('operational_error', lambda x: x['records'][0]['predictions']['b2'].__setitem__('error', 'synthetic'))]:
            bad = copy.deepcopy(payload)
            mutate(bad)
            try:
                analyze(bad, path)
            except ValueError:
                pass
            else:
                raise AssertionError(label)
    return dict(complete=True, synthetic_only=True, GPU_used=False,
                official_anls_and_exact_CF_functions='pass',
                paired_rescue_break_and_document_bootstrap='pass',
                joint_DS_interaction_covariance_and_CI='pass',
                four_stream_score_sums_signs_topk_and_overlap='pass',
                synthetic_32_documents_15_arms_27_contrasts='pass',
                truncation_and_full_wrong_auxiliary_strata='pass',
                reject_missing_duplicate_contrast_operational_error_and_corrupt_scores='pass',
                analysis_script_sha256=sha256(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=QWEN / 'outputs/multistream_oracle/predictions.json')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--out', type=Path, default=QWEN / 'outputs/multistream_oracle/analysis.json')
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
