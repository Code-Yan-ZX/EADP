"""CPU-only, source-backed threshold audit. Writes only beside this file."""
from collections import Counter
from decimal import Decimal, getcontext
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json

getcontext().prec = 32
ROOT = Path('/media/disk2/YZX/research/EADP_amp')
OUT = ROOT / 'Qwen_vl/outputs/audit_followup_20261008'
HERE = Path(__file__).parent
COVERAGE = OUT / 'additional_reproduction_audit_20261009/coverage_inventory.json'
PAPER = OUT / 'paper_reference.json'
SUMMARY = OUT / 'rerun_batch/repaired_results_summary.json'
THRESHOLD = Decimal('0.2')
registry = {}
counts_cache = {}


def read(path):
    return json.loads(Path(path).read_text())


def read_snapshot(path):
    path = Path(path).absolute()
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    stable = (before.st_ino, before.st_mtime_ns, before.st_size) == (after.st_ino, after.st_mtime_ns, after.st_size)
    if not stable:
        raise ValueError('Input changed during snapshot: ' + str(path))
    registry[str(path)] = dict(path=str(path), exists=True, bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(), stable_during_read=True,
        modified_utc=dt.datetime.fromtimestamp(after.st_mtime, dt.timezone.utc).isoformat(),
        timestamp_meaning='Filesystem mtime for these exact captured bytes; not reconstructed generation time.')
    return json.loads(payload)


def source(path, role='evidence'):
    path = Path(path).absolute()
    key = str(path)
    if not path.is_file():
        return dict(path=key, exists=False, role=role)
    if key not in registry:
        before = path.stat()
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        after = path.stat()
        registry[key] = dict(path=key, exists=True, bytes=after.st_size,
            sha256=h.hexdigest(), modified_utc=dt.datetime.fromtimestamp(after.st_mtime, dt.timezone.utc).isoformat(),
            stable_during_read=(before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size),
            timestamp_meaning='Filesystem mtime at audit; not a reconstructed generation timestamp.')
    return dict(registry[key], role=role)


def raw_counts(path):
    path = Path(path)
    if str(path) in counts_cache:
        return counts_cache[str(path)]
    invalid = 0
    if path.suffix == '.jsonl':
        rows = []
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    invalid += 1
        result = dict(n=len(rows), invalid_lines=invalid,
            unique_composite_keys=len({(str(r.get('question_id', r.get('id'))), r.get('prompt', r.get('text'))) for r in rows}),
            FAILED_sentinels=sum(str(r.get('text', '')).strip().startswith('FAILED') for r in rows))
    elif path.suffix == '.xlsx':
        import openpyxl
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        iterator = book.active.iter_rows(values_only=True)
        next(iterator)
        result = dict(n=sum(1 for _ in iterator), invalid_lines=0)
        book.close()
    else:
        records = read(path)['records']
        result = dict(n=len(records), unique_record_keys=len(records), invalid_lines=0)
    counts_cache[str(path)] = result
    return result


def decimal(value):
    return Decimal(str(value))


def number(value):
    return float(value) if value is not None else None


def classify(delta):
    if delta is None:
        return 'missing_or_unscored'
    if delta > 0:
        return 'exceeds_paper'
    if delta >= -THRESHOLD:
        return 'numerically_meets_user_threshold'
    return 'below_paper_by_more_than_0.2'


def scaling(task):
    if task == 'MME':
        return Decimal(20), 'paper Avg equivalent points (raw MME /20); not accuracy percentage points'
    if task == 'OCRBench':
        return Decimal(10), 'OCRBench 0–100 equivalent points (raw score /10)'
    return Decimal(1), 'official task metric percentage points (accuracy/F1/ANLS/fAcc as explicitly named)'


def metric_name(model, task):
    if task == 'MME':
        return 'MME perception+cognition total raw points' if model == 'qwen' else 'MME perception raw points'
    return dict(TextVQA='official TextVQA accuracy (%)', ChartQA='official relaxed accuracy (%)',
        AI2D='official exact-match accuracy (%)', OCRBench='OCRBench raw score (maximum1000)',
        HallusionBench='HallusionBench fAcc (%)', SQA_IMG='ScienceQA IMG accuracy (%)',
        GQA='official GQA accuracy (%)', POPE='POPE three-category macroF1 (%)',
        VizWiz='VizWiz validation official leave-one-out VQA accuracy (%)', MMB_EN='MMBench circular accuracy (%)',
        MMB_CN='MMBench circular accuracy (%)', DocVQA='official ANLS (%)', InfoVQA='official ANLS (%)',
        VQAv2='official VQAv2 test submission accuracy (%)', MMVet='official GPT-evaluated MMVet score (%)')[task]


def verified_existing(evidence):
    prediction = source(evidence['raw']['prediction']['path'], 'full_raw_prediction')
    score = source(evidence['official_score_artifact']['path'], 'official_score')
    for current, previous in [(prediction, evidence['raw']['prediction']), (score, evidence['official_score_artifact'])]:
        if not current.get('stable_during_read') or current.get('sha256') != previous['sha256']:
            raise ValueError('Frozen complete evidence changed: ' + current['path'])
    counts = raw_counts(prediction['path'])
    if counts['n'] != evidence['expected_n'] or counts['invalid_lines'] or counts.get('FAILED_sentinels', 0):
        raise ValueError('Full raw evidence no longer complete: ' + prediction['path'])
    result = dict(evidence)
    result['raw'] = dict(evidence['raw'], **counts, prediction=prediction)
    result['official_score_artifact'] = score
    result['source_verification'] = 'Current full raw/score SHA equals frozen coverage evidence; full count independently reread.'
    result['score_value_derivation'] = 'Existing verified official metric, source preserved.'
    return result


def exact_value(task, evidence):
    if task == 'POPE':
        data = read(evidence['official_score_artifact']['path'])
        categories = data.get('categories', data)
        values = []
        confusion = {}
        for name in ['random', 'popular', 'adversarial']:
            c = categories[name]
            tp, fp, fn = [Decimal(c[key]) for key in ['TP', 'FP', 'FN']]
            values.append(Decimal(200) * tp / (2 * tp + fp + fn))
            confusion[name] = {key: c[key] for key in ['TP', 'FP', 'TN', 'FN', 'n_pred']}
        evidence['score_value_derivation'] = 'Exact unrounded mean of the three official confusion-matrix F1 values; no per-answer judgment changed.'
        evidence['official_confusion_matrices'] = confusion
        return sum(values) / 3
    return decimal(evidence['score'])


def protocol_assessment(model, task, budget, evidence):
    flags = []
    status = 'bounded_metric_budget_comparison_paper_runtime_argv_unpublished'
    if model == 'qwen':
        flags.append('Pruned EADP uses the official legacy DeepStack-off/1d engine; native EADP scores are excluded.')
        if task == 'HallusionBench' and budget == 'FULL' and evidence:
            status = 'protocol_incomparable_or_unresolved'
            flags.append('Local FULL_HALLB is a deliberate legacy-engine keep-all control (DeepStack off/1d), not a verified reproduction of the paper FULL reference. Native FULL_HALLB_NATIVE remains separate and is not substituted.')
        elif task == 'HallusionBench' and budget == 256:
            status = 'generation_source_identity_unclosed'
            flags.append('Recorded bank SHA differs from current compressed bank SHA; complete prediction/score remains valid as a measurement, generation-time bank identity remains unclosed.')
        elif task in ('MMB_EN', 'MMB_CN'):
            status = 'paper_dataset_variant_unverified'
            flags.append('Qwen local input is MMBench V11, 4876 rows/1292 circular groups. Table4 exact input variant is not proven by a score; the dated LLaVA release-script difference is not automatically a Qwen configuration error.')
    elif task == 'SQA_IMG':
        status = 'paper_question_format_unavailable'
        flags.append('Local repaired input uses shipped CQM-A, 2017 image questions; authors actual CQM-I file was not published. Numerical work acceptance does not establish identical question formatting.')
    elif task in ('MMB_EN', 'MMB_CN'):
        status = 'protocol_incomparable_or_unresolved'
        flags.append('Current V11 differs from official released LLaVA dated MMBench inputs; actual paper dataset file remains unverified.')
        if task == 'MMB_CN':
            flags.append('Available local AZ uses default lang=en rather than released --lang cn. No local EADP measurement is manufactured from that AZ output.')
    elif task == 'VizWiz':
        status = 'paper_scorer_execution_unverified'
        flags.append('Use starred validation reference, not historical challenge test score. Released min(matches/3) scorer differs from benchmark official leave-one-out scorer; paper actual execution is unknown.')
    elif task == 'MME':
        flags.append('LLaVA uses perception only; Qwen uses perception+cognition total. Raw points and /20 paper-Avg equivalent units are both retained.')
    if model in ('v15', 'next') and budget != 'FULL':
        flags.append('Frozen local alpha=.5/beta=2; released task defaults and unpublished paper runtime argv do not establish paper hyperparameter identity. No parameter search or best-score selection.')
    if model == 'next':
        flags.append('128/64/32 CLI budgets map to paper nominal640/320/160; AnyRes importance allocations use official floor/min-one and can realize slightly smaller totals.')
    if evidence and evidence.get('stage') == 'historical_legacy_entry':
        flags.append('Historical complete result remains exposed to stream dependency; registered wait-only repair is pending unless separately validated.')
    return status, flags


coverage = read_snapshot(COVERAGE)
paper = read_snapshot(PAPER)
summary = read_snapshot(SUMMARY)
lookup = {(r['model'], r['pipeline'], r['benchmark'], str(r['budget'])): r for r in coverage['rows']}
summary_lookup = {(r['model'], 'SQA_IMG' if r['task'] == 'sqa' else 'TextVQA',
    'FULL' if r['method'] == 'FULL' else str(r['budget_parameter'])): r for r in summary['rows'] if r['method'] in ('EADP', 'FULL')}

# A physically complete newly finished control is not promoted across an unresolved
# controller validation gate. Preserve its exact existing score as auxiliary evidence.
stab = OUT / 'next_gap_diagnosis_20261008/stability'
controller_state = read_snapshot(stab/'next_text_full_streamwait.controller.state.json')
overlay_proof_path = HERE/'json_count_distribution_overlay_cpu_validation.json'
overlay_proof = read_snapshot(overlay_proof_path) if overlay_proof_path.is_file() else None
recovery = controller_state.get('recovery', {})
overlay_recovered = bool(overlay_proof and recovery and overlay_proof.get('success') is True
    and overlay_proof.get('CPU_only') is True and overlay_proof.get('production_modified') is False
    and overlay_proof.get('GPU_started') is False and all(overlay_proof.get('gates', {}).values())
    and overlay_proof.get('overlay_sha256') == source(recovery['overlay'], 'JSON_type_only_overlay')['sha256']
    and recovery.get('overlay_sha256') == overlay_proof.get('overlay_sha256')
    and recovery.get('changed_values') is False and recovery.get('reran_complete_beta2') is False
    and overlay_proof.get('frozen_worker_sha256') == source(stab/'next_text_full_streamwait.py', 'unchanged_original_control_worker')['sha256'])
control_candidates = {}
for arm in ['EADP_beta2', 'EADP_beta1']:
    job = controller_state.get('jobs', {}).get(arm, {})
    output_paths = job.get('output_paths', {})
    if not output_paths or not Path(output_paths['score']).is_file() or not Path(output_paths['finished']).is_file():
        continue
    stored = read(output_paths['score'])
    marker = read(output_paths['finished'])
    individual = read(output_paths['state'])
    protocol = read(output_paths['control_protocol'])
    pred = source(output_paths['prediction'], 'new_full_control_prediction')
    official = source(output_paths['score'], 'new_full_control_official_score')
    counts = raw_counts(pred['path'])
    physical = (counts['n'] == 5000 and counts['invalid_lines'] == 0 and not counts.get('FAILED_sentinels', 0)
        and marker.get('success') is True and marker.get('complete') is True and marker.get('n') == 5000
        and marker.get('generation_exit_code') == 0 and individual.get('complete') is True
        and individual.get('generation_exit_code') == 0 and marker['prediction_sha256'] == pred['sha256']
        and marker['score_sha256'] == official['sha256'] and stored['prediction_sha256'] == pred['sha256'])
    if not physical:
        raise ValueError('New full control evidence incomplete/inconsistent: ' + arm)
    validation = job.get('validation', {})
    independently_validated = (job.get('status') == 'complete' and validation.get('success') is True
        and validation.get('n') == 5000 and validation.get('score_sha256') == official['sha256']
        and validation.get('prediction_sha256') == pred['sha256'])
    if recovery and independently_validated and not overlay_recovered:
        raise ValueError('Recovered full control lacks successful pinned CPU overlay proof')
    control_candidates[arm] = dict(stage='completed_wait_only_control' if independently_validated else 'complete_control_validation_pending',
        arm=arm, beta=protocol['parameters']['beta'], physical_full_complete=True,
        validation_status='controller_independent_gate_passed' if independently_validated else 'validation_pending',
        independent_validation=validation,
        JSON_gate_recovery=dict(overlay_CPU_report=source(overlay_proof_path, 'exact_CPU_overlay_validation'),
            recovered_without_regeneration=True, changed_values=False, only_raw_mismatch='actual_count_distribution integer vs JSON string keys') if overlay_recovered else None,
        controller_job_status=job.get('status'), controller_error=job.get('error'),
        raw=dict(counts, prediction=pred), official_score_artifact=official, score=stored['accuracy_percent'], expected_n=5000,
        finished_utc=marker['finished_utc'], score_value_derivation='Existing official exact5000 score; no partial accuracy.',
        finished_artifact=source(output_paths['finished'], 'finished_marker'),
        control_protocol=source(output_paths['control_protocol'], 'control_protocol'),
        native_protocol=source(output_paths['native_protocol'], 'native_protocol'),
        runtime=source(output_paths['runtime'], 'runtime_trace'), parameters=protocol['parameters'],
        source_and_input_sha256=protocol['source_and_input_sha256'])

rows = []
for table in paper['tables']:
    model = {1: 'v15', 2: 'next', 4: 'qwen'}[table['table']]
    pipeline = 'official_legacy' if model == 'qwen' else 'llava'
    for paper_row in table['rows']:
        budget = 'FULL' if paper_row['method'] == 'FULL' else paper_row['budget_per_crop_equivalent']
        for task in table['columns']:
            if task == 'Avg':
                continue
            old = lookup[(model, pipeline, task, str(budget))]
            options = old['completed_local']
            # Predeclared protocol recency, never max(score) and never another seed.
            selected = verified_existing(options[0]) if options else None
            selection = 'Latest complete repaired input/official entry recorded in frozen inventory; no score-based choice.'
            auxiliary = []
            if model == 'next' and task == 'TextVQA' and budget == 32:
                for arm, candidate in control_candidates.items():
                    auxiliary.append(candidate)
                    if arm == 'EADP_beta2' and candidate['validation_status'] == 'controller_independent_gate_passed':
                        selected = candidate
                        selection = 'Predefined beta2 wait-only repair, independently complete and validated; replaces as-is for current-protocol diagnosis, old score retained.'
            # Latest exported complete official score and SHA take precedence over an
            # inherited rounded transcription for repaired TextVQA/SQA.
            current = summary_lookup.get((model, task, str(budget)))
            if selected and selected['stage'] == 'repaired_input_or_official_entry' and current:
                if current['status'] != 'complete' or current['prediction_sha256'] != selected['raw']['prediction']['sha256']:
                    raise ValueError('Latest repaired summary contradicts full raw source')
                selected['score'] = current['accuracy']
                selected['latest_summary_source'] = source(SUMMARY, 'latest_complete_summary')
                selected['official_score_sources'] = current['score_sources']
            value = exact_value(task, selected) if selected else None
            paper_value = decimal(paper_row['values'][task])
            divisor, unit = scaling(task)
            delta_raw = value - paper_value if value is not None else None
            delta = delta_raw / divisor if delta_raw is not None else None
            numeric = classify(delta)
            comparability, caveats = protocol_assessment(model, task, budget, selected)
            # Keep the numeric user criterion visible even for a deliberate engine
            # counterfactual; do not present that control as paper reproduction.
            triage = ('protocol_incomparable_or_unresolved' if selected and comparability == 'protocol_incomparable_or_unresolved' else numeric)
            pending_repair = old['registered_local_reruns']
            if auxiliary and any(c['validation_status'] == 'validation_pending' for c in auxiliary):
                priority = 'P0_close_full_control_CPU_validation_gate_then_P1_deficit'
                action = 'Preserve new5000 predictions/scores and close independent CPU validation gate; do not regenerate for a JSON/validation-only mismatch. Quantify remaining deficit on the predefined valid beta2 protocol.'
            elif triage == 'protocol_incomparable_or_unresolved':
                priority = 'P1_establish_comparable_protocol'
                action = 'Keep measured score, resolve reference engine/input protocol before treating numerical difference as a reproduction deficit.'
            elif numeric == 'below_paper_by_more_than_0.2':
                priority = 'P1_investigate_verified_deficit'
                action = 'Investigate concrete input/runtime/scoring/configuration evidence; use already registered repair where applicable. No parameter sweep or best-seed selection.'
            elif numeric == 'missing_or_unscored':
                priority = 'P2_missing_local_reproduction_or_official_score'
                action = 'No numerical pass/fail is assigned. Complete local EADP/FULL and official scoring would be required; available AZ/paper/native values are not substitutes.'
            else:
                priority = 'P3_numerically_accepted_no_score_chasing'
                action = 'Accepted by user numerical criterion; do not rerun solely to match paper. Independent functional/protocol repairs, if already authorized, are separate.'
            if model == 'next' and task == 'TextVQA' and budget != 'FULL':
                unresolved = 'FULL60.370 matches/exceeds paper60.3; pruning residual remains. Canonical inputs/core sources/LLM weights checked. Public beta1 default differs from frozen beta2, but paper actual beta/q unknown; predefined beta1 control is not a sweep.'
            elif task == 'POPE':
                unresolved = 'Old release as-asked random2910/poplular3000/adversarial3000 identity/GT verified. Wait-only E32 is complete; remaining F1 gap is not explained by absent90 questions and is not a method gain.'
            elif task == 'MME':
                unresolved = '162 local GT flips were corrected by CPU rescore; residual is displayed separately. /20 is a table-Avg comparison unit, not an accuracy.'
            elif task == 'SQA_IMG':
                unresolved = 'CQM-A repaired full question format improves v15 FULL by4.6604; authors actual CQM-I absent. This is question-format evidence, not proof of exact paper input.'
            elif model == 'qwen' and task == 'HallusionBench' and budget == 256:
                unresolved = 'Resolve recorded/current bank provenance without changing score; gzip-byte difference alone does not prove selected-index changes.'
            else:
                unresolved = 'No confirmed remaining cause inferred from the deficit alone; declared metric/protocol/source caveats remain.'
            row = dict(model=model, model_name=table['model'], pipeline=pipeline, paper_table=table['table'],
                benchmark=task, method=paper_row['method'], budget_parameter=budget,
                paper_budget_total_nominal=paper_row['budget_total_nominal'],
                paper_budget_note=paper_row['budget_note'],
                expected_prediction_n=old['expected_prediction_n'],
                metric_denominator_n=old.get('metric_denominator_n'),
                local_score_raw=number(value), local_score_decimal=str(value) if value is not None else None,
                paper_score_raw=number(paper_value), metric=metric_name(model, task),
                official_source_reported_score=selected['score'] if selected else None,
                threshold_divisor=number(divisor), threshold_unit=unit,
                local_score_threshold_units=number(value / divisor) if value is not None else None,
                paper_score_threshold_units=number(paper_value / divisor),
                delta_local_minus_paper_raw=number(delta_raw), delta_local_minus_paper_threshold_units=number(delta),
                delta_decimal=str(delta) if delta is not None else None,
                numeric_status=numeric, triage_status=triage, protocol_comparability=comparability,
                priority=priority, recommended_action=action, confirmed_or_unresolved_reason=unresolved,
                local_evidence=selected, selection_rule=selection, previous_completed_measurements=options,
                new_complete_controls=auxiliary, incomplete_or_unscored_local_evidence=old['unscored_or_partial_local'],
                registered_EADP_or_FULL_repair=pending_repair,
                registered_repair_inventory_snapshot_utc=coverage['created_utc'],
                registered_repair_snapshot_note='These inherited registration/progress fields describe the frozen coverage snapshot. Current K32 canonical validation and scores are separately captured above; an old partial count does not supersede the current complete score.',
                paper_source=dict(url=paper_row['source_url'], snapshot=source(PAPER, 'paper_reference_snapshot'),
                    printed_value=paper_row['raw_cells'][task]),
                caveats=caveats,
                new_control_gap=[dict(arm=c['arm'], beta=c['beta'],
                    local_score_raw=c['score'], delta_raw=c['score'] - float(paper_value),
                    numeric_status=classify(decimal(c['score']) - paper_value), validation_status=c['validation_status']) for c in auxiliary])
            rows.append(row)

assert len(rows) == 120 and len({(r['model'], r['benchmark'], str(r['budget_parameter'])) for r in rows}) == 120
assert {r['budget_parameter'] for r in rows if r['model'] == 'qwen'} == {'FULL', 512, 256, 128}
assert all(r['pipeline'] != 'native_restored' for r in rows)
summary_counts = {}
for model in ['v15', 'next', 'qwen']:
    selected = [r for r in rows if r['model'] == model]
    summary_counts[model] = dict(n_cells=len(selected), numeric_status=dict(Counter(r['numeric_status'] for r in selected)),
        triage_status=dict(Counter(r['triage_status'] for r in selected)),
        complete_scored_local=sum(r['local_score_raw'] is not None for r in selected),
        numerically_accepted=sum(r['numeric_status'] in ('exceeds_paper', 'numerically_meets_user_threshold') for r in selected))
ranked = sorted([r for r in rows if r['local_score_raw'] is not None and r['numeric_status'] == 'below_paper_by_more_than_0.2'],
    key=lambda r: (r['triage_status'] == 'protocol_incomparable_or_unresolved', r['delta_local_minus_paper_threshold_units']))
now = dt.datetime.now(dt.timezone.utc)
report = dict(schema='user-0.2pp-deficit-triage-v1', created_utc=now.isoformat(),
    created_beijing=now.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat(),
    CPU_only=True, GPU_launches=0, live_source_queue_protocol_prediction_score_service_Git_modified=False,
    user_threshold=dict(value=0.2, unit='official metric percentage points; MME paper-Avg equivalent /20; OCRBench /10',
        inclusive=True, rule='local>=paper passes; paper-0.2<=local<=paper numerically meets; local<paper-0.2 prioritizes cause diagnosis',
        authority='Explicit user work-acceptance criterion, not evidence of closed paper provenance/configuration identity'),
    scope='Tables1/2/4:120 FULL/EADP cells, correct paper budgets. No AZ substitution, native mixing, aggregate Avg from incomplete coverage, or best-seed/maximum-score choice.',
    main_selection_policy='Latest predeclared protocol with complete official score and valid independent gate; newly full but gate-failed/pending controls remain auxiliary until validation is closed.',
    input_snapshots=[source(COVERAGE, 'frozen_full_count_coverage'), source(PAPER, 'paper_reference'), source(SUMMARY, 'latest_original_complete_summary'),
        source(ROOT/'docs/project_handoff.md', 'completely_read_research_context'),
        source(stab/'next_text_full_streamwait.controller.state.json', 'read_only_controller_snapshot')],
    handoff_lines_read=len((ROOT/'docs/project_handoff.md').read_text().splitlines()),
    rows=rows, summary=summary_counts,
    all_numeric_status=dict(Counter(r['numeric_status'] for r in rows)),
    all_triage_status=dict(Counter(r['triage_status'] for r in rows)),
    prioritized_deficits=[{k:r[k] for k in ['model','benchmark','budget_parameter','local_score_raw','paper_score_raw',
        'delta_local_minus_paper_raw','delta_local_minus_paper_threshold_units','threshold_unit','triage_status','protocol_comparability','priority']} for r in ranked],
    new_control_gate_snapshot=dict(status=controller_state.get('status'), updated_utc=controller_state.get('updated_utc'),
        completed={a:j['validation'] for a,j in controller_state.get('jobs',{}).items() if j.get('status')=='complete' and j.get('validation')},
        failures=controller_state.get('error') if controller_state.get('status')=='failed' else {},
        JSON_type_only_recovery=overlay_recovered,
        recovery_evidence=[source(path, 'gate_recovery_evidence') for path in [
            overlay_proof_path, HERE/'resume_next_full_controls_cpu_validation.json', HERE/'json_gate_resume_root_registration.json'] if path.is_file()],
        validation_pending_arms=[k for k,v in control_candidates.items() if v['validation_status']=='validation_pending']),
    limitations=['Paper rounded cells are the user comparison targets; no post hoc rounding before threshold classification.',
        'Passing the numerical criterion is not equivalent to reproducing paper input, engine, checkpoint, parameters, or score execution.',
        'MME perception (LLaVA) and total (Qwen) raw points stay distinct; /20 is an Avg-equivalent comparison unit, not accuracy.',
        'POPE exact macroF1 is derived from the preserved official confusion matrices; disk headline rounded to3 decimals is also retained as official_source_reported_score. F1 is not accuracy.',
        'Current source/file mtimes are not invented historical generation times.',
        'Missing/partial/unscored local EADP is never scored via AnchorZip, native, or paper numbers.',
        'Existing historical protocol scores remain visible; newly complete validation-pending controls are not selected by high score.',
        'No incomplete LLaVA or Qwen overall Avg is computed.'],
    builder=source(__file__, 'CPU_triage_builder'))
report['artifact_registry'] = list(registry.values())
(HERE/'deficit_triage.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
columns = ['model','model_name','pipeline','paper_table','benchmark','method','budget_parameter','paper_budget_total_nominal',
    'expected_prediction_n','metric_denominator_n','local_score_raw','official_source_reported_score','paper_score_raw','metric','threshold_divisor','threshold_unit',
    'delta_local_minus_paper_raw','delta_local_minus_paper_threshold_units','numeric_status','triage_status','protocol_comparability',
    'priority','recommended_action','local_evidence','new_complete_controls','new_control_gap','paper_source','caveats']
with (HERE/'deficit_triage.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], (dict,list)) else row[k] for k in columns})
print(json.dumps(summary_counts, ensure_ascii=False, indent=2))
print('total', report['all_numeric_status'], 'triage', report['all_triage_status'])
for row in report['prioritized_deficits']:
    print(row['model'],row['benchmark'],row['budget_parameter'],row['local_score_raw'],row['paper_score_raw'],row['delta_local_minus_paper_threshold_units'],row['triage_status'])
