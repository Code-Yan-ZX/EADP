"""Export complete repaired-protocol results without scoring partial generations.

TextVQA question_id is an image identifier, so integrity uses (ID, full prompt).
The two detached GPU lanes can run independently of this CPU exporter.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import json
import hashlib
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Qwen_vl/outputs/audit_followup_20261008'
BATCH = OUT / 'rerun_batch'
EVAL = ROOT / 'LLaVA/playground/data/eval'
TEXT_CACHE = {}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def entries():
    jobs = []
    for model, lane in [('v15', 1), ('next', 2)]:
        plan = read(BATCH / f'lane{lane}_{model}_plan.json')
        jobs.extend(plan['jobs'])
    # The initial NeXT pair belongs to the launching helper rather than its queue.
    for arm, method in [('EGATHER', 'EADP'), ('LRMAIN025', 'AnchorZip')]:
        output = EVAL / 'anchorzip_p3/next_textvqa_official_20261008' / f'{arm}.jsonl'
        jobs.append(dict(key=f'next/textvqa/{arm}', task='textvqa', model='next',
                         arm=arm, method=method, output=str(output),
                         question_file=str(EVAL / 'textvqa/llava_textvqa_val_v051_ocr.jsonl'),
                         score_log=str(output.with_suffix('.score.txt'))))
    # Already complete shipped-input controls are included under their own protocol.
    for arm in ['FULL', 'E_GATHER_K128', 'AZ_K128']:
        output = (OUT / 'sqa_FULL_CQMA_image2017.jsonl' if arm == 'FULL' else
                  OUT / 'sqa_shipped_control/v15' / f'{arm}.jsonl')
        result = (OUT / 'sqa_FULL_CQMA_image2017_result.json' if arm == 'FULL' else
                  output.with_suffix('.result.json'))
        jobs.append(dict(key=f'v15/sqa/{arm}', task='sqa', model='v15', arm=arm,
                         output=str(output), result=str(result),
                         question_file=str(OUT / 'sqa_CQMA_image2017.json')))
    return jobs


def arm_fields(job):
    arm = job.get('arm') or job['key'].split('/')[-1]
    method = 'FULL' if arm == 'FULL' else ('EADP' if arm.startswith('E') else 'AnchorZip')
    budget = (0 if arm == 'FULL' else 128 if arm in ('EGATHER', 'LRMAIN025') else
              64 if arm == 'LRMAIN0125' else 32 if arm == 'LRMAIN00625' else
              int(re.search(r'K(\d+)$', arm).group(1)))
    return arm, method, budget


def text_score(job, predictions):
    annotation_path = EVAL / 'textvqa/TextVQA_0.5.1_val.json'
    evaluator_path = EVAL.parents[2] / 'llava/eval/m4c_evaluator.py'
    dependencies = [Path(job['output']), Path(job['question_file']),
                    Path(job['score_log']), annotation_path, evaluator_path]
    fingerprint = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in dependencies)
    if fingerprint in TEXT_CACHE:
        return TEXT_CACHE[fingerprint]
    questions = [json.loads(line) for line in Path(job['question_file']).open()]
    keys = [(str(p['question_id']), p['prompt']) for p in predictions]
    expected = [(str(q['question_id']), q['text']) for q in questions]
    if len(keys) != 5000 or len(set(keys)) != 5000 or set(keys) != set(expected):
        raise ValueError('TextVQA composite question keys incomplete or duplicated')
    annotations = read(annotation_path)['data']
    annotation_by_key = {(str(a['image_id']), a['question'].lower()): a for a in annotations}
    if len(annotation_by_key) != 5000:
        raise ValueError('Unexpected TextVQA annotations')
    spec = importlib.util.spec_from_file_location('repair_m4c', evaluator_path)
    evaluator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator_module)
    evaluator = evaluator_module.TextVQAAccuracyEvaluator()
    pairs = []
    for row in predictions:
        annotation = annotation_by_key[(str(row['question_id']), row['prompt'].split('\n')[0].lower())]
        pairs.append(dict(pred_answer=row['text'], gt_answers=annotation['answers']))
    exact = 100 * evaluator.eval_pred_list(pairs)
    # Independent precision check against the official process's rounded output.
    logged = Path(job['score_log']).read_text()
    displayed = float(re.search(r'Accuracy: ([\d.]+)%', logged).group(1))
    if abs(displayed - exact) > .005001:
        raise ValueError('CPU exact score differs from official score log')
    TEXT_CACHE[fingerprint] = exact, [str(annotation_path), str(evaluator_path)]
    return TEXT_CACHE[fingerprint]


def export():
    rows, errors = [], []
    for job in entries():
        arm, method, budget = arm_fields(job)
        output = Path(job['output'])
        task = job['task']
        row = dict(model=job['model'], task=task, arm=arm, method=method,
                   budget_parameter=budget,
                   nominal_budget=(budget * (5 if job['model'] == 'next' else 1)) if budget else None,
                   protocol='official_textvqa_loader' if task == 'textvqa' else 'shipped_CQM_A_image2017',
                   expected_n=5000 if task == 'textvqa' else 2017,
                   generated_n=0, status='pending', accuracy=None,
                   delta_anchorzip_minus_same_local_e=None, output=str(output))
        try:
            if output.exists():
                # The last JSONL row may be being written; only count complete lines.
                raw = output.read_bytes()
                row['generated_n'] = raw.count(b'\n')
                row['status'] = 'generating' if row['generated_n'] < row['expected_n'] else 'awaiting_score'
                result = Path(job.get('result', str(output.with_suffix('.result.json'))))
                scored = (result.exists() if task == 'sqa' else
                          Path(job['score_log']).exists() and 'Accuracy:' in Path(job['score_log']).read_text())
                if scored:
                    predictions = [json.loads(line) for line in raw.decode().splitlines()]
                    if len(predictions) != row['expected_n'] or any(str(p.get('text', '')).strip() in ('', 'FAILED') for p in predictions):
                        raise ValueError('Incomplete/failed prediction file')
                    if task == 'sqa':
                        d = read(result)
                        expected = {str(q['id']) for q in read(job['question_file'])}
                        keys = [str(p['question_id']) for p in predictions]
                        if set(keys) != expected or len(set(keys)) != 2017 or d['count'] != 2017 or set(d['results']) != expected:
                            raise ValueError('SQA IDs or scorer denominator changed')
                        row['accuracy'] = 100 * d['correct'] / d['count']
                        if abs(row['accuracy'] - d['acc']) > 1e-9:
                            raise ValueError('SQA accuracy differs from correct/count')
                        sources = [str(result)]
                    else:
                        row['accuracy'], sources = text_score(job, predictions)
                    row.update(status='complete', prediction_sha256=sha(output),
                               score_sources={s: sha(s) for s in sources})
        except Exception as exc:
            row['status'] = 'invalid'
            row['error'] = str(exc)
            errors.append(dict(job=job['key'], error=str(exc)))
        rows.append(row)
    scores = {(r['model'], r['task'], r['budget_parameter'], r['method']): r['accuracy']
              for r in rows if r['status'] == 'complete'}
    for row in rows:
        e = scores.get((row['model'], row['task'], row['budget_parameter'], 'EADP'))
        if row['method'] == 'AnchorZip' and row['accuracy'] is not None and e is not None:
            row['delta_anchorzip_minus_same_local_e'] = row['accuracy'] - e
    states = {f.name: read(f) for f in BATCH.glob('lane*_state.json')}
    summary = dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   deadline_beijing='2026-10-09 12:00', new_generation_questions=101097,
                   caveats=['No partial-generation accuracy is exported.',
                            'ScienceQA is the complete 2017-image subset, not overall4241.',
                            'NeXT total budgets are nominal five-crop values; actual counts are separate runtime metadata.',
                            'Frozen beta2 is not verified to match every published paper runtime.',
                            'Historical and repaired input protocols must remain separate.'],
                   completed_rows=sum(r['status'] == 'complete' for r in rows), rows=rows, errors=errors,
                   queue_states=states)
    save(BATCH / 'repaired_results_summary.json', summary)
    columns = ['model', 'task', 'arm', 'method', 'budget_parameter', 'nominal_budget',
               'protocol', 'status', 'generated_n', 'expected_n', 'accuracy',
               'delta_anchorzip_minus_same_local_e', 'output']
    temporary = BATCH / 'repaired_results_summary.csv.tmp'
    with temporary.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(BATCH / 'repaired_results_summary.csv')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true', help='Export while the two detached GPU queues run')
    args = parser.parse_args()
    while True:
        summary = export()
        print(json.dumps(dict(updated_utc=summary['updated_utc'], complete=summary['completed_rows'],
                              errors=summary['errors']), ensure_ascii=False), flush=True)
        if not args.watch or summary['errors'] or all(s['status'] in ('complete', 'failed') for s in summary['queue_states'].values()):
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
