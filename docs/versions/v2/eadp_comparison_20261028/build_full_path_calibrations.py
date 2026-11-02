"""Render four accepted FULL calibrations separately from EADP pruning results."""
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RUNS = ROOT / 'reproduction/20261028'


def artifact(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    specs = [
        ('v15', 'POPE', 'pope/baseline_full_rescore/baseline_full_summary.json',
         '5dea93323de5309e0a2599dd29fe453f5ea176af2e377c0baf34c3ffd703709e'),
        ('v15', 'MMB_EN', 'mmbench/full_token_baseline_control/independent_verification.json',
         'b4227aaec954a0b9aeaa640b334518d85b32fe4d2785f5a30f2ed9080a2d2252'),
        ('next', 'TextVQA', 'next_textvqa/full_token_baseline_recheck/baseline_recheck.json',
         '1433989686cc10cc2c6029a4b355f06a6a0f8fc6851cb2017f74e8dedef2ae29'),
        ('qwen', 'DocVQA', 'qwen/independent_audit/full_doc_baseline_recheck/baseline_recheck.json',
         '29275be39dc9215b0264f3e02d61ee82089ace6100de0d2470e67debe00f3c18'),
    ]
    evidence_path = OUT.parent / 'evidence.json'
    evidence = json.loads(evidence_path.read_text())['records']
    rows = []
    for model, task, relative, digest in specs:
        path = RUNS / relative
        report_source = artifact(path)
        assert report_source['sha256'] == digest
        report = json.loads(path.read_text())
        references = [r for r in evidence if r.get('origin') == 'eadp_paper_reported'
                      and (r.get('model'), r.get('task'), r.get('method')) == (model, task, 'FULL')]
        assert len(references) == 1
        paper = references[0]
        additional = []
        if task == 'POPE':
            record = report['wait_only_full']
            assert record['complete'] and record['n_predictions'] == 8910
            value, n = record['official_macro_f1_percent'], 8910
            assert value == report['original_full']['official_macro_f1_percent']
            raw = dict(path=record['prediction_path'], sha256=record['predictions_sha256'])
            condition = '历史完整基线重新评分；两份完整回答相同'
            generation = 'historical_rechecked'
        elif task == 'MMB_EN':
            directory = path.parent
            proof_path = directory / 'root_terminal_process_proof.json'
            proof = json.loads(proof_path.read_text())
            assert proof['root_owned_exec_actual_exit_code'] == 0 and proof['both_original_proc_paths_absent']
            assert proof['independent_verification_sha256'] == digest
            assert report['success'] and report['mask_observation_checks'] == 4377
            value, n = report['primary_value_percent'], 4377
            raw = dict(path=str(directory / 'v15_FULL_oldEN_streamwait.jsonl'), sha256=proof['raw_sha256'])
            additional = [artifact(proof_path), artifact(directory / 'peer_terminal_review.json')]
            condition = '本轮完整新生成；旧EN4377/1176组；每题实际576 tokens；公开exact circular'
            generation = 'new_complete_generation'
        elif task == 'TextVQA':
            assert report['success'] and report['n'] == 5000
            value, n = float(report['independent_full_accuracy_percent']), 5000
            source = next(s for s in report['sources_before_after_unchanged']
                          if s['path'].endswith('/next_text_FULL_streamwait.jsonl'))
            raw = dict(path=source['path'], sha256=source['sha256'])
            condition = '历史完整基线重新评分；每题实际2880 tokens；一位小数为60.4'
            generation = 'historical_rechecked'
        else:
            assert report['status'] == 'complete_CPU_baseline_recheck' and report['n'] == 5349
            value, n = report['native_value_percent'], 5349
            raw = dict(report['sources'][0])
            condition = '历史注册FULL路径；当时版本/backend及逐题tokens未记录，不是新FA2同环境对照'
            generation = 'historical_registered_full_route_rechecked'
        assert artifact(raw['path'])['sha256'] == raw['sha256']
        delta = Decimal(str(value)) - Decimal(str(paper['value']))
        rows.append(dict(model=model, task=task, method='FULL', n=n,
            nominal_tokens=paper['budget_nominal'], paper_value=paper['value'], local_value=value,
            local_minus_paper=str(delta), unit='percentage_points', generation=generation,
            same_one_decimal=Decimal(str(value)).quantize(Decimal('.1'), rounding=ROUND_HALF_UP) == Decimal(str(paper['value'])),
            conditions=condition, report=report_source, raw_prediction=raw, additional_evidence=additional,
            paper_evidence_id=paper['id']))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        definition='FULL-path calibration only; kept outside the 99 paper pruning cells and 9 new EADP runs. No maximum selection or missing-task average.',
        sources=[artifact(evidence_path), artifact(__file__)], rows=rows, goal_complete=False)
    (OUT / 'full_path_calibrations.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    fields = ['model', 'task', 'method', 'n', 'nominal_tokens', 'paper_value', 'local_value',
              'local_minus_paper', 'unit', 'generation', 'same_one_decimal', 'conditions']
    with (OUT / 'full_path_calibrations.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(FULL_calibrations=len(rows), new_FULL_generations=1, goal_complete=False)))


if __name__ == '__main__':
    main()
