#!/usr/bin/env python3
"""Read-only native prediction inventory, including Git-ignored output trees."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOTS = [Path('/media/disk2/YZX/research') / name for name in ('EADP', 'EADP_amp', 'EADP_v2')]
OLD_INVENTORY = ROOTS[1] / 'Qwen_vl/outputs/audit_followup_20261008/mmb_old_version_reuse_20261010/prediction_inventory'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    csv.field_size_limit(2**31 - 1)
    with (BASE / 'data/mmbench_dev_20230712.tsv').open(newline='') as stream:
        expected = [int(row['index']) for row in csv.DictReader(stream, delimiter='\t')]
    paths = subprocess.check_output(['rg', '--files', '-uuu', *map(str, ROOTS), '-g', '*.jsonl'], text=True).splitlines()
    candidates, errors, huge = [], [], []
    for name in paths:
        path = Path(name)
        try:
            with path.open() as stream:
                line = stream.readline(4 * 1024 * 1024)
            if len(line) == 4 * 1024 * 1024 and not line.endswith('\n'):
                huge.append(name)
                continue
            first = json.loads(line)
            if not isinstance(first, dict) or not {'question_id', 'options', 'option_char', 'prompt', 'text', 'model_id'} <= first.keys():
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            record = {'path': name, 'sha256': sha(path), 'n_rows': len(rows),
                      'model_ids': sorted({str(row.get('model_id')) for row in rows}),
                      'ordered_ids_match_old_EN4377': [int(row['question_id']) for row in rows] == expected}
            related = []
            for q in (path.with_name('protocol.json'), Path(str(path) + '.protocol.json'),
                      Path(str(path).replace('.jsonl', '.control.protocol.json')),
                      path.parent / 'observed/loaded_model.json'):
                if q.exists():
                    meta = json.loads(q.read_text())
                    related.append({'path': str(q), 'sha256': sha(q),
                                    'parameters': meta.get('parameters'), 'beta': meta.get('beta'),
                                    'alpha': meta.get('alpha'), 'K': meta.get('visual_token_num')})
            record['metadata'] = related
            if record['ordered_ids_match_old_EN4377'] and record['model_ids'] == ['llava-v1.5-7b']:
                assert related, 'Target full-run identity must have metadata'
                actual = [m['beta'] for m in related if m['beta'] is not None]
                actual += [m['parameters']['beta'] for m in related if m['parameters'] and 'beta' in m['parameters']]
                record['observed_beta_values'] = actual
                record['matches_beta2_target'] = bool(actual) and all(float(beta) == 2 for beta in actual)
            else:
                record['matches_beta2_target'] = False
            candidates.append(record)
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append({'path': name, 'error': type(error).__name__, 'message': str(error)[:250]})
    previous = json.loads((OLD_INVENTORY / 'summary.json').read_text())
    assert not previous['confirmed_target_old_full_prediction_found']
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'GPU_used': False,
              'scope': 'Current three project trees, including ignored files: first record of all native JSONL, complete parse of every native MMB candidate. Previous audited alternate-format/archive inventory is retained as supporting evidence.',
              'roots': list(map(str, ROOTS)), 'jsonl_paths_found': len(paths),
              'native_MMB_candidates': candidates, 'first_record_errors_or_non_native_formats': errors,
              'first_line_exceeds4MiB': huge,
              'matching_full_old_EN_v15_K128_alpha05_beta2_wait_found': any(r['matches_beta2_target'] for r in candidates),
              'previous_archive_and_alternate_format_inventory': [
                  {'path': str(OLD_INVENTORY / name), 'sha256': sha(OLD_INVENTORY / name)}
                  for name in ('summary.json', 'native_MMB_inventory.json', 'packed_MMB_inventory.json', 'alternate_format_paths.json')],
              'limits': 'A bounded inventory cannot establish absence outside the listed project trees. Prior V11/beta2 runs and old 13B examples are not exact old4377 v15-7B controls.',
              'auditor_sha256': sha(__file__)}
    (OUT / 'history_inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'jsonl_paths': len(paths), 'native_MMB_candidates': len(candidates),
                      'exact_oldEN_v15_candidates': [r for r in candidates if r['ordered_ids_match_old_EN4377'] and r['model_ids'] == ['llava-v1.5-7b']],
                      'full_beta2_target_found': report['matching_full_old_EN_v15_K128_alpha05_beta2_wait_found'],
                      'overlarge_first_lines': huge}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
