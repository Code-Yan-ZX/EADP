"""Select complete historical AnchorZip scores for the user's inspection table."""
import copy
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/media/disk2/YZX/research/EADP_amp')
BASE = ROOT / 'Qwen_vl/outputs/audit_followup_20261008'
HERE = BASE / 'historical_best_20261010'
TABLE = BASE / 'submission_tables_20261009/latest_tables.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def string(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)

def main():
    current = json.loads(TABLE.read_text())
    candidate_paths = [HERE / 'llava_candidates.json', HERE / 'qwen_candidates.json']
    by_key = {}
    for path in candidate_paths:
        audit = json.loads(path.read_text())
        for cell in audit['cells']:
            key = cell['model'], cell['budget'], cell['task']
            assert key not in by_key, key
            by_key[key] = cell
    rows = []
    selected_sources = {}
    changed = []
    missing = []
    for row in current['rows']:
        if row['method'] != 'AnchorZip (ours)':
            continue
        out = {'model': row['model'], 'budget': row['budget'], 'cells': {}}
        tasks = ([
            'TextVQA', 'ChartQA', 'AI2D', 'OCRBench', 'HallusionBench',
            'MME', 'MMB_EN', 'MMB_CN', 'DocVQA', 'InfoVQA'
        ] if row['model'] == 'qwen' else [
            'GQA', 'VizWiz', 'SQA_IMG', 'TextVQA', 'POPE', 'MME',
            'MMB_EN', 'MMB_CN', 'MMVet'
        ])
        for task in tasks:
            key = row['model'], row['budget'], task
            entry = by_key[key]
            candidates = [c for c in entry['candidates']
                          if c.get('complete') and c.get('eligible', c.get('display_eligible', True))]
            if not candidates:
                assert row['cells'].get(task, {}).get('value') is None, key
                out['cells'][task] = {'value': None, 'status': 'unavailable', 'source': None,
                                      'recipe': '', 'caveat': 'No complete scored historical result.'}
                missing.append(key)
                continue
            chosen = max(candidates, key=lambda c: c['value'])
            source = Path(chosen['source'])
            assert source.is_absolute() and source.is_file(), source
            digest = sha(source)
            assert digest == chosen['source_sha256'], source
            selected_sources[str(source)] = digest
            native = row['model'] == 'qwen' and bool(chosen.get('native_vs_legacy_change'))
            cn = row['model'] != 'qwen' and task == 'MMB_CN'
            caveat = chosen.get('caveat', '')
            if isinstance(caveat, list):
                caveat = ' '.join(caveat)
            if cn:
                caveat += ' Chinese task was generated with English instructions (lang=en); this protocol issue remains unresolved.'
            if native:
                caveat += ' Native DeepStack and 3D mRoPE differ from the legacy execution of the EADP row.'
            value = chosen['value']
            maximum = 4000 if task == 'MME' and row['model'] == 'qwen' else 2000 if task == 'MME' else 1000 if task == 'OCRBench' else 100
            assert 0 <= value <= maximum, (key, value)
            cell = dict(value=value, status='historical_best_protocol_caveat' if native or cn else 'historical_best',
                        source=str(source), source_sha256=digest, recipe=string(chosen.get('recipe', 'Audited historical run')),
                        caveat=caveat, historical_candidate_count=len(candidates), complete=True,
                        evaluator='EADP released VizWiz evaluator' if task == 'VizWiz' else chosen.get('metric', 'Corrected benchmark metric'),
                        candidate_audit=str(candidate_paths[1 if row['model'] == 'qwen' else 0]))
            for field in ['prediction', 'prediction_sha256', 'prediction_count', 'metric_denominator', 'n', 'sources']:
                if field in chosen:
                    cell[field] = copy.deepcopy(chosen[field])
            out['cells'][task] = cell
            old = row['cells'].get(task, {}).get('value')
            if old is None or value > old + 0.00005:
                changed.append({'model': row['model'], 'budget': row['budget'], 'task': task,
                                'previous': old, 'historical_best': value})
        rows.append(out)
    selection = {
        'metadata': {
            'created_beijing': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'user_request': '最新的表里面把我们的方法历史中最好的成绩放进去，我看一下。',
            'purpose': 'Historical best inspection; independent per-cell maxima across complete same-model, same-budget AnchorZip runs.',
            'baseline_table': str(TABLE), 'baseline_table_sha256': sha(TABLE),
            'candidate_files_sha256': {str(p): sha(p) for p in candidate_paths},
            'selected_sources_sha256': selected_sources,
            'changed_or_filled_cells': changed, 'unscored_cells': missing,
            'gpu_work': 'None; user abandoned further EADP baseline completion before October 10 at 11:00.',
            'limits': ['LLaVA Chinese MMB histories used English instructions and remain daggered.',
                       'Some Qwen K256 maxima use native DeepStack/mRoPE; the row mixes execution variants.',
                       'Averages summarize independently selected maxima, rather than one experiment.']
        },
        'rows': rows
    }
    target = HERE / 'selected_cells.json'
    target.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'selection': str(target), 'rows': len(rows), 'changed_or_filled': len(changed),
                      'changes': changed, 'unscored': missing}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
