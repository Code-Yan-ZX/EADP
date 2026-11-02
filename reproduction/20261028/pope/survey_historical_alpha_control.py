#!/usr/bin/env python3
"""Inventory small structured protocols, never open large per-question traces."""
import collections
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path('/media/disk2/YZX/research')
ROOTS = [BASE / repo / suffix for repo in ['EADP', 'EADP_amp'] for suffix in [
    'Qwen_vl/outputs', 'LLaVA/playground/data/eval/anchorzip_p3']]
ROOTS.append(ROOT)
SKIP = {'hourly_monitor', 'group_publish', 'published_results', '__pycache__', '.git',
        'model_identity', 'fresh_coco_panel_images', 'inputs', 'pinned_pope_labels'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    scanned, skipped_large, rows, errors = [], [], [], []
    candidate_files = 0
    for root in ROOTS:
        if not root.exists():
            continue
        for directory, directories, filenames in os.walk(root, followlinks=False):
            directories[:] = [d for d in directories if d not in SKIP and not Path(directory, d).is_symlink()]
            for name in sorted(filenames):
                path = Path(directory, name)
                if not name.endswith('.json') or path.is_symlink():
                    continue
                if not any(x in name.lower() for x in ['protocol', 'plan', 'state', 'summary', 'finished', 'manifest', 'pope']):
                    continue
                if path.stat().st_size > 2_000_000:
                    skipped_large.append({'path': str(path), 'bytes': path.stat().st_size})
                    continue
                candidate_files += 1
                data = path.read_bytes()
                if b'pope' not in data.lower() and 'pope' not in str(path).lower():
                    continue
                try:
                    value = json.loads(data)
                except Exception as exc:
                    errors.append({'path': str(path), 'error': str(exc)})
                    continue
                scanned.append({'path': str(path), 'bytes': len(data), 'sha256': sha(data)})

                def walk(obj, pointer='', context=None):
                    context = dict(context or {})
                    if isinstance(obj, dict):
                        for k in ['task', 'model', 'model_path', 'model-path', 'key', 'job_id', 'run_id', 'question_file', 'n', 'budget', 'K']:
                            if k in obj and isinstance(obj[k], (str, int, float, bool)):
                                context[k] = obj[k]
                        if 'alpha' in obj and 'beta' in obj:
                            hint = ' '.join(str(v) for v in context.values()) + ' ' + str(path) + ' ' + pointer
                            if 'pope' in hint.lower():
                                params = {k: obj.get(k) for k in ['alpha', 'beta', 'visual_token_num', 'anchorzip', 'temperature', 'max_new_tokens', 'num_beams', 'conv_mode', 'question_file', 'model_path'] if k in obj}
                                waits = 'wait' in (str(path) + ' ' + json.dumps(obj) + ' ' + str(value.get('protocol', '') if isinstance(value, dict) else '')).lower()
                                rows.append({'metadata_path': str(path), 'json_pointer': pointer, 'context': context,
                                    'parameters': params, 'wait_mentioned_in_path_or_protocol': waits,
                                    'metadata_sha256': sha(data)})
                        for k, v in obj.items():
                            if isinstance(v, (dict, list)):
                                walk(v, pointer + '/' + str(k), context)
                    elif isinstance(obj, list):
                        for i, v in enumerate(obj):
                            if isinstance(v, (dict, list)):
                                walk(v, pointer + '/' + str(i), context)
                walk(value)
    exact = []
    for row in rows:
        p, c = row['parameters'], row['context']
        budget = p.get('visual_token_num', c.get('budget', c.get('K')))
        model_hint = str(p.get('model_path', c.get('model_path', c.get('model', '')))).lower()
        if p.get('alpha') == .5 and p.get('beta') == 1. and budget == 32 and ('v1.5' in model_hint or model_hint == 'v15') and p.get('anchorzip', False) is False:
            exact.append(row)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'roots': [str(x) for x in ROOTS],
        'scope': 'Small structured JSON protocols/plans/states/manifests only; no runtime.jsonl, prediction.jsonl or large monitor snapshots were scanned. Publication/hourly copies are excluded because they duplicate preserved original runs.',
        'excluded_directory_names': sorted(SKIP), 'small_metadata_candidates_examined': candidate_files,
        'pope_related_metadata_read': len(scanned), 'large_metadata_skipped': skipped_large, 'parse_errors': errors,
        'pope_parameter_records': rows, 'exact_v15_K32_alpha05_beta1_nonanchor_metadata': exact,
        'found_complete_wait_run': False if not exact else None,
        'conclusion': 'No matching structured historical run was found; a new single-alpha control is warranted, subject to root GPU allocation.' if not exact else 'Matching metadata requires manual completeness, wait proof, and raw-prediction verification before declaring duplication.',
        'metadata_inventory': scanned}
    (ROOT / 'historical_alpha_control_survey.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'metadata_candidates': candidate_files, 'pope_metadata': len(scanned), 'parameter_records': len(rows),
        'exact_candidate_records': len(exact), 'large_metadata_skipped': len(skipped_large), 'parse_errors': len(errors)}))
    for row in exact:
        print(json.dumps(row))


if __name__ == '__main__':
    main()
