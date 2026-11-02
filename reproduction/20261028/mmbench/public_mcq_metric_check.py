"""Check completed MMB answers with unchanged released VLMEvalKit MCQ helpers.

CPU only, no model/API imports. This is not an external server submission.
"""
import ast
from collections import defaultdict
import copy as cp
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import string
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
AUTHOR = HERE.parents[2] / '_reproduction/author_eadp'
UTILS = AUTHOR / 'Qwen_vl/VLMEvalKit/vlmeval'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_helpers(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    # Function ASTs are compiled unchanged; package imports that include model
    # registries are deliberately not executed.
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return {'path': str(path), 'sha256': sha(path), 'functions': names,
            'function_AST_sha256': hashlib.sha256(ast.dump(module).encode()).hexdigest()}


def main():
    os.environ.pop('VERBOSE', None)
    namespace = dict(cp=cp, os=os, re=re, string=string, pd=pd)
    sources = [original_helpers(UTILS / 'utils/matching_util.py',
        ['can_infer_option', 'can_infer_text', 'can_infer'], namespace)]
    sources.append(original_helpers(UTILS / 'dataset/utils/multiple_choice.py',
        ['build_choices', 'prefetch_answer', 'prefetch_circular_group'], namespace))
    runs = []
    for directory in [HERE, HERE / 'streamwait_control']:
        protocol = json.loads((directory / 'protocol.json').read_text())
        verified = json.loads((directory / 'independent_verification.json').read_text())
        assert verified['success'] and verified['all_rows_checked']
        dataset = Path(protocol['dataset']['path'])
        assert sha(dataset) == protocol['dataset']['sha256']
        command = protocol['command']
        predictions = Path(command[command.index('--answers-file') + 1])
        assert sha(predictions) == verified['prediction_sha256']
        answers = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
        data = pd.read_table(dataset)
        assert len(data) == len(answers) == 4377
        assert data['index'].tolist() == [int(answer['question_id']) for answer in answers]
        assert data['index'].is_unique
        data['prediction'] = [answer['text'] for answer in answers]
        data['GT'] = data['answer']
        data['g_index'] = [int(index % 1000000) for index in data['index']]
        assert set(data['g_index']) == set(data.loc[data['index'] == data['g_index'], 'index'])
        parser_changes, group_results, unresolved = [], {}, []
        for _, item in data.iterrows():
            public = namespace['prefetch_answer'](item)
            labels = re.findall(r'\b[A-E]\b', item['prediction'].strip())
            local = labels[0] if labels else None
            if public != local:
                parser_changes.append({'index': int(item['index']), 'local': local, 'public': public})
        for index, group in data.groupby('g_index', sort=True):
            result = namespace['prefetch_circular_group'](group)
            if result is None:
                # The public exact_matching mode gives unresolved groups hit0;
                # no API or random matching is substituted.
                unresolved.append(int(index))
                hit = 0
            else:
                hit = result['hit']
            assert hit in (0, 1)
            group_results[int(index)] = hit
        assert len(group_results) == 1176
        public_score = 100 * sum(group_results.values()) / len(group_results)
        assert abs(public_score - verified['circular_accuracy_percent']) < 1e-12
        runs.append({'run_dir': str(directory), 'n_rows': len(data), 'n_groups': len(group_results),
            'circular_correct': sum(group_results.values()), 'public_exact_matching_percent': public_score,
            'independent_local_percent': verified['circular_accuracy_percent'],
            'parser_changed_rows': parser_changes, 'unresolved_groups': unresolved,
            'prediction_sha256': sha(predictions), 'dataset_sha256': sha(dataset),
            'verification_sha256': sha(directory / 'independent_verification.json')})
    assert 'torch' not in sys.modules
    report = {'success': True, 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'runs': runs, 'unchanged_public_functions': sources, 'script_sha256': sha(__file__),
        'CPU_only': True, 'model_loaded': False, 'API_used': False,
        'boundary': 'Released author-vendored VLMEvalKit exact_matching helpers on all completed oldEN4377 answers. External official-server/judge evaluation has not been run.'}
    (HERE / 'public_mcq_metric_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'success': True, 'runs': [{key: row[key] for key in (
        'n_rows', 'n_groups', 'circular_correct', 'public_exact_matching_percent',
        'parser_changed_rows', 'unresolved_groups')} for row in runs], 'CPU_only': True}))


if __name__ == '__main__':
    main()
