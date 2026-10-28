"""CPU-only historical maxima overlay; never runs a model or changes frozen code.

Usage: python export_historical_best.py --selection selected_cells.json
Optional --dry-run renders and checks a staging export, without touching latest.

Selection schema: {"metadata": {...}, "rows": [{"model": "v15|next|qwen",
"budget": 128, "cells": {"GQA": {"value": 60.0,
"source": "/absolute/path/to/audited/artifact", "source_sha256": "optional",
"recipe": "original|wait-repaired|...", "caveat": "...",
"status": "historical_best", "evaluator": "..."}}}]}
All existing AnchorZip rows must be represented. Missing cells retain their base
value; averages are always recomputed, never selected independently. VizWiz
numeric cells must explicitly name the released evaluator. A status of
protocol_unresolved remains hidden; historical_best_protocol_caveat is shown
with a dagger, including its derived average when all components are present.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import textwrap
from decimal import Decimal, ROUND_HALF_UP

ROOT = Path('/media/disk2/YZX/research/EADP_amp')
HERE = ROOT / 'Qwen_vl/outputs/audit_followup_20261008/historical_best_20261010'
TABLE = ROOT / 'Qwen_vl/outputs/audit_followup_20261008/submission_tables_20261009'
FILES = ['latest_tables.json', 'latest_tables.csv', 'latest_tables.html',
         'latest_tables.tex', 'latest_tables.pdf', 'table_1_v15.png',
         'table_2_next.png', 'table_3_qwen.png', 'core_tables.txt']
LL = ['GQA', 'VizWiz', 'SQA_IMG', 'TextVQA', 'POPE', 'MME',
      'MMB_EN', 'MMB_CN', 'MMVet', 'Avg7']
QW = ['TextVQA', 'ChartQA', 'AI2D', 'OCRBench', 'HallusionBench', 'MME',
      'MMB_EN', 'MMB_CN', 'DocVQA', 'InfoVQA', 'Avg10']
LL_AVG = ['GQA', 'SQA_IMG', 'TextVQA', 'POPE', 'MME', 'MMB_EN', 'MMB_CN']
NAMES = {'v15': 'LLaVA-1.5-7B', 'next': 'LLaVA-NeXT-7B', 'qwen': 'Qwen3-VL-8B'}
NOTE = ('Same model/budget; per-cell maxima across historical runs and generation '
        'recipes, for inspection; no single joint run or paired accuracy claim.')
CN_NOTE = ('Chinese tasks used English instructions in historical runs; daggered '
           'scores and derived averages retain this unresolved protocol caveat.')
QW_NOTE = ('For Qwen, dagger also includes Native DeepStack + 3D mRoPE: historical '
           'execution differs from official legacy. The row contains per-cell '
           'maxima; Avg10 averages these maxima, not one joint run.')
POLICY = ('User-requested historical inspection: choose the highest audited score '
          'for each AnchorZip model/budget/task across historical recipes. '
          'Averages are recomputed from selected components.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def cols(model):
    return QW if model == 'qwen' else LL


def nominal(model, budget):
    return 'FULL' if not budget else str(budget * 5 if model == 'next' else budget)


def shown(cell):
    return bool(cell and cell.get('value') is not None and
                cell.get('status') != 'protocol_unresolved')


def format_cell(cell, task, latex=False):
    if not shown(cell):
        return '--' if latex else '—'
    unit = Decimal('1') if task == 'OCRBench' else Decimal('0.1')
    text = str(Decimal(str(cell['value'])).quantize(unit, rounding=ROUND_HALF_UP))
    if cell.get('status') in {'stale_pending_rerun', 'historical_best_protocol_caveat',
                            'derived_historical_best_protocol_caveat'}:
        text += r'$^{\dagger}$' if latex else '†'
    if task == 'VizWiz' and (cell.get('validation_star') or cell.get('status') not in
                            {'reported_attachment', 'reported_paper'}):
        text += r'$^{*}$' if latex else '*'
    return text


def tex_escape(text):
    mapping = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
               '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
               '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'}
    return ''.join(mapping.get(c, c) for c in str(text))


def checked_cell(cell, model, task, source_hashes):
    result = copy.deepcopy(cell)
    value = result.get('value')
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{model}/{task}: value must be finite numeric or null')
        maximum = 4000 if task == 'MME' and model == 'qwen' else 2000 if task == 'MME' else 1000 if task == 'OCRBench' else 100
        if not 0 <= value <= maximum:
            raise ValueError(f'{model}/{task}: value {value} outside [0,{maximum}]')
        if not isinstance(result.get('recipe'), str) or not result['recipe'].strip():
            raise ValueError(f'{model}/{task}: numeric selection needs recipe')
        source = Path(result.get('source') or '')
        if not source.is_absolute() or not source.is_file():
            raise ValueError(f'{model}/{task}: numeric selection source must be an existing absolute file')
        digest = source_hashes.setdefault(str(source), sha(source))
        expected = result.get('source_sha256')
        if expected and (not re.fullmatch(r'[0-9a-f]{64}', expected) or expected != digest):
            raise ValueError(f'{model}/{task}: source SHA mismatch: {source}')
        result['source_sha256'] = digest
        result['source'] = str(source)
        if task == 'VizWiz' and 'released' not in str(result.get('evaluator', '')).lower():
            raise ValueError(f'{model}/{task}: released evaluator must be explicit')
    result.setdefault('status', 'historical_best' if value is not None else 'unavailable')
    result.setdefault('source', None)
    result.setdefault('recipe', '')
    result.setdefault('caveat', '')
    if not isinstance(result['caveat'], str):
        raise ValueError(f'{model}/{task}: caveat must be a string')
    if result['status'] == 'historical_best_protocol_caveat' and not result['caveat']:
        raise ValueError(f'{model}/{task}: protocol caveat status needs an explanation')
    result['selection_policy'] = NOTE
    return result


def average_cell(row, selection_path, selection_sha):
    model, cells = row['model'], row['cells']
    tasks = QW[:-1] if model == 'qwen' else LL_AVG
    unresolved = [task for task in tasks if not shown(cells.get(task))]
    base = {'value': None, 'status': 'unavailable_or_protocol_pending',
            'source': str(selection_path), 'source_sha256': selection_sha,
            'recipe': 'derived from selected task cells', 'caveat': '',
            'selection_policy': NOTE, 'derived_components': tasks}
    if unresolved:
        base['caveat'] = 'Missing or protocol unresolved: ' + ', '.join(unresolved)
        return base
    base['value'] = sum(cells[t]['value'] / (20 if t == 'MME' else 10 if t == 'OCRBench' else 1)
                        for t in tasks) / len(tasks)
    has_caveat = any(cells[t].get('status') == 'historical_best_protocol_caveat' for t in tasks)
    base['status'] = 'derived_historical_best_protocol_caveat' if has_caveat else 'derived_historical_best'
    base['caveat'] = (QW_NOTE if model == 'qwen' else CN_NOTE) if has_caveat else 'Composite per-cell maxima; no single joint run.'
    return base


def overlay(base, selection, selection_path, selection_sha, stamp):
    data = copy.deepcopy(base)
    targets = {(r['model'], r['budget']): r for r in data['rows'] if r['method'].startswith('AnchorZip')}
    selected = {}
    for item in selection['rows']:
        key = item['model'], item['budget']
        if key in selected or key not in targets:
            raise ValueError(f'Duplicate or unsupported selection row {key}')
        selected[key] = item
    if set(selected) != set(targets):
        raise ValueError(f'Selection must cover all AnchorZip rows; missing={set(targets)-set(selected)}')
    hashes = {}
    changes = []
    for key, row in targets.items():
        model, budget = key
        item = selected[key]
        extra = set(item['cells']) - set(cols(model))
        if extra:
            raise ValueError(f'{key}: unknown tasks {extra}')
        old_cells = copy.deepcopy(row['cells'])
        for task in cols(model)[:-1]:
            cell = item['cells'].get(task)
            if cell is None:
                cell = copy.deepcopy(old_cells.get(task, {'value': None, 'status': 'unavailable', 'source': None}))
                cell.setdefault('recipe', 'unchanged latest historical source')
                if task == 'VizWiz' and cell.get('value') is not None:
                    cell.setdefault('evaluator', 'EADP released evaluator; current latest source')
            row['cells'][task] = checked_cell(cell, model, task, hashes)
        average = cols(model)[-1]
        row['cells'][average] = average_cell(row, selection_path, selection_sha)
        row['method'] = 'AnchorZip (historical best)'
        row['historical_best'] = True
        for task in cols(model):
            old, new = old_cells.get(task, {}), row['cells'][task]
            changes.append({'model': model, 'budget_parameter': budget,
                            'budget_nominal': nominal(model, budget), 'task': task,
                            'old_value': old.get('value'), 'old_display': format_cell(old, task),
                            'old_status': old.get('status'), 'old_source': old.get('source'),
                            'new_value': new.get('value'), 'new_display': format_cell(new, task),
                            'delta': new['value']-old['value'] if new.get('value') is not None and old.get('value') is not None else None,
                            'new_status': new['status'], 'source': new.get('source'),
                            'source_sha256': new.get('source_sha256'), 'recipe': new['recipe'],
                            'caveat': new['caveat'], 'historical_best_note': NOTE})
    unchanged_old = [r for r in base['rows'] if not r['method'].startswith('AnchorZip')]
    unchanged_new = [r for r in data['rows'] if not r['method'].startswith('AnchorZip')]
    if unchanged_old != unchanged_new:
        raise AssertionError('Non-AnchorZip rows changed')
    old_notes = data.get('notes', [])
    data.update(snapshot_beijing=stamp, selection_policy=POLICY,
                historical_best_metadata=copy.deepcopy(selection.get('metadata', {})),
                historical_previous_policy={'selection_policy': base.get('selection_policy'),
                                            'vizwiz_scoring_policy': base.get('vizwiz_scoring_policy')},
                historical_best_selection={'path': str(selection_path), 'sha256': selection_sha},
                historical_best_note=NOTE,
                preceding_snapshot_beijing=base.get('snapshot_beijing'),
                historical_best_source_sha256=hashes)
    data['sources_sha256'].update(hashes)
    data['sources_sha256'][str(selection_path)] = selection_sha
    # Do not carry superseded whole-block selection/no-max wording into the new table.
    data['notes'] = [n for n in old_notes if not ('6072' in n or '60.72' in n or
                     'per-cell maximum' in n.lower() or 'no per-cell' in n.lower() or
                     'recipe block' in n.lower() or 'complete original alpha' in n.lower() or
                     'Qwen128/64 use' in n or 'official-legacy' in n.lower() or
                     'Missing or unresolved MMB cells keep Avg7 empty' in n)]
    data['vizwiz_scoring_policy'] = ('EADP released evaluator for all historical candidates; '
                                   'generation recipe may differ by selected cell. ' + NOTE)
    data['notes'] += [NOTE, CN_NOTE, QW_NOTE, 'VizWiz historical candidates use the EADP released evaluator.',
                      'Avg7 is recomputed from seven visible selected components, including daggered protocol-caveat scores; missing or hidden protocol inputs leave it blank.',
                      'Generation recipes may differ by cell; source, SHA, recipe and caveat are recorded in changes.csv.']
    return data, changes


def footnotes(model):
    lines = [NOTE,
             '— Missing or protocol unresolved. * VizWiz validation, EADP released evaluator. † Historical protocol caveat.',
             'Recipes vary by cell; see changes.csv for sources/SHA/recipes/caveats. NeXT budgets are nominal.',
             'SQA: shipped CQM-A, 2017 image questions. POPE: mean F1 (2910 / 3000 / 3000).',
             'LLaVA MME: perception; Qwen MME: perception + cognition. Reported rows are paper references.']
    if model == 'qwen':
        lines += [QW_NOTE, 'Avg10 = (8 percentage columns + OCRBench/10 + MME/20)/10; derived from selected cells.',
                  'Qwen legacy generation-bank provenance caveats remain; see JSON and changes.csv.']
    else:
        lines += [CN_NOTE, 'Avg7 = (GQA + SQA-IMG + TextVQA + POPE-F1 + MME/20 + MMB-EN + MMB-CN)/7.',
                  'Missing or hidden protocol inputs leave Avg7 blank; a daggered Chinese input also marks the average.']
    return lines


def write_text_formats(out, data, changes):
    save(out / 'latest_tables.json', data)
    with (out / 'changes.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changes[0]))
        writer.writeheader()
        writer.writerows(changes)
    with (out / 'latest_tables.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'budget_nominal', 'method', 'task', 'value', 'display',
                         'status', 'source', 'source_sha256', 'recipe', 'caveat', 'historical_best_note'])
        for row in data['rows']:
            for task in cols(row['model']):
                cell = row['cells'].get(task, {'value': None, 'status': 'unavailable', 'source': None})
                writer.writerow([NAMES[row['model']], nominal(row['model'], row['budget']),
                                 row['method'], task, cell.get('value'), format_cell(cell, task),
                                 cell.get('status'), cell.get('source'), cell.get('source_sha256'),
                                 cell.get('recipe'), cell.get('caveat'), NOTE])
    web = ['<!doctype html><meta charset="utf-8"><title>AnchorZip historical best</title>',
           '<style>body{font-family:Georgia,serif;margin:30px;color:#17202a}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{padding:5px 7px;text-align:right;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}.budget{background:#e7e9ec;text-align:center}.ours{font-weight:bold;background:#f0f6ff}.note{font:14px/1.5 sans-serif;color:#394754}.main-note{font-weight:bold;color:#17202a;background:#fff4cf;padding:12px}@media print{section{break-after:page}table{font-size:10px}}</style>',
           '<p class="main-note">' + html.escape(NOTE) + '</p>',
           '<p class="note">Snapshot: ' + html.escape(data['snapshot_beijing']) + '</p>']
    tex = [r'\documentclass[10pt,a4paper]{article}', r'\usepackage[margin=12mm]{geometry}',
           r'\usepackage{booktabs,graphicx,array,amsmath}', r'\usepackage[table]{xcolor}',
           r'\pagestyle{empty}', r'\setlength{\parindent}{0pt}', r'\begin{document}']
    core = ['Snapshot ' + data['snapshot_beijing'], NOTE, CN_NOTE, '']
    for number, model in enumerate(NAMES, 1):
        tasks = cols(model)
        title = f'Table {number}. {NAMES[model]} -- historical best inspection'
        web += ['<section><h2>' + html.escape(title) + '</h2><table><thead><tr><th>Method</th>' + ''.join('<th>' + html.escape(t) + '</th>' for t in tasks) + '</tr></thead><tbody>']
        tex += [r'{\large\textbf{' + tex_escape(title) + r'}}\par\medskip',
                r'\begingroup\setlength{\tabcolsep}{3.2pt}\renewcommand{\arraystretch}{1.03}',
                r'\resizebox{\textwidth}{!}{\begin{tabular}{l' + 'r' * len(tasks) + '}',
                r'\toprule', 'Method & ' + ' & '.join(tex_escape(t) for t in tasks) + r' \\ \midrule']
        core += [NAMES[model], '', '| 预算 | 方法 | ' + ' | '.join(tasks) + ' |',
                 '|---|---|' + '---:|' * len(tasks)]
        last = None
        for row in (r for r in data['rows'] if r['model'] == model):
            budget = row['budget']
            if budget != last:
                desc = 'Full model' if not budget else 'Retain ' + nominal(model, budget) + ' tokens' + (' (nominal)' if model == 'next' else '')
                web.append('<tr><td class="budget" colspan="' + str(len(tasks) + 1) + '">' + desc + '</td></tr>')
                tex.append(r'\rowcolor{black!12}\multicolumn{' + str(len(tasks) + 1) + r'}{c}{\textit{' + desc + r'}} \\')
                last = budget
            ours = row['method'].startswith('AnchorZip')
            values = [format_cell(row['cells'].get(t), t, True) for t in tasks]
            method = tex_escape(row['method'])
            if ours:
                method = r'\textbf{' + method + '}'
                values = [r'\textbf{' + v + '}' for v in values]
            tex.append(method + ' & ' + ' & '.join(values) + r' \\')
            web.append('<tr' + (' class="ours"' if ours else '') + '><td>' + html.escape(row['method']) + '</td>' + ''.join('<td title="' + html.escape(str(row['cells'].get(t, {}).get('recipe', '')) + ' | ' + str(row['cells'].get(t, {}).get('caveat', '')), quote=True) + '">' + format_cell(row['cells'].get(t), t) + '</td>' for t in tasks) + '</tr>')
            if row['kind'] == 'local' or row['method'] in {'EADP (reported)', 'FULL (reported)'}:
                core.append('| ' + nominal(model, budget) + ' | ' + row['method'] + ' | ' + ' | '.join(format_cell(row['cells'].get(t), t) for t in tasks) + ' |')
        tex += [r'\bottomrule\end{tabular}}\endgroup\par\medskip',
                r'{\footnotesize ' + ' '.join(tex_escape(t) for t in footnotes(model)) + r'\par}']
        web += ['</tbody></table><p class="note">' + '<br>'.join(html.escape(t) for t in footnotes(model)) + '</p></section>']
        core += ['', *footnotes(model), '']
        if model != 'qwen':
            tex.append(r'\newpage')
    tex.append(r'\end{document}')
    (out / 'latest_tables.html').write_text('\n'.join(web) + '\n')
    (out / 'latest_tables.tex').write_text('\n'.join(tex) + '\n')
    (out / 'core_tables.txt').write_text('\n'.join(core) + '\n')


def render(out, data):
    # Same saved-snapshot table rendering approach as the frozen renderer,
    # with independent labels/footnotes and no import of GPU code.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    labels = {'SQA_IMG': r'SQA$^{IMG}$', 'TextVQA': r'VQA$^{Text}$',
              'POPE': 'POPE-F1', 'MMB_EN': r'MMB$^{EN}$',
              'MMB_CN': r'MMB$^{CN}$', 'HallusionBench': 'HallBench',
              'Avg7': r'Avg$_7$', 'Avg10': r'Avg$_{10}$'}
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'pdf.fonttype': 42, 'axes.unicode_minus': False})
    with PdfPages(out / 'latest_tables.pdf') as pdf:
        for number, model in enumerate(NAMES, 1):
            tasks = cols(model)
            rows, kinds, last = [], [], None
            for row in (r for r in data['rows'] if r['model'] == model):
                budget = row['budget']
                if budget != last:
                    desc = 'Full model' if not budget else 'Retain ' + nominal(model, budget) + ' tokens' + (' (nominal)' if model == 'next' else '')
                    rows.append([desc] + [''] * len(tasks))
                    kinds.append('budget')
                    last = budget
                rows.append([row['method']] + [format_cell(row['cells'].get(t), t) for t in tasks])
                kinds.append('ours' if row['method'].startswith('AnchorZip') else 'regular')
            footer = '\n'.join(part for line in footnotes(model) for part in textwrap.wrap(line, 156))
            footer_height = .2 + .17 * (footer.count('\n') + 1)
            height = max(6.5, 1.0 + footer_height + .25 * (len(rows) + 1))
            fig = plt.figure(figsize=(14.8, height), facecolor='white')
            ax = fig.add_axes([.025, footer_height / height, .95, (height - footer_height - .95) / height])
            ax.axis('off')
            table = ax.table(cellText=rows, colLabels=['Method'] + [labels.get(t, t) for t in tasks],
                             colWidths=[.27] + [.73 / len(tasks)] * len(tasks), cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10.4)
            for (i, j), cell in table.get_celld().items():
                cell.set_edgecolor('white')
                cell.set_linewidth(.3)
                cell.PAD = .035
                if i == 0:
                    cell.set_facecolor('#edf0f4')
                    cell.set_text_props(weight='bold', fontsize=9.5)
                    cell.set_edgecolor('#8b929b')
                    cell.set_linewidth(.5)
                elif kinds[i - 1] == 'budget':
                    cell.set_facecolor('#dfe2e6')
                    cell.set_text_props(style='italic', fontsize=10.1)
                elif kinds[i - 1] == 'ours':
                    cell.set_facecolor('#ecf3ff')
                    cell.set_text_props(weight='bold')
                if j == 0:
                    cell.set_text_props(ha='left')
            fig.text(.025, 1 - .3 / height, f'Table {number}: {NAMES[model]} — historical best inspection', fontsize=16, weight='bold')
            fig.text(.025, 1 - .62 / height, 'Snapshot: ' + data['snapshot_beijing'].replace('T', ' ') + ' | AnchorZip cells selected separately across historical recipes', fontsize=9, color='#505966')
            fig.text(.025, .12 / height, footer, fontsize=8.3, linespacing=1.35, va='bottom', color='#414a56')
            fig.savefig(out / f'table_{number}_{model}.png', dpi=150, facecolor='white')
            pdf.savefig(fig)
            plt.close(fig)


def validate_stage(stage, data, changes):
    for name in FILES + ['changes.csv']:
        path = stage / name
        if not path.is_file() or path.stat().st_size < 20:
            raise ValueError(f'Missing or empty staged export: {name}')
    if json.loads((stage / 'latest_tables.json').read_text()) != data:
        raise AssertionError('JSON export round-trip mismatch')
    for name in ['latest_tables.tex', 'latest_tables.html', 'core_tables.txt', 'latest_tables.csv', 'changes.csv']:
        if NOTE not in (stage / name).read_text(encoding='utf-8-sig'):
            raise AssertionError(f'Historical selection note missing from {name}')
    with (stage / 'latest_tables.csv').open(encoding='utf-8-sig', newline='') as handle:
        exported = list(csv.DictReader(handle))
    expected = sum(len(cols(r['model'])) for r in data['rows'])
    if len(exported) != expected:
        raise AssertionError('CSV has unexpected cell count')
    if len(changes) != sum(len(cols(r['model'])) for r in data['rows'] if r['method'].startswith('AnchorZip')):
        raise AssertionError('changes.csv does not cover every AnchorZip cell')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    selection_path = args.selection.resolve()
    selection_sha = sha(selection_path)
    selection = json.loads(selection_path.read_text())
    if not isinstance(selection.get('rows'), list) or not isinstance(selection.get('metadata', {}), dict):
        raise ValueError('Selection needs rows list and optional metadata object')
    for name in FILES:
        if not (TABLE / name).is_file():
            raise ValueError(f'Cannot archive complete old table; missing {name}')
    original_sha = {name: sha(TABLE / name) for name in FILES}
    base = json.loads((TABLE / 'latest_tables.json').read_text())
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    stamp = now.isoformat(timespec='seconds')
    data, changes = overlay(base, selection, selection_path, selection_sha, stamp)
    stage = Path(tempfile.mkdtemp(prefix='staged_historical_best_', dir=HERE))
    write_text_formats(stage, data, changes)
    render(stage, data)
    validate_stage(stage, data, changes)
    if args.dry_run:
        print(json.dumps({'status': 'dry_run_validated', 'staged': str(stage), 'changes': len(changes)}, indent=2))
        return
    if any(sha(TABLE / name) != digest for name, digest in original_sha.items()):
        raise RuntimeError('Latest table changed while staging; refuse overwrite')
    if sha(selection_path) != selection_sha:
        raise RuntimeError('Selection changed while staging; refuse overwrite')
    for path, digest in data['historical_best_source_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError(f'Selected source changed while staging: {path}')
    archive = HERE / ('archive_before_historical_best_' + now.strftime('%Y%m%dT%H%M%S%f%z'))
    archive.mkdir()
    for name in FILES:
        shutil.copy2(TABLE / name, archive / name)
    for name, digest in original_sha.items():
        if sha(archive / name) != digest or sha(TABLE / name) != digest:
            raise RuntimeError(f'Archive failed or latest changed; refuse overwrite: {name}')
    save(archive / 'archive_manifest.json', {'archived_beijing': stamp, 'sources': str(TABLE),
                                           'files_sha256': original_sha, 'selection_sha256': selection_sha})
    if json.loads((archive / 'archive_manifest.json').read_text())['files_sha256'] != original_sha:
        raise RuntimeError('Archive manifest verification failed')
    # Every old format is archived and verified before the first replacement.
    # Roll back old formats from that exact archive if any replacement fails.
    installed = []
    try:
        for name in FILES + ['changes.csv']:
            os.replace(stage / name, TABLE / name)
            installed.append(name)
    except BaseException:
        for name in installed:
            if name in original_sha:
                shutil.copy2(archive / name, TABLE / name)
        raise
    save(HERE / 'last_export_manifest.json', {'exported_beijing': stamp, 'archive': str(archive),
            'selection': str(selection_path), 'selection_sha256': selection_sha,
            'files_sha256': {name: sha(TABLE / name) for name in FILES + ['changes.csv']},
            'changes': len(changes), 'non_ours_rows_unchanged': True,
            'note': NOTE, 'tex_compilation': 'Not attempted; PDF rendered with Matplotlib on CPU.'})
    stage.rmdir()
    print(json.dumps({'status': 'exported', 'archive': str(archive), 'output': str(TABLE),
                      'changes': len(changes), 'non_ours_rows_unchanged': True}, indent=2))


if __name__ == '__main__':
    main()
