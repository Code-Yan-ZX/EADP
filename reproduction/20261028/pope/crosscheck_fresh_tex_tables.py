#!/usr/bin/env python3
"""Read-only99-cell cross-check: fresh official TeX, archived HTML/JSON, comparison."""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import tarfile

from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'arxiv_release_check_20261028'
OUT = ROOT / 'table_source_crosscheck'
ARCHIVE = ROOT / 'official_latest_source.tar.gz'
COMPARISON = HERE.parents[2] / 'docs/versions/v2/eadp_comparison_20261028/comparison.json'
HTML = HERE.parents[2] / 'Qwen_vl/outputs/audit_followup_20261008/submission_tables_20261009/paper_tables_source.html'
CONFIG = [
    ('v15', 'tables/main_llava-1.5-7b.tex', 'S5.T1', 1, [128, 64, 32], 'Avg9'),
    ('next', 'tables/main_llava-1.6-7b.tex', 'S5.T2', 2, [640, 320, 160], 'Avg9'),
    ('qwen', 'tables/main_qwen3-vl-8b.tex', 'S5.T4', 4, [512, 256, 128], 'Avg10')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task(raw, average, html=False):
    if html:
        raw = raw.split('\\')[0]
    key = re.sub(r'[^A-Za-z0-9]', '', re.sub(r'\\[A-Za-z]+', '', raw)).lower()
    return {'vqav2': 'VQAv2', 'gqa': 'GQA', 'vizwiz': 'VizWiz', 'sqaimg': 'SQA_IMG',
        'vqatext': 'TextVQA', 'textvqa': 'TextVQA', 'pope': 'POPE', 'mme': 'MME',
        'mmben': 'MMB_EN', 'mmbcn': 'MMB_CN', 'mmvet': 'MMVet', 'avg': average,
        'chartqa': 'ChartQA', 'ai2d': 'AI2D', 'ocrbench': 'OCRBench', 'hallbench': 'HallusionBench',
        'docvqa': 'DocVQA', 'infovqa': 'InfoVQA'}[key]


def number(raw):
    found = re.findall(r'[-+]?\d+(?:\.\d+)?', raw)
    assert len(found) == 1, raw
    return Decimal(found[0])


def extract_html_rows(rows, average):
    header = next(row for row in rows if row and row[0] == 'Method')
    tasks = [task(value, average, html=True) for value in header[1:]]
    output, budget = {}, None
    for row_index, row in enumerate(rows):
        if not row:
            continue
        retained = re.search(r'Retain\s+(\d+)\s+Tokens', row[0])
        if retained:
            budget = int(retained[1])
        if row[0].startswith('EADP'):
            assert budget is not None and len(row) == len(tasks) + 1
            output[budget] = {'row_index': row_index, 'values': {name: number(value) for name, value in zip(tasks, row[1:])}}
    return tasks, output


def main():
    OUT.mkdir(exist_ok=True)
    comparison_sha, html_sha = sha(COMPARISON), sha(HTML)
    comparison = json.loads(COMPARISON.read_text())
    assert comparison['paper_cells'] == len(comparison['rows']) == 99
    by_key = {(row['model_key'], row['budget_nominal'], row['task']): row for row in comparison['rows']}
    assert len(by_key) == 99
    audit = json.loads((ROOT / 'release_audit.json').read_text())
    assert sha(ARCHIVE) == audit['official_latest_source_archive']['sha256']
    soup = BeautifulSoup(HTML.read_text(), 'html.parser')
    cells, sources, differences = [], [], []
    with tarfile.open(ARCHIVE) as archive:
        for model, member, table_id, table_number, budgets, average in CONFIG:
            payload = archive.extractfile(member).read()
            source_path = OUT / Path(member).name
            source_path.write_bytes(payload)
            lines = payload.decode().splitlines()
            header_index, header_line = next((i, line) for i, line in enumerate(lines, 1) if line.lstrip().startswith('\\textbf{Method}'))
            tex_headers = [column.strip() for column in header_line.split('&')[1:]]
            tasks = [task(value, average) for value in tex_headers]
            expected_tasks = list(dict.fromkeys(row['task'] for row in comparison['rows'] if row['model_key'] == model))
            assert tasks == expected_tasks and len(tasks) == 11
            caption_index, caption_line = next((i, line) for i, line in enumerate(lines, 1) if '\\caption{' in line)
            html_table = soup.find(id=table_id)
            html_rows = [[cell.get_text(' ', strip=True) for cell in row.find_all(['th', 'td'], recursive=False)] for row in html_table.select('tr')]
            html_tasks, html_eadp = extract_html_rows(html_rows, average)
            assert tasks == html_tasks
            json_source_paths = {row['paper_source'] for row in comparison['rows'] if row['model_key'] == model}
            assert len(json_source_paths) == 1
            json_source = Path(next(iter(json_source_paths)))
            archived_json = json.loads(json_source.read_text())
            json_tasks, json_eadp = extract_html_rows(archived_json, average)
            assert tasks == json_tasks
            tex_eadp, current_budget, budget_line = {}, None, None
            for index, line in enumerate(lines, 1):
                retained = re.search(r'Retain\s+(\d+)\s+Tokens', line)
                if retained:
                    current_budget, budget_line = int(retained[1]), index
                if not line.lstrip().startswith('\\textbf{EADP}'):
                    continue
                columns = [part.strip() for part in line.split('&')[1:]]
                assert len(columns) == 11 and current_budget in budgets
                tex_eadp[current_budget] = {'line': index, 'budget_header_line': budget_line, 'raw': line, 'values': {name: number(value) for name, value in zip(tasks, columns)}}
                for column_index, (name, raw_value) in enumerate(zip(tasks, columns), 1):
                    key = (model, current_budget, name)
                    existing = by_key[key]
                    fresh_value = number(raw_value)
                    values = {'fresh_tex': fresh_value, 'archived_html': html_eadp[current_budget]['values'][name],
                        'archived_html_extracted_json': json_eadp[current_budget]['values'][name],
                        'comparison_paper_value': Decimal(str(existing['paper_value']))}
                    matched = len(set(values.values())) == 1
                    record = {'model_key': model, 'budget_nominal': current_budget, 'comparison_budget_cli': existing['budget_cli'],
                        'task': name, 'source_column_1based_after_method': column_index,
                        'source_header': tex_headers[column_index - 1], 'source_cell': raw_value,
                        'source_tex_path': str(source_path), 'archive_member': member,
                        'source_tex_sha256': hashlib.sha256(payload).hexdigest(), 'source_line_1based': index,
                        'budget_header_line_1based': budget_line, 'values': {label: str(value) for label, value in values.items()},
                        'all_four_match_exact_decimal': matched, 'paper_evidence_id': existing['paper_evidence_id'],
                        'paper_split': existing['paper_split'], 'is_paper_reported_average': name.startswith('Avg')}
                    cells.append(record)
                    if not matched:
                        differences.append(record)
            assert list(tex_eadp) == budgets == list(html_eadp) == list(json_eadp)
            assert sorted({row['budget_nominal'] for row in comparison['rows'] if row['model_key'] == model}, reverse=True) == budgets
            sources.append({'model_key': model, 'table_id': table_id, 'table_number': table_number, 'archive_member': member,
                'source_tex_path': str(source_path), 'source_tex_sha256': hashlib.sha256(payload).hexdigest(),
                'source_header_line_1based': header_index, 'source_header_raw': header_line,
                'caption_line_1based': caption_index, 'caption_raw': caption_line,
                'task_order': tasks, 'budget_order': budgets,
                'EADP_source_rows': {str(budget): {name: value for name, value in row.items() if name != 'values'} for budget, row in tex_eadp.items()},
                'archived_html_extraction_json': {'path': str(json_source), 'sha256': sha(json_source)},
                'matched_cells': sum(row['all_four_match_exact_decimal'] for row in cells if row['model_key'] == model)})
    assert len(cells) == 99 and len({(row['model_key'], row['budget_nominal'], row['task']) for row in cells}) == 99
    assert sha(COMPARISON) == comparison_sha and sha(HTML) == html_sha
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'paper_version': 'arXiv2607.02484v1',
        'purpose': 'Only three EADP main tables, three budgets each, eleven columns each. Fresh official TeX vs archivedHTML vs extractedJSON vs existing comparison paper cells.',
        'archive': {'path': str(ARCHIVE), 'sha256': sha(ARCHIVE), 'official_url': 'https://arxiv.org/src/2607.02484'},
        'comparison': {'path': str(COMPARISON), 'sha256_before_and_after': comparison_sha, 'paper_cells': 99, 'unchanged': True},
        'archived_html': {'path': str(HTML), 'sha256_before_and_after': html_sha, 'unchanged': True},
        'matched_cells': len(cells) - len(differences), 'checked_cells': len(cells), 'differences': differences,
        'checked_reported_average_cells': sum(row['is_paper_reported_average'] for row in cells),
        'average_policy': 'Literal published Avg9/Avg10 values only. No local mean created, inferred, filled or overwritten.',
        'table_sources': sources, 'cells': cells,
        'result': 'PASS:99/99 exactDecimal matches; task/budget order matched; existing files not changed.' if not differences else 'MISMATCH:preserve all originals; root must resolve version/mapping before any overwrite.',
        'no_GPU_or_fixed_experiment_change': True}
    (OUT / 'crosscheck.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['result', 'checked_cells', 'matched_cells', 'checked_reported_average_cells']}))
    print(json.dumps({'source_rows': {source['model_key']: source['EADP_source_rows'] for source in sources}}))


if __name__ == '__main__':
    main()
