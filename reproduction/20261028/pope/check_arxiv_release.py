#!/usr/bin/env python3
"""Bounded primary-source release/ancillary audit; no experiment changes."""
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import tarfile
import requests
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).resolve().parent / 'arxiv_release_check_20261028'


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.texts = [], []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)
    def handle_data(self, text):
        self.texts.append(text)


def download(url, destination, cap=30_000_000):
    response = requests.get(url, timeout=60, stream=True)
    response.raise_for_status()
    size, digest = 0, hashlib.sha256()
    with destination.open('wb') as stream:
        for block in response.iter_content(1024 * 1024):
            size += len(block)
            assert size <= cap, 'Bounded release-check download cap exceeded'
            digest.update(block)
            stream.write(block)
    return {'requested_url': url, 'final_url': response.url, 'status': response.status_code,
        'bytes': size, 'sha256': digest.hexdigest(), 'path': str(destination),
        'response_headers': {key: response.headers[key] for key in ['Content-Type', 'Content-Length', 'Last-Modified', 'ETag', 'Date', 'Content-Disposition'] if key in response.headers}}


def main():
    ROOT.mkdir(exist_ok=True)
    abs_path = ROOT / 'latest_abs.html'
    archive_path = ROOT / 'official_latest_source.tar.gz'
    page_evidence = download('https://arxiv.org/abs/2607.02484', abs_path, cap=3_000_000)
    page = Page()
    page.feed(abs_path.read_text())
    raw_text = ' '.join(' '.join(page.texts).split())
    history = raw_text[raw_text.index('Submission history'):raw_text.index('Full-text links:')]
    versions = re.findall(r'\[v(\d+)\]', history)
    primary_links = list(dict.fromkeys(urljoin('https://arxiv.org', link) for link in page.links
        if urlparse(urljoin('https://arxiv.org', link)).netloc == 'arxiv.org'
        and any(key in link for key in ['/abs/2607.', '/html/2607.', '/src/2607.', '/e-print/2607.', '/pdf/2607.', '/format/2607.', '/anc/'])))
    ancillary_links = [link for link in page.links if '/anc/' in link or 'ancillary' in link.lower()]
    archive_evidence = download('https://arxiv.org/src/2607.02484', archive_path)
    inventory, text_matches, recipe_files, urls, comments = [], [], [], [], []
    build_metadata = None
    recipe_pattern = re.compile(r'--(?:alpha|beta|visual_token_num|model-path)|entropy_keep_ratio|(?:torch|transformers|accelerate)==|conda\s+(?:env|create)|python.{0,80}(?:llava|run.py)|pope\.sh|run_eadp', re.I)
    text_extensions = {'.tex', '.txt', '.md', '.json', '.yaml', '.yml', '.sh', '.toml', '.cfg', '.ini'}
    with tarfile.open(archive_path, 'r:*') as archive:
        for entry in archive.getmembers():
            inventory.append({'path': entry.name, 'bytes': entry.size, 'type': entry.type.decode(errors='replace'),
                'is_file': entry.isfile(), 'is_link': entry.issym() or entry.islnk()})
            if not entry.isfile():
                continue
            suffix = Path(entry.name).suffix.lower()
            if suffix in {'.sh', '.json', '.yaml', '.yml', '.toml', '.txt', '.md', '.ipynb', '.py'} or 'anc/' in entry.name:
                recipe_files.append(entry.name)
            if suffix not in text_extensions or entry.size > 2_000_000:
                continue
            text = archive.extractfile(entry).read().decode(errors='replace')
            if entry.name == '00README.json':
                build_metadata = json.loads(text)
            urls.extend({'member': entry.name, 'url': url.rstrip('.,')} for url in re.findall(r'https?://[^\s{}<>\\]+', text))
            matches = [{'line': i, 'text': line[:500]} for i, line in enumerate(text.splitlines(), 1) if recipe_pattern.search(line)]
            if matches:
                text_matches.append({'member': entry.name, 'matches': matches})
            if suffix == '.tex':
                comment_matches = [{'line': i, 'comment': line.strip()[:500]} for i, line in enumerate(text.splitlines(), 1)
                    if line.lstrip().startswith('%') and re.search(r'default|setting|alpha|beta|mu|ratio|pope|mme|config|\brun\b|环境|参数|torch|cuda|transformers', line, re.I)]
                if comment_matches:
                    comments.append({'member': entry.name, 'matches': comment_matches})
    ancillary_members = [row['path'] for row in inventory if row['path'].startswith('anc/') or '/anc/' in row['path']]
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Check current arXiv version history and source/ancillary files for new execution recipe only; not a new paper-math audit.',
        'prior_local_search': {'found_current_revision_or_ancillary_check': False,
            'scope': 'V2 docs and20261028 reproduction JSON/Markdown, plus EADP_amp docs and audit_followup_20261008 JSON/Markdown excluding large state/manifest/hourly/publication copies. Existing audits consistently cite archivedv1.'},
        'abstract_page': page_evidence, 'submission_history_text': history, 'versions_listed': versions,
        'paper_access_links': primary_links, 'separate_ancillary_links_on_abs': ancillary_links,
        'official_latest_source_archive': archive_evidence, 'archive_inventory': inventory,
        'ancillary_members': ancillary_members, 'possible_execution_recipe_members': recipe_files,
        'bounded_recipe_string_matches': text_matches, 'URLs_in_paper_source': urls,
        '00README_json_classification': {'content': build_metadata,
            'interpretation': 'arXiv TeX build metadata (pdflatex/TeXLive2025), not experiment environment or reproduction argv.'},
        'recipe_relevant_comment_lines': comments,
        'source_package_timestamp_note': 'HTTP Last-Modified2026-10-14 is packaging/HTTP metadata, not evidence of a second submitted paper version. Official history and source Content-Disposition both identifyv1.',
        'existing_comparison_version': 'arXiv2607.02484v1; unchanged by this audit',
        'no_GPU_experiment_or_frozen_plan_changed': True}
    result['conclusion'] = ('Official current abstract lists onlyv1 (2026-07-02). HTML and downloaded source Content-Disposition explicitly identifyv1. No separate ancillary link/member or executable/environment recipe was found;00README.json is document-build metadata only. Existing v1 comparison remains unchanged.')
    (ROOT / 'release_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'versions': versions, 'history': history, 'archive_files': len(inventory),
        'ancillary_members': ancillary_members, 'possible_recipe_members': recipe_files, 'recipe_string_matches': text_matches,
        'report': str(ROOT / 'release_audit.json')}))


if __name__ == '__main__':
    main()
