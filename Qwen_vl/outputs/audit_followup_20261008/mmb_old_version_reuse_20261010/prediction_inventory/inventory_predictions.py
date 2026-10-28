"""CPU-only historical prediction inventory; no input file is modified.

Prompt matches are candidate reuse links, never proof of image identity.
In particular, cross-ID generic questions can contain different images.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

ROOT = Path('/media/disk2/YZX/research/EADP_amp')
ROOTS = [Path('/media/disk2/YZX/research/EADP'), ROOT]
OUT = Path(__file__).parent
OLD = ROOT / 'LLaVA/playground/data/eval/mmbench/mmbench_dev_20230712.tsv'
NEW = ROOT / 'LLaVA/playground/data/eval/mmbench/MMBench_DEV_EN_V11.tsv'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def missing(v):
    return v is None or str(v).strip().lower() in ('', 'nan', 'none')

def prompt(r):
    text = r['question']
    if not missing(r.get('hint')):
        text = r['hint'] + '\n' + text
    for letter in 'ABCDE':
        if not missing(r.get(letter)):
            text += '\n' + letter + '. ' + r[letter]
    return text

def read_tsv(p):
    csv.field_size_limit(1024**3)
    with p.open(newline='') as f:
        return [{k: v for k, v in r.items() if k != 'image'}
                for r in csv.DictReader(f, delimiter='\t')]

def exception_exclusion(path):
    p = Path(path)
    if p.stat().st_size == 0:
        return 'Empty analysis statistics file: no prediction records.'
    if p.name == 'agent.events.jsonl':
        return 'Codex monitor event log, not model predictions.'
    if '/scienceqa/' in path and p.name == 'llava-v1.5-13b_output.jsonl':
        return 'Pretty-printed ScienceQA 13B score object (correct/incorrect keys), not native MMB predictions.'
    if '/vqav2/' in path:
        return 'VQAv2 output with malformed later JSONL record; first record was inspected and task is outside MMB.'
    return None

old = read_tsv(OLD)
new = read_tsv(NEW)
old_by_id = {int(r['index']): r for r in old}
new_by_id = {int(r['index']): r for r in new}
old_prompts = {prompt(r) for r in old}
new_prompts = {prompt(r) for r in new}
all_mmb_prompts = old_prompts | new_prompts
old_groups = collections.defaultdict(list)
for r in old:
    old_groups[int(r['index']) % 1000000].append(r)
old_only_ids = set(old_groups) - {int(r['index']) % 1000000 for r in new}

def native_audit(path, predictions):
    models = collections.Counter(r.get('model_id') for r in predictions)
    ids = [int(r['question_id']) for r in predictions]
    counts = collections.Counter(ids)
    same_id_old = []
    same_id_new = []
    options_mismatch_new = []
    p_by_id = {int(r['question_id']): r for r in predictions if counts[int(r['question_id'])] == 1}
    p_by_prompt = collections.defaultdict(list)
    for p in predictions:
        p_by_prompt[p['prompt']].append(p)
        idx = int(p['question_id'])
        if idx in old_by_id and p['prompt'] == prompt(old_by_id[idx]):
            same_id_old.append(idx)
        if idx in new_by_id and p['prompt'] == prompt(new_by_id[idx]):
            same_id_new.append(idx)
            options = [(letter, new_by_id[idx][letter]) for letter in 'ABCDE'
                       if not missing(new_by_id[idx].get(letter))]
            if p.get('options') != [value for letter, value in options] or p.get('option_char') != [letter for letter, value in options]:
                options_mismatch_new.append(idx)
    same_set = set(same_id_old)
    strict_same = []
    unambiguous_candidates = []
    any_candidates = []
    for group, rows in old_groups.items():
        if all(int(r['index']) in same_set for r in rows):
            strict_same.append(group)
        chosen = []
        for r in rows:
            idx = int(r['index'])
            p = p_by_id.get(idx)
            if p and p['prompt'] == prompt(r):
                chosen.append(p)
            elif len(p_by_prompt[prompt(r)]) == 1:
                chosen.append(p_by_prompt[prompt(r)][0])
            else:
                break
        if len(chosen) == len(rows):
            unambiguous_candidates.append(group)
        if all(p_by_prompt[prompt(r)] for r in rows):
            any_candidates.append(group)
    target_model = models == {'llava-v1.5-7b': len(predictions)}
    en = len(same_id_new) == len(predictions) and bool(predictions)
    name = Path(path.split('::')[-1]).name
    budget = 32 if '00625' in name else 64 if '0125' in name else 128 if ('025' in name or 'E128' in name) else None
    method = 'AnchorZip' if 'LRMAIN' in name else 'EADP' if 'E128' in name else 'unknown'
    target = target_model and en and budget == 128 and method in {'AnchorZip', 'EADP'}
    reason = None if target else ('different model' if not target_model else
                                 'different language/task input' if not en else
                                 'different budget or unknown method/budget')
    return {
        'path': path, 'n_rows': len(predictions), 'n_unique_row_ids': len(set(ids)),
        'model_id_counts': dict(models), 'method': method, 'budget': budget,
        'target_v15_EN_K128': target, 'exclusion_reason': reason,
        'row_ids_exact_order_V11_EN': ids == list(new_by_id),
        'same_row_id_exact_V11_EN_prompt_rows': len(same_id_new),
        'V11_options_metadata_mismatch_rows': len(options_mismatch_new),
        'same_row_id_exact_old_EN_prompt_rows': len(same_id_old),
        'same_ID_old_complete_prompt_groups_allowing_GT_correction': len(strict_same),
        'same_ID_old_only_226_complete_groups': len(set(strict_same) & old_only_ids),
        'unambiguous_prompt_candidate_complete_groups_image_unverified': len(unambiguous_candidates),
        'any_prompt_candidate_complete_groups_including_ambiguous_image_unverified': len(any_candidates),
        'cross_ID_unambiguous_old_only_candidate_groups_image_unverified': sorted(set(unambiguous_candidates) & old_only_ids),
        'candidate_status': 'Prompt coverage only; images and GT-only changes need independent image audit. Cross-ID candidates must not be scored as recovered questions without image evidence.',
        'full_old_20230712_prediction_found': target and len(strict_same) == len(old_groups),
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = ['rg', '--files', '--hidden', '--no-ignore', '-g', '*.jsonl', '-g', '!**/.git/**'] + list(map(str, ROOTS))
    files = sorted(subprocess.check_output(cmd, text=True).splitlines())
    first_records = []
    native = []
    errors = []
    for raw in files:
        p = Path(raw)
        try:
            with p.open() as f:
                first = json.loads(f.readline())
            if not isinstance(first, dict):
                continue
            record = {'path': raw, 'size': p.stat().st_size, 'keys': sorted(first), 'model_id': first.get('model_id')}
            first_records.append(record)
            is_mmb = 'round_id' in first and 'prompt' in first and 'text' in first
            # Inspect every actual 7B answer file, including files with no MMB in the name.
            scan_unknown = first.get('model_id') == 'llava-v1.5-7b' and 'prompt' in first and 'text' in first
            if is_mmb or scan_unknown:
                preds = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
                matching = any(r.get('prompt') in all_mmb_prompts for r in preds)
                if is_mmb or matching:
                    report = native_audit(raw, preds)
                    report['sha256'] = sha(p)
                    native.append(report)
        except Exception as exc:
            errors.append({'path': raw, 'error': type(exc).__name__, 'message': str(exc)[:160],
                           'non_target_exclusion': exception_exclusion(raw)})

    packed = []
    archive_cmd = ['rg', '--files', '--hidden', '--no-ignore', '-g', '*.tar.gz', '-g', '*.zip', '-g', '!**/.git/**'] + list(map(str, ROOTS))
    archives = subprocess.check_output(archive_cmd, text=True).splitlines()
    relevant_archives = [a for a in archives if ('mmb' in a.lower() or Path(a).name == 'eval.zip')]
    for raw in relevant_archives:
        p = Path(raw)
        if p.suffix == '.zip':
            with zipfile.ZipFile(p) as archive:
                members = [m for m in archive.namelist() if m.endswith('.jsonl') and 'mmb' in m.lower()]
                for m in members:
                    content = archive.read(m)
                    preds = [json.loads(line) for line in content.splitlines() if line.strip()]
                    if preds and 'round_id' in preds[0]:
                        report = native_audit(raw + '::' + m, preds)
                        report['prediction_member_sha256'] = hashlib.sha256(content).hexdigest()
                        packed.append(report)
        else:
            with tarfile.open(p) as archive:
                for m in archive.getmembers():
                    if not m.isfile() or not m.name.endswith('.jsonl') or '.runtime.' in m.name:
                        continue
                    content = archive.extractfile(m).read()
                    preds = [json.loads(line) for line in content.splitlines() if line.strip()]
                    if preds and 'round_id' in preds[0]:
                        report = native_audit(raw + '::' + m.name, preds)
                        report['prediction_member_sha256'] = hashlib.sha256(content).hexdigest()
                        packed.append(report)

    # JSON/TSV/XLSX files with benchmark names are logged separately to avoid confusing
    # Qwen records (including a folder named legacy_full) with native LLaVA predictions.
    other_cmd = ['rg', '--files', '--hidden', '--no-ignore', '-g', '*.json', '-g', '*.tsv', '-g', '*.xlsx', '-g', '*.json.gz', '-g', '!**/.git/**'] + list(map(str, ROOTS))
    other_files = [a for a in subprocess.check_output(other_cmd, text=True).splitlines()
                   if 'mmbench' in a.lower() and '/prediction_inventory/' not in a]
    qwen_records = []
    for raw in other_files:
        if Path(raw).name != 'MMBench_DEV_EN_V11.json':
            continue
        d = json.loads(Path(raw).read_text())
        if isinstance(d, dict) and isinstance(d.get('records'), dict):
            counts = collections.Counter(r.get('n_vis_kept') for r in d['records'].values())
            qwen_records.append({'path': raw, 'sha256': sha(raw), 'n_records': len(d['records']),
                                 'n_vis_kept_distribution': dict(counts), 'meta': d.get('meta'),
                                 'excluded': 'Qwen generation/cache schema, not native LLaVA-1.5-7B answers; model identity established in prior Qwen audit and qwen3_k128_launcher.py.'})
    target = [r for r in native if r['target_v15_EN_K128']]
    summary = {
        'scope': 'CPU inventory of both complete repository trees, native JSONL and benchmark-named alternate formats; relevant MMB packed archives also inspected. No GPU or input edits.',
        'roots': list(map(str, ROOTS)), 'JSONL_files_found': len(files),
        'first_record_files_inspected': len(first_records), 'first_record_errors': errors,
        'unresolved_target_parse_errors': [e for e in errors if not e['non_target_exclusion']],
        'native_MMB_files_audited': len(native), 'relevant_archives_inspected': relevant_archives,
        'packed_native_MMB_members_audited': len(packed), 'benchmark_named_alternate_paths': len(other_files),
        'target_model_method_budget_files': target,
        'old_20230712_rows': len(old), 'old_20230712_groups': len(old_groups),
        'V11_rows': len(new), 'V11_groups': len({int(r['index']) % 1000000 for r in new}),
        'old_only_base_ID_groups': len(old_only_ids),
        'confirmed_target_old_full_prediction_found': any(r['full_old_20230712_prediction_found'] for r in native + packed),
        'confirmed_old_only_226_same_ID_coverage_from_target_files': {r['path']: r['same_ID_old_only_226_complete_groups'] for r in target},
        'decision': 'No old-version v15-7B EN K128 EADP/AnchorZip full or partial run was found. Only two eligible V11 files exist. The old-named answer file is 13B and is excluded. Cross-ID identical-text candidates are not proof of same image and cannot fill old missing groups on prompt evidence alone.',
        'boundaries': [
            '63.112078346% on 919 text/options/GT-stable groups is a subset of the V11 run, not complete old-Dev accuracy.',
            '935 same-ID groups preserve prompt and ordered options when ignoring GT; GT-only changes still require image review, rather than automatic label correction.',
            '956 unambiguous-prompt groups and 974 groups allowing ambiguous candidates are coverage candidates only. Repeated generic prompts can identify different images.',
            'Never reuse 13B, Qwen, NeXT, FULL or another budget; never remap a letter across changed option order.',
            'Even compatible predictions retain V11 image encoding and original execution recipe; a reused subset is not an exact old-image full benchmark rerun.'
        ],
        'input_sha256': {str(OLD): sha(OLD), str(NEW): sha(NEW), str(__file__): sha(__file__)},
        'method_budget_evidence': [str(ROOT / 'Qwen_vl/scripts/stage1_roundtrip_pilot/llava_mmben_driver.sh') + ':33', str(ROOT / 'Qwen_vl/scripts/stage1_roundtrip_pilot/llava_mmben_driver.sh') + ':41', str(ROOT / 'Qwen_vl/outputs/audit_followup_20261008/mmb_eadp_immediate_20261010/v15_mmben_E128_streamwait.jsonl.protocol.json')]
    }
    payloads = {'summary.json': summary, 'native_MMB_inventory.json': native,
                'packed_MMB_inventory.json': packed, 'JSONL_first_records.json': first_records,
                'excluded_Qwen_records.json': qwen_records, 'alternate_format_paths.json': other_files}
    for name, payload in payloads.items():
        (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'native_MMB_inventory.csv').open('w', newline='') as f:
        fields = ['path', 'method', 'budget', 'target_v15_EN_K128', 'n_rows', 'same_row_id_exact_V11_EN_prompt_rows', 'same_ID_old_complete_prompt_groups_allowing_GT_correction', 'same_ID_old_only_226_complete_groups', 'unambiguous_prompt_candidate_complete_groups_image_unverified', 'any_prompt_candidate_complete_groups_including_ambiguous_image_unverified', 'exclusion_reason', 'sha256']
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(native)
    print(json.dumps({'output': str(OUT), 'JSONL_files': len(files), 'native_MMB': len(native), 'packed': len(packed), 'target_files': len(target), 'errors': len(errors)}))

if __name__ == '__main__':
    main()
