"""Read-only benchmark audit; writes only this independent audit directory.

No CUDA/model imports, model loading, source changes, or native scorer writes.
"""
from pathlib import Path
import ast
import base64
import copy
import collections
import csv
import hashlib
import importlib.util
import io
import json
import re
import os
import string
import sys
import zipfile
from PIL import Image
from pandas._libs.parsers import STR_NA_VALUES

ROOT = Path('/media/disk2/YZX/research/EADP_amp')
OUT = Path(__file__).resolve().parent
EV = ROOT / 'LLaVA/playground/data/eval'
WRAP = ROOT / 'Qwen_vl/scripts/stage1_roundtrip_pilot'
OFF = Path('/media/disk2/YZX/research/audit_base_gap_20261003/eadp_official/LLaVA')
csv.field_size_limit(sys.maxsize)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def jsonl(path):
    return [json.loads(line) for line in Path(path).open()]


def small_tsv(path):
    # Stream the embedded image field instead of retaining ~500MB of payload.
    result = []
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            image = row.pop('image', None)
            row = {k: None if v in STR_NA_VALUES else v for k,v in row.items()}
            row['_image_bytes_sha256'] = None
            row['_image_RGB_sha256'] = None
            if image and len(image) > 100 and int(row['index']) < 1000000:
                data = base64.b64decode(image)
                row['_image_bytes_sha256'] = hashlib.sha256(data).hexdigest()
                rgb = Image.open(io.BytesIO(data)).convert('RGB')
                row['_image_size'] = list(rgb.size)
                row['_image_RGB_sha256'] = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                g = list(rgb.convert('L').resize((9,8)).getdata())
                row['_image_dhash'] = sum(int(g[i*9+j] > g[i*9+j+1]) << (i*8+j) for i in range(8) for j in range(8))
            result.append(row)
    return result


def none(value):
    return value is None or str(value).lower() in ('', 'nan', 'none')


def prompt(row):
    s = row['question']
    if not none(row.get('hint')):
        s = row['hint'] + '\n' + s
    for letter in 'ABCD':
        if none(row.get(letter)):
            break
        s += '\n' + letter + '. ' + row[letter]
    return s


def main():
    report = {'cpu_only': True, 'production_sources_modified': False, 'gpu_started': False}
    conv = load_module('additional_audit_conversation', ROOT / 'LLaVA/llava/conversation.py')
    m4c = load_module('additional_audit_m4c', ROOT / 'LLaVA/llava/eval/m4c_evaluator.py')
    norm = load_module('additional_audit_viz_norm', WRAP / 'vizwiz_official_normalization.py')
    mmben = load_module('additional_audit_mmb_score', WRAP / 'mmben_circular_score.py')
    matching = ROOT / 'Qwen_vl/VLMEvalKit/vlmeval/utils/matching_util.py'
    tree = ast.parse(matching.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('can_infer_option', 'can_infer_text', 'can_infer')]
    assert len(nodes) == 3
    scope = {'os': os, 'cp': copy, 'string': string, 're': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(matching), 'exec'), scope)
    can_infer = scope['can_infer']
    mmb = {}
    for lang in ('EN', 'CN'):
        path = EV / f'mmbench/MMBench_DEV_{lang}_V11.tsv'
        rows = small_tsv(path)
        groups = collections.defaultdict(list)
        for row in rows:
            groups[int(row['index']) % 1000000].append(row)
        mmb[lang] = {'question_file': str(path), 'sha256': sha(path), 'n_rows': len(rows),
                     'n_groups': len(groups), 'group_size_distribution': dict(collections.Counter(map(len, groups.values()))),
                     'unique_indices': len({row['index'] for row in rows}), 'n_base_rows': sum(int(r['index']) < 1000000 for r in rows)}
        structural_errors = []
        for base_id, rotations in groups.items():
            base_row = next(r for r in rotations if int(r['index']) == base_id)
            base_options = [base_row[l] for l in 'ABCD' if not none(base_row.get(l))]
            gt_content = base_row[base_row['answer']]
            if len(rotations) != len(base_options):
                structural_errors.append({'index': base_id, 'error': 'n_rotations_vs_options'})
            for rotation in rotations:
                options = [rotation[l] for l in 'ABCD' if not none(rotation.get(l))]
                if len(options) != len(base_options) or not any(options == base_options[i:] + base_options[:i] for i in range(len(base_options))):
                    structural_errors.append({'index': int(rotation['index']), 'error': 'not_cyclic_permutation'})
                if rotation[rotation['answer']] != gt_content:
                    structural_errors.append({'index': int(rotation['index']), 'error': 'GT_content_changes_across_rotations'})
                if rotation['question'] != base_row['question'] or (rotation.get('hint') or '') != (base_row.get('hint') or ''):
                    structural_errors.append({'index': int(rotation['index']), 'error': 'question_or_hint_changes_across_rotations'})
        mmb[lang]['circular_structural_observations'] = structural_errors
        mmb[lang]['rotation_option_multiset_or_GT_content_errors'] = sum(
            sorted(r[l] for l in 'ABCD' if not none(r.get(l))) != sorted(rs[0][l] for l in 'ABCD' if not none(rs[0].get(l)))
            or r[r['answer']] != rs[0][rs[0]['answer']]
            for rs in groups.values() for r in rs)
        mmb[lang]['rotation_observation_interpretation'] = '28 official V11 groups retain D fixed and rotate ABC only; this is present in input TSV, not missing generated rows or incorrect GT mapping.'
        for task in (f'mmb{lang.lower()}', f'next_mmb{lang.lower()}'):
            for pred in sorted((EV / 'anchorzip_p3' / task).glob('*.jsonl')):
                ps = jsonl(pred)
                if len(ps) != len(rows):
                    continue
                assert [str(p['question_id']) for p in ps] == [r['index'] for r in rows], pred
                regex_hit, toolkit_hit, diffs = {}, {}, []
                for p, r in zip(ps, rows):
                    choices = {l: r[l] for l in 'ABCD' if not none(r.get(l))}
                    regex = mmben.extract_letter(p['text'])
                    toolkit = can_infer(p['text'], choices)
                    regex_hit[r['index']] = regex == r['answer']
                    toolkit_hit[r['index']] = toolkit == r['answer']
                    if (regex or None) != (toolkit or None):
                        diffs.append({'question_id': int(r['index']), 'text': p['text'], 'GT': r['answer'], 'local_regex': regex, 'toolkit_no_judge': toolkit})
                mmb[lang].setdefault('completed_predictions', []).append({
                    'path': str(pred), 'sha256': sha(pred), 'n': len(ps),
                    'n_prompt_mismatches': sum(p['prompt'] != prompt(r) for p, r in zip(ps, rows)),
                    'round_id_distribution': dict(collections.Counter(str(p.get('round_id')) for p in ps)),
                    'n_multiple_distinct_capital_options': sum(len(set(re.findall(r'\b([A-D])\b', p['text']))) > 1 for p in ps),
                    'n_regex_parse_failures': sum(mmben.extract_letter(p['text']) is None for p in ps),
                    'n_parser_differences_vs_toolkit_no_judge': len(diffs), 'parser_difference_examples': diffs[:8],
                    'local_regex_circular_acc_percent': 100 * sum(all(regex_hit[r['index']] for r in rs) for rs in groups.values()) / len(groups),
                    'toolkit_no_judge_circular_acc_percent': 100 * sum(all(toolkit_hit[r['index']] for r in rs) for rs in groups.values()) / len(groups)})
        if lang == 'CN':
            changed = 0
            identities = []
            for i, row in enumerate(rows):
                content = prompt(row)
                prompts = {}
                for code, instruction in [('en', "Answer with the option's letter from the given choices directly."),
                                          ('cn', '请直接回答选项字母。')]:
                    conversation = conv.conv_templates['vicuna_v1'].copy()
                    conversation.append_message(conversation.roles[0], '<image>\n' + content + '\n' + instruction)
                    conversation.append_message(conversation.roles[1], None)
                    prompts[code] = conversation.get_prompt()
                changed += prompts['en'] != prompts['cn']
                identities.append({'position': i, 'question_id': int(row['index']),
                                   'guidance_sha256': hashlib.sha256(content.encode()).hexdigest(),
                                   'llm_prompt_en_sha256': hashlib.sha256(prompts['en'].encode()).hexdigest(),
                                   'llm_prompt_cn_sha256': hashlib.sha256(prompts['cn'].encode()).hexdigest()})
            assert changed == len(rows)
            mmb[lang].update(n_llm_prompt_changes_if_lang_cn=changed, n_guidance_changes_if_lang_cn=0,
                            native_prediction_prompt_field_excludes_language_instruction=True)
            (OUT / 'mmbcn_prompt_identity.json').write_text(json.dumps(identities, indent=1, ensure_ascii=False) + '\n')
        if lang == 'EN':
            oldpath = EV / 'mmbench/mmbench_dev_20230712.tsv'
            oldrows = small_tsv(oldpath)
            oldbase = [r for r in oldrows if int(r['index']) < 1000000]
            oldmap = {str(int(r['index'])): r for r in oldbase}
            base = [r for r in rows if int(r['index']) < 1000000]
            shared = [(r, oldmap[r['index']]) for r in base if r['index'] in oldmap]
            fields = ['question', 'hint', 'A', 'B', 'C', 'D', 'answer']
            def semantic_key(r, include_image=True):
                s = (r['question'], r.get('hint') or '', tuple(sorted(r[l] for l in 'ABCD' if not none(r.get(l)))), r[r['answer']])
                return s + (r['_image_RGB_sha256'],) if include_image else s
            oldsemantic, newsemantic = {semantic_key(r) for r in oldbase}, {semantic_key(r) for r in base}
            oldimages, newimages = {r['_image_bytes_sha256'] for r in oldbase}, {r['_image_bytes_sha256'] for r in base}
            mmb[lang]['legacy_split_comparison'] = {
                'path': str(oldpath), 'sha256': sha(oldpath), 'n_rows': len(oldrows),
                'n_legacy_base_groups': len(oldbase), 'n_v11_base_groups': len(base),
                'n_shared_base_indices': len(shared), 'v11_base_not_in_legacy': len(base) - len(shared),
                'legacy_not_in_v11_base': len(set(oldmap) - {r['index'] for r in base}),
                'changed_fields_counts': {k: sum((a.get(k) or '') != (b.get(k) or '') for a,b in shared) for k in fields},
                'n_changed_content_or_gt': sum(any((a.get(k) or '') != (b.get(k) or '') for k in fields) for a,b in shared),
                'shared_base_id_same_image_bytes': sum(a['_image_bytes_sha256'] == b['_image_bytes_sha256'] for a,b in shared),
                'shared_base_id_same_RGB_pixels_and_dimensions': sum(a['_image_RGB_sha256'] == b['_image_RGB_sha256'] for a,b in shared),
                'shared_base_id_same_dimensions': sum(a['_image_size'] == b['_image_size'] for a,b in shared),
                'shared_base_id_dhash_hamming_distribution': dict(collections.Counter((a['_image_dhash'] ^ b['_image_dhash']).bit_count() for a,b in shared)),
                'legacy_base_dimensions_distribution': dict(collections.Counter(str(r['_image_size']) for r in oldbase)),
                'v11_base_dimensions_distribution': dict(collections.Counter(str(r['_image_size']) for r in base)),
                'shared_base_id_exact_question_options_GT_image': sum(all((a.get(k) or '') == (b.get(k) or '') for k in fields) and a['_image_bytes_sha256'] == b['_image_bytes_sha256'] for a,b in shared),
                'n_shared_question_hint_unordered_choices_answer_text_RGB_exact_unique_keys': len(oldsemantic & newsemantic),
                'n_shared_question_hint_unordered_choices_answer_text_without_image': len({semantic_key(r,False) for r in oldbase} & {semantic_key(r,False) for r in base}),
                'n_unique_images_legacy': len(oldimages), 'n_unique_images_v11': len(newimages), 'n_shared_image_bytes': len(oldimages & newimages)}
    report['mmbench'] = mmb

    gtpath = EV / 'vizwiz/val.json'
    gt = json.load(gtpath.open())
    qs = jsonl(EV / 'vizwiz/llava_val.jsonl')
    suffix = "\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase."
    assert len(gt) == len(qs) == 4319
    assert all(q['text'] == g['question'] + suffix and q['image'] == g['image'] for q,g in zip(qs,gt))
    viz = {'n': len(gt), 'official_conversion_exact_all_questions': True,
           'qfile': str(EV / 'vizwiz/llava_val.jsonl'), 'gt_file': str(gtpath),
           'n_guidance_changes_loader_vs_model_vqa': sum(q['text'].replace('\nAnswer the question using a single word or phrase.', '') != q['text'] for q in qs)}
    class Stub:
        def getImgs(self):
            return []
    normalizer = norm.VizWizNormalizer(Stub(), Stub())
    processor = m4c.EvalAIAnswerProcessor()
    for task in ('vizwiz', 'next_vizwiz'):
        for pred in sorted((EV / 'anchorzip_p3' / task).glob('*.jsonl')):
            ps = jsonl(pred)
            assert len(ps) == len(gt) and [int(p['question_id']) for p in ps] == list(range(len(gt)))
            author, author_norm_loo, loo, n_formula_changed = [], [], [], 0
            for g, p in zip(gt,ps):
                pn = processor(p['text'])
                answers_norm = [processor(a['answer']) for a in g['answers']]
                author.append(min(1.0, answers_norm.count(pn) / 3))
                author_norm_loo.append(sum(min(1.0, sum(a == pn for j,a in enumerate(answers_norm) if i != j) / 3) for i in range(len(answers_norm))) / len(answers_norm))
                res = normalizer.processDigitArticle(normalizer.processPunctuation(p['text'].replace('\n', ' ').replace('\t', ' ').strip()))
                answers = [a['answer'] for a in g['answers']]
                loo.append(sum(min(1.0, sum(a == res for j,a in enumerate(answers) if i != j) / 3) for i in range(len(answers))) / len(answers))
                n_formula_changed += abs(author[-1] - loo[-1]) > 1e-12
            viz.setdefault('complete_prediction_scores', []).append({
                'path': str(pred), 'prediction_sha256': sha(pred), 'n': len(ps),
                'EADP_published_code_vizwiz_score_percent': 100 * sum(author) / len(author),
                'benchmark_official_leave_one_out_percent': 100 * sum(loo) / len(loo),
                'author_code_minus_official_pp': 100 * (sum(author) - sum(loo)) / len(loo),
                'leave_one_out_change_only_using_author_normalization_percent': 100 * sum(author_norm_loo) / len(author_norm_loo),
                'author_naive_minus_author_norm_leave_one_out_pp': 100 * (sum(author) - sum(author_norm_loo)) / len(loo),
                'author_norm_leave_one_out_minus_official_norm_leave_one_out_pp': 100 * (sum(author_norm_loo) - sum(loo)) / len(loo),
                'n_changed_per_question_scores': n_formula_changed,
                'interpretation': 'Both evaluated on identical raw answers; published-code result is auxiliary, not replacement official score.'})
    report['vizwiz'] = viz

    zip_path = EV / 'eval.zip'
    with zipfile.ZipFile(zip_path) as z:
        for key, suffix in [('pope', '/pope/llava_pope_test.jsonl'), ('scienceqa', '/scienceqa/llava_test_CQM-A.json')]:
            matches = [p for p in z.namelist() if p.endswith(suffix.lstrip('/'))]
            assert len(matches) == 1, matches
            data = z.read(matches[0])
            file = EV / suffix.removeprefix('/')
            report[key] = {'official_LLaVA_eval_zip': str(zip_path), 'zip_member': matches[0],
                           'zip_member_sha256': hashlib.sha256(data).hexdigest(),
                           'on_disk_path': str(file), 'on_disk_sha256': sha(file),
                           'exact_member_bytes_match': data == file.read_bytes()}
            if key == 'pope':
                q = [json.loads(line) for line in data.splitlines()]
                report[key]['n_by_category'] = dict(collections.Counter(r['category'] for r in q))
                report[key]['n'] = len(q)
            else:
                sq = json.loads(data)
                report[key].update(n_total=len(sq), n_image=sum('image' in r for r in sq))
                subset = json.load((ROOT / 'Qwen_vl/outputs/audit_followup_20261008/sqa_CQMA_image2017.json').open())
                report[key]['current_2017_subset_exact'] = subset == [r for r in sq if 'image' in r]
        for key, member in [('textvqa', 'textvqa/llava_textvqa_val_v051_ocr.jsonl'), ('gqa', 'gqa/llava_gqa_testdev_balanced.jsonl'),
                            ('mme', 'MME/llava_mme.jsonl'), ('mmvet', 'mm-vet/llava-mm-vet.jsonl')]:
            data = z.read(member)
            file = EV / member
            report[key] = {'zip_member': member, 'zip_member_sha256': hashlib.sha256(data).hexdigest(),
                           'on_disk_sha256': sha(file), 'exact_member_bytes_match': data == file.read_bytes(),
                           'n': len(data.splitlines())}
        gqa_gt_path = Path('/tmp/gqa12/testdev_balanced_questions.json')
        gqa_gt = json.load(gqa_gt_path.open())
        gqa_q = jsonl(EV / 'gqa/llava_gqa_testdev_balanced.jsonl')
        report['gqa'].update(official_GT_file=str(gqa_gt_path), official_GT_sha256=sha(gqa_gt_path),
                             n_key_mismatch=len({str(q['question_id']) for q in gqa_q} ^ set(gqa_gt)),
                             n_prompt_question_mismatch=sum(q['text'].replace('\nAnswer the question using a single word or phrase.', '') != gqa_gt[str(q['question_id'])]['question'] for q in gqa_q),
                             n_image_filename_mismatch=sum(Path(q['image']).stem != str(gqa_gt[str(q['question_id'])]['imageId']) for q in gqa_q))

    checks = [report['pope']['exact_member_bytes_match'], report['scienceqa']['exact_member_bytes_match'],
              report['scienceqa']['current_2017_subset_exact'], viz['official_conversion_exact_all_questions'],
              mmb['CN']['n_llm_prompt_changes_if_lang_cn'] == 4876,
              all(p['n_prompt_mismatches'] == 0 for v in mmb.values() for p in v['completed_predictions']),
              all(v['rotation_option_multiset_or_GT_content_errors'] == 0 for v in mmb.values()),
              all(report[k]['exact_member_bytes_match'] for k in ('textvqa','gqa','mme','mmvet')),
              all(report['gqa'][k] == 0 for k in ('n_key_mismatch','n_prompt_question_mismatch','n_image_filename_mismatch'))]
    report['all_cpu_identity_checks_pass'] = all(checks)
    report['n_cpu_identity_checks'] = len(checks)
    assert all(checks)
    (OUT / 'cpu_protocol_identity_and_scores.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'pass': all(checks), 'n_checks':len(checks), 'output': str(OUT / 'cpu_protocol_identity_and_scores.json'),
                      'n_complete_viz_scores':len(viz['complete_prediction_scores']),
                      'n_complete_mmb_scores':sum(len(v['completed_predictions']) for v in mmb.values())}))


if __name__ == '__main__':
    main()
