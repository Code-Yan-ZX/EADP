#!/usr/bin/env python3
"""Stdlib runtime-input tests: no preparer import, GT TSV, model, or scorer."""
import argparse
import base64
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

csv.field_size_limit(16 * 1024 * 1024)

HERE = Path(__file__).resolve().parent
MMB = HERE.parent
GT_TSV = MMB / 'data/mmbench_dev_20230712.tsv'
FIXED_MANIFEST = MMB / 'question_only_guidance_panel/inputs/manifest.json'
COLUMNS = ['index', 'question', 'hint', 'A', 'B', 'C', 'D', 'image']
MANIFEST_SHA = '4b2257d2e6078019a7105e940b11413ebfdbfca993599119c10761611072447b'
PANEL_SHA = '9a3d1804ea68c9810ff4d3afd88de3449d7dfe5715c8d1509b66c816440a4062'
FIXED_SHA = '920a465f64e220e519b41e13664a5959abb5cf55c1b4c5ab2410c539f721ebfb'
CLIP_SHA = 'b35a2b60bababb1c99884b248c90e64a409bc8c7ef72b5c83f6bf8faf205108e'
LLM_SHA = '05f7245c2b248a85f0fa56b6d3a298d1b6ed038b22a0661b80cb6832158be80c'


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def ids_sha(values):
    return digest(b''.join(struct.pack('<q', value) for value in values))


def canonical_sha(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())


def require_columns(columns):
    assert list(columns) == COLUMNS, 'GT or unexpected runtime input column'


def require_ids(rows, reference_ids):
    ids = [int(row['index']) for row in rows]
    assert len(ids) == len(set(ids)) == 4377
    assert ids == reference_ids, 'Canonical row order changed'
    assert [row['position'] for row in rows] == [row['original_position'] for row in rows] == list(range(4377))
    groups = Counter(idx % 1000000 for idx in ids)
    assert len(groups) == 1176 and Counter(groups.values()) == {4: 938, 3: 149, 2: 89}
    assert all(row['group_id'] == row['index'] % 1000000 and row['round_id'] == 0 for row in rows)


def require_fixed(rows, fixed, reference):
    assert reference['rows'] == fixed['rows'] and len(reference['links']) == 60
    assert reference['manifest']['sha256'] == FIXED_SHA
    assert reference['only_normalized_fields'] == ['position', 'image.path']
    for old, link in zip(fixed['rows'], reference['links']):
        assert link['panel_position'] == old['position'] and link['full_position'] == old['original_position']
        assert link['index'] == old['index']
        new = copy.deepcopy(rows[old['original_position']])
        new['position'] = old['position']; new['image']['path'] = old['image']['path']
        assert new == old, 'Fixed60 row changed beyond declared path/order metadata'
        assert canonical_sha(old) == link['original_row_sha256'] == link['normalized_full_row_sha256']


class FullInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert sha(HERE / 'inputs/manifest.json') == MANIFEST_SHA
        assert sha(HERE / 'inputs/panel.tsv') == PANEL_SHA
        assert sha(FIXED_MANIFEST) == FIXED_SHA
        assert sha(MMB / 'cpu_inputs_current.json') == CLIP_SHA
        assert sha(MMB / 'cpu_llm_tokens_current.json') == LLM_SHA
        cls.manifest = load(HERE / 'inputs/manifest.json')
        cls.rows = cls.manifest['rows']
        cls.fixed = load(FIXED_MANIFEST)
        cls.reference = load(MMB / 'cpu_inputs_current.json')
        cls.clips = {row['index']: row for row in cls.reference['guidance']}
        cls.llms = {row['index']: row for row in load(MMB / 'cpu_llm_tokens_current.json')['records']}

    def test_full_canonical_order_and_complete_rotations(self):
        require_ids(self.rows, list(self.clips))
        self.assertEqual(self.manifest['n_rows'], 4377)
        self.assertEqual(self.manifest['n_groups'], 1176)
        self.assertEqual(self.manifest['group_ids'], sorted({row['group_id'] for row in self.rows}))
        self.assertFalse(self.manifest['all_rounds'])

    def test_fixed60_original_records_and_mapping(self):
        require_fixed(self.rows, self.fixed, self.manifest['fixed60_reference'])

    def test_GT_source_is_documentary_and_forbidden_in_runtime_guards(self):
        manifest = self.manifest
        self.assertEqual(manifest['columns_parsed'], COLUMNS)
        self.assertTrue(manifest['source_TSV_bytes_read_including_GT'])
        self.assertTrue(manifest['source_TSV_contains_GT_bytes'])
        self.assertFalse(manifest['GT_columns_parsed'])
        self.assertTrue(manifest['contains_historical_model_responses'])
        self.assertFalse(manifest['historical_responses_are_GT'])
        for item in manifest['sources'].values():
            self.assertNotEqual(Path(item['path']).resolve(), GT_TSV.resolve())
        self.assertEqual(len(manifest['provenance_only_sources']), 1)
        source = manifest['provenance_only_sources'][0]
        self.assertEqual(Path(source['path']).resolve(), GT_TSV.resolve())
        self.assertTrue(source['contains_GT'] and source['runtime_access_forbidden'])
        self.assertFalse(source['GT_columns_parsed'])

    def test_all_CLIP_and_LLM_arrays_without_model_import(self):
        self.assertEqual(sys.byteorder, 'little')
        long_counts = Counter()
        for row in self.rows:
            clip, llm = self.clips[row['index']], self.llms[row['index']]
            self.assertEqual(digest(row['full_guidance'].encode()), clip['prompt_sha256'])
            self.assertEqual(digest(row['llm_prompt'].encode()), llm['prompt_sha256'])
            self.assertEqual(row['llm_n_tokens'], llm['n_tokens'])
            self.assertEqual(len(row['llm_input_ids']), row['llm_n_tokens'])
            self.assertEqual(ids_sha(row['llm_input_ids']), row['llm_input_ids_sha256'])
            self.assertEqual(row['llm_input_ids_sha256'], llm['token_ids_sha256'])
            for key in ('source_clip', 'question_only_clip'):
                item = row[key]
                long_counts[key] += item['n_tokens'] > 77
                for field in ('input_ids', 'attention_mask'):
                    self.assertEqual(len(item[field]), item['n_segments'])
                    self.assertTrue(all(len(segment) == 77 for segment in item[field]))
                    self.assertEqual(ids_sha([value for segment in item[field] for value in segment]), item[field + '_sha256'])
            self.assertEqual(row['source_clip']['input_ids_sha256'], clip['input_ids_sha256'])
            self.assertEqual(row['source_clip']['attention_mask_sha256'], clip['attention_mask_sha256'])
            self.assertEqual(digest(row['question_only_guidance'].encode()), row['question_only_clip']['text_sha256'])
        self.assertEqual(long_counts, {'source_clip': 703, 'question_only_clip': 4})

    def test_sanitized_TSV_roundtrip_images_prompts_and_pixels(self):
        cached = set()
        with (HERE / 'inputs/panel.tsv').open(newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t'); require_columns(reader.fieldnames)
            count = 0
            for row in self.rows:
                raw = next(reader)
                self.assertEqual(int(raw['index']), row['index'])
                full = raw['question']
                if raw['hint'] and raw['hint'].lower() not in ('none', 'nan'): full = raw['hint'] + '\n' + full
                for letter in 'ABCD':
                    value = raw[letter]
                    if not value or value.lower() in ('none', 'nan'): break
                    full += '\n' + letter + '. ' + value
                self.assertEqual(full, row['full_guidance'])
                self.assertEqual(raw['question'], row['question_only_guidance'])
                image = row['image']; reference = self.reference['images'][image['payload_sha256']]
                self.assertEqual(digest(base64.b64decode(raw['image'])), image['payload_sha256'])
                if image['payload_sha256'] not in cached:
                    self.assertEqual(sha(image['path']), image['payload_sha256']); cached.add(image['payload_sha256'])
                self.assertEqual(image['pixel_float16_sha256'], reference['float16_sha256'])
                self.assertEqual(image['pixel_float32_sha256'], reference['float32_sha256'])
                self.assertEqual(image['decoded_rgb_sha256'], reference['decoded_rgb_sha256'])
                self.assertEqual(image['pixel_shape'], [1, 3, 336, 336])
                self.assertEqual(image['pixel_dtype'], 'torch.float16')
                count += 1
            self.assertEqual(count, 4377)
            self.assertIsNone(next(reader, None), 'Extra TSV row')
        self.assertEqual(len(cached), 1132)

    def test_reject_order_duplicate_GT_column_and_fixed_anchor_mutation(self):
        reference = list(self.clips)
        for bad in (self.rows[:-1], self.rows[:-1] + [self.rows[0]], [self.rows[1], self.rows[0]] + self.rows[2:]):
            with self.assertRaises(AssertionError): require_ids(bad, reference)
        with self.assertRaises(AssertionError): require_columns(COLUMNS + ['answer'])
        fixed = copy.deepcopy(self.fixed); fixed['rows'][0]['llm_input_ids'][0] += 1
        with self.assertRaises(AssertionError): require_fixed(self.rows, fixed, self.manifest['fixed60_reference'])

    def test_no_preparer_neural_model_or_torch_import(self):
        self.assertFalse(any('prepare_inputs' in name for name in sys.modules))
        self.assertNotIn('torch', sys.modules)
        self.assertFalse(any(name.startswith('llava') for name in sys.modules))
        self.assertFalse(self.manifest['model_loaded'])
        self.assertFalse(self.manifest['CUDA_initialized'])
        self.assertFalse(self.manifest['accuracy_computed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report', action='store_true', help='Preparation stage only; default tests write no files')
    args = parser.parse_args()
    original_open, forbidden_attempts = Path.open, []
    def guarded_open(path, *positional, **keywords):
        if Path(path).resolve() == GT_TSV.resolve():
            forbidden_attempts.append(str(path)); raise AssertionError('GT-bearing original TSV forbidden during runtime input tests')
        return original_open(path, *positional, **keywords)
    with patch.object(Path, 'open', guarded_open):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(FullInputs)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert not forbidden_attempts
    report = dict(success=result.wasSuccessful(), testsRun=result.testsRun, errors=len(result.errors), failures=len(result.failures),
                  n_rows=4377, n_groups=1176, fixed60_anchors=60, unique_images=1132,
                  manifest_sha256=MANIFEST_SHA, panel_tsv_sha256=PANEL_SHA, test_sha256=sha(__file__),
                  original_GT_TSV_open_attempts=0, preparer_imported=False, torch_imported='torch' in sys.modules,
                  model_or_score_called=False, default_tests_write_files=False)
    if args.write_report:
        with (HERE / 'input_CPU_tests.json').open('x') as stream:
            json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
