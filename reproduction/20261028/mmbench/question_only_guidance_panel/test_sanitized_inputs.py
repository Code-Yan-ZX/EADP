#!/usr/bin/env python3
"""Runtime-safe stdlib input tests; never import preparer or open original TSV."""
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

HERE = Path(__file__).resolve().parent
MMB = HERE.parent
GT_TSV = MMB / 'data/mmbench_dev_20230712.tsv'
EXPECTED_MANIFEST = '920a465f64e220e519b41e13664a5959abb5cf55c1b4c5ab2410c539f721ebfb'
EXPECTED_PANEL = '66dc079872510098d12ec4e2fdab79eedd5df462ce6f9cd6ba6b0cd42b14ee83'
GROUPS = [72, 217, 370, 498, 632, 767, 909, 1030, 1143, 1255, 1364, 1480, 1591, 1706, 1821, 1936]
COLUMNS = ['index', 'question', 'hint', 'A', 'B', 'C', 'D', 'image']
digest = lambda payload: hashlib.sha256(payload).hexdigest()
sha = lambda path: digest(Path(path).read_bytes())
load = lambda path: json.loads(Path(path).read_text())
ids_sha = lambda values: digest(b''.join(struct.pack('<q', value) for value in values))


def require_columns(columns):
    assert list(columns) == COLUMNS, 'GT or unexpected runtime column'


def require_panel_ids(rows, source_ids):
    ids = [row['index'] for row in rows]
    assert len(ids) == len(set(ids)) == 60
    assert {idx % 1000000 for idx in ids} == set(GROUPS)
    assert [row['position'] for row in rows] == list(range(60))
    assert [row['original_position'] for row in rows] == sorted(row['original_position'] for row in rows)
    for group in GROUPS:
        assert {idx for idx in ids if idx % 1000000 == group} == {idx for idx in source_ids if idx % 1000000 == group}
    assert Counter(sum(idx % 1000000 == group for idx in ids) for group in GROUPS) == {4: 14, 2: 2}


class SanitizedInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert sha(HERE / 'inputs/manifest.json') == EXPECTED_MANIFEST
        assert sha(HERE / 'inputs/panel.tsv') == EXPECTED_PANEL
        cls.manifest = load(HERE / 'inputs/manifest.json')
        cls.rows = cls.manifest['rows']
        # All these references contain inputs/fingerprints, not GT labels.
        cls.reference = load(MMB / 'cpu_inputs_current.json')
        cls.clips = {row['index']: row for row in cls.reference['guidance']}
        cls.llms = {row['index']: row for row in load(MMB / 'cpu_llm_tokens_current.json')['records']}
        with (HERE / 'inputs/panel.tsv').open(newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            cls.columns = reader.fieldnames; cls.panel = list(reader)

    def test_coverage_and_no_GT_runtime_column(self):
        require_columns(self.columns)
        require_panel_ids(self.rows, list(self.clips))
        self.assertEqual([int(row['index']) for row in self.panel], [row['index'] for row in self.rows])
        self.assertFalse(self.manifest['all_rounds'])
        self.assertFalse(self.manifest['GT_columns_parsed'])
        self.assertTrue(self.manifest['source_TSV_bytes_read_including_GT'])

    def test_reject_duplicate_missing_rotation_and_GT_column(self):
        for bad in (self.rows[:-1], self.rows[:-1] + [self.rows[0]]):
            with self.assertRaises(AssertionError): require_panel_ids(bad, list(self.clips))
        with self.assertRaises(AssertionError): require_panel_ids(self.rows, list(self.clips) + [9000000 + GROUPS[0]])
        with self.assertRaises(AssertionError): require_columns(self.columns + ['answer'])

    def test_native_prompt_guidance_and_historical_binding_metadata(self):
        for row, raw in zip(self.rows, self.panel):
            ref = self.clips[row['index']]
            full = raw['question']
            if raw['hint'] and raw['hint'].lower() not in ('none', 'nan'):
                full = raw['hint'] + '\n' + full
            for letter in 'ABCD':
                value = raw[letter]
                if not value or value.lower() in ('none', 'nan'): break
                full += '\n' + letter + '. ' + value
            self.assertEqual(full, row['full_guidance'])
            self.assertEqual(raw['question'], row['question_only_guidance'])
            self.assertEqual(digest(full.encode()), ref['prompt_sha256'])
            self.assertEqual(row['source_clip']['input_ids_sha256'], ref['input_ids_sha256'])
            self.assertEqual(row['source_clip']['attention_mask_sha256'], ref['attention_mask_sha256'])
            self.assertTrue(row['accepted_source']['text'])
            self.assertEqual(row['accepted_source']['source_runtime_call'], row['original_position'] + 1)
            self.assertEqual(len(row['accepted_source']['raw_line_sha256']), 64)
            self.assertFalse(self.manifest['historical_responses_are_GT'])

    def test_all_int64_arrays_and_LLM_reference(self):
        self.assertEqual(sys.byteorder, 'little')
        for row in self.rows:
            old = self.llms[row['index']]
            self.assertEqual(digest(row['llm_prompt'].encode()), old['prompt_sha256'])
            self.assertEqual(len(row['llm_input_ids']), row['llm_n_tokens'])
            self.assertEqual(row['llm_n_tokens'], old['n_tokens'])
            self.assertEqual(ids_sha(row['llm_input_ids']), row['llm_input_ids_sha256'])
            self.assertEqual(row['llm_input_ids_sha256'], old['token_ids_sha256'])
            for key in ('source_clip', 'question_only_clip'):
                item = row[key]
                for field in ('input_ids', 'attention_mask'):
                    self.assertEqual(len(item[field]), item['n_segments'])
                    self.assertTrue(all(len(segment) == 77 for segment in item[field]))
                    self.assertEqual(ids_sha([value for segment in item[field] for value in segment]), item[field + '_sha256'])
            self.assertEqual(row['question_only_clip']['n_segments'], 1)

    def test_images_and_processed_input_fingerprints(self):
        for row, raw in zip(self.rows, self.panel):
            image = row['image']; reference = self.reference['images'][image['payload_sha256']]
            self.assertEqual(sha(image['path']), image['payload_sha256'])
            self.assertEqual(digest(base64.b64decode(raw['image'], validate=True)), image['payload_sha256'])
            self.assertEqual(image['pixel_float16_sha256'], reference['float16_sha256'])
            self.assertEqual(image['pixel_float32_sha256'], reference['float32_sha256'])
            self.assertEqual(image['decoded_rgb_sha256'], reference['decoded_rgb_sha256'])
            self.assertEqual(image['pixel_shape'], [1, 3, 336, 336])
            self.assertEqual(image['pixel_dtype'], 'torch.float16')

    def test_no_preparer_model_or_torch_import(self):
        self.assertFalse(any('prepare_inputs' in name for name in sys.modules))
        self.assertNotIn('torch', sys.modules)
        self.assertFalse(any(name.startswith('llava') for name in sys.modules))
        self.assertFalse(self.manifest['model_loaded'])
        self.assertFalse(self.manifest['accuracy_computed'])


if __name__ == '__main__':
    # Fail closed if any test accidentally opens the original GT-bearing TSV.
    original_open = Path.open
    attempts = []
    def guarded_open(path, *args, **kwargs):
        if Path(path).resolve() == GT_TSV.resolve():
            attempts.append(str(path)); raise AssertionError('Original GT TSV forbidden in runtime preflight')
        return original_open(path, *args, **kwargs)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SanitizedInputs)
    with patch.object(Path, 'open', guarded_open):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert not attempts
    print(json.dumps(dict(success=result.wasSuccessful(), testsRun=result.testsRun,
                          original_GT_TSV_open_attempts=len(attempts), preparer_imported=False,
                          torch_imported='torch' in sys.modules, files_written=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
