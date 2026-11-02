#!/usr/bin/env python3
"""Preparation-side tests; hashes original GT-bearing TSV bytes, parses no GT.

Not a runtime/container preflight entry. Use test_sanitized_inputs.py there.
No model/tokenizer/processor invocation occurs in either test script.
"""
import base64
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('MMB_panel_inputs', HERE / 'prepare_inputs.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class PreparedInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = prep.read(HERE / 'inputs/manifest.json')
        cls.rows = cls.manifest['rows']
        cls.clip = prep.read(prep.MMB / 'cpu_inputs_current.json')
        cls.clips = {row['index']: row for row in cls.clip['guidance']}
        cls.llms = {row['index']: row for row in prep.read(prep.MMB / 'cpu_llm_tokens_current.json')['records']}
        cls.images = {row['index']: row for row in prep.read(prep.MMB / 'image_manifest.json')}
        cls.predictions = [json.loads(line) for line in (prep.OLD / 'v15_EADP_K128_oldEN_beta1_alpha05_streamwait.jsonl').read_text().splitlines()]
        cls.runtime = [json.loads(line) for line in (prep.OLD / 'observed/runtime.jsonl').read_text().splitlines()]
        with (HERE / 'inputs/panel.tsv').open(newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            cls.columns = reader.fieldnames
            cls.panel = list(reader)

    def anchor(self, record):
        index, position = record['index'], record['original_position']
        prep.validate_anchor(record, self.predictions[position], self.runtime[position],
                             self.clips[index], self.llms[index], self.images[index])

    def test_60_native_anchors_and_complete_rotations(self):
        prep.validate_panel_structure(self.rows, list(self.clips))
        for record in self.rows:
            self.anchor(record)
        self.assertEqual(self.manifest['n_groups'], 16)
        self.assertEqual(self.manifest['n_rows'], 60)
        self.assertFalse(self.manifest['all_rounds'])

    def test_sanitized_runtime_input_has_no_GT_column(self):
        prep.validate_columns(self.columns)
        self.assertEqual(len(self.panel), 60)
        self.assertEqual([int(row['index']) for row in self.panel], [row['index'] for row in self.rows])
        self.assertNotIn('answer', self.columns)
        self.assertTrue(self.manifest['source_TSV_bytes_read_including_GT'])
        self.assertFalse(self.manifest['GT_columns_parsed'])
        self.assertTrue(self.manifest['contains_historical_model_responses'])
        self.assertFalse(self.manifest['historical_responses_are_GT'])
        with self.assertRaises(AssertionError):
            prep.validate_columns(self.columns + ['answer'])

    def test_pixels_payloads_and_all_saved_hash_bindings(self):
        manifest = self.manifest
        self.assertEqual(prep.sha(manifest['panel_tsv']['path']), manifest['panel_tsv']['sha256'])
        for reference in manifest['sources'].values():
            self.assertEqual(prep.sha(reference['path']), reference['sha256'])
        for row, panel in zip(self.rows, self.panel):
            image = row['image']
            expected = self.clip['images'][image['payload_sha256']]
            self.assertEqual(prep.sha(image['path']), image['payload_sha256'])
            self.assertEqual(prep.digest_bytes(base64.b64decode(panel['image'], validate=True)), image['payload_sha256'])
            self.assertEqual(image['pixel_float16_sha256'], expected['float16_sha256'])
            self.assertEqual(image['pixel_float32_sha256'], expected['float32_sha256'])
            self.assertEqual(image['decoded_rgb_sha256'], expected['decoded_rgb_sha256'])
            self.assertEqual(image['pixel_shape'], [1, 3, 336, 336])
            self.assertEqual(image['pixel_dtype'], 'torch.float16')

    def test_native_token_array_dimensions_without_torch(self):
        # int64 byte checks use the documented little-endian native platform.
        import struct
        self.assertEqual(sys.byteorder, 'little')
        fingerprint = lambda values: hashlib.sha256(b''.join(struct.pack('<q', value) for value in values)).hexdigest()
        for row in self.rows:
            self.assertEqual(len(row['llm_input_ids']), row['llm_n_tokens'])
            self.assertTrue(all(type(value) is int for value in row['llm_input_ids']))
            self.assertEqual(fingerprint(row['llm_input_ids']), row['llm_input_ids_sha256'])
            for key in ('source_clip', 'question_only_clip'):
                clip = row[key]
                for array, digest in [('input_ids', 'input_ids_sha256'), ('attention_mask', 'attention_mask_sha256')]:
                    self.assertEqual(len(clip[array]), clip['n_segments'])
                    self.assertTrue(all(len(segment) == 77 for segment in clip[array]))
                    self.assertEqual(fingerprint([value for segment in clip[array] for value in segment]), clip[digest])
            self.assertEqual(row['question_only_clip']['n_segments'], 1)

    def test_duplicate_and_missing_rotation_are_rejected(self):
        for bad in [self.rows[:-1], self.rows[:-1] + [self.rows[0]]]:
            with self.assertRaises(AssertionError):
                prep.validate_panel_structure(bad, list(self.clips))
        missing_universe = list(self.clips) + [self.rows[0]['group_id'] + 9000000]
        with self.assertRaises(AssertionError):
            prep.validate_panel_structure(self.rows, missing_universe)

    def test_changed_guidance_LLM_and_old_text_are_rejected(self):
        for key in ('full_guidance', 'llm_input_ids_sha256'):
            row = copy.deepcopy(self.rows[0]); row[key] += 'changed'
            with self.assertRaises(AssertionError): self.anchor(row)
        row = copy.deepcopy(self.rows[0]); row['accepted_source']['text'] += 'changed'
        with self.assertRaises(AssertionError): self.anchor(row)

    def test_no_model_score_GPU_or_GT_claim(self):
        self.assertFalse(self.manifest['model_loaded'])
        self.assertFalse(self.manifest['CUDA_initialized'])
        self.assertFalse(self.manifest['accuracy_computed'])
        self.assertNotIn('torch', sys.modules)
        self.assertFalse(any(name.startswith('llava') for name in sys.modules))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PreparedInputs)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(success=result.wasSuccessful(), testsRun=result.testsRun,
                  errors=len(result.errors), failures=len(result.failures),
                  test=prep.artifact(__file__), preparer=prep.artifact(HERE / 'prepare_inputs.py'),
                  manifest=prep.artifact(HERE / 'inputs/manifest.json'),
                  panel_tsv=prep.artifact(HERE / 'inputs/panel.tsv'),
                  model_or_score_called=False, torch_imported='torch' in sys.modules,
                  preparation_phase_only=True, original_TSV_GT_bytes_hashed=True,
                  GT_columns_parsed=False, runtime_preflight_entry=False)
    (HERE / 'input_CPU_tests.json').write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
