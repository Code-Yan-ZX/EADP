"""CPU checks for benchmark identities, input pixels, and overlap accounting."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from . import data


class TransferDataTests(unittest.TestCase):
    def image_bytes(self, color, fmt='PNG'):
        buffer = io.BytesIO()
        Image.new('RGB', (21, 17), color).save(buffer, format=fmt)
        return buffer.getvalue()

    def make_training(self, root):
        run = root / 'source_run'
        (run / 'data').mkdir(parents=True)
        splits = {}
        for split, color in [('fit', (22, 90, 131)), ('val', (131, 22, 90))]:
            raw = self.image_bytes(color)
            path = run / 'data' / (split + '.png')
            path.write_bytes(raw)
            with Image.open(io.BytesIO(raw)) as image:
                pixels = data.pixel_sha256(image)
            splits[split] = [dict(page_id=split + '_1.png', parent_id=split,
                                 image_path=str(path), image_sha256=hashlib.sha256(raw).hexdigest(),
                                 pixel_sha256=pixels, questions=[dict(question='a?', answers=['b'])])]
        (run / 'data/manifest.json').write_text(json.dumps(dict(splits=splits)))
        return run

    def write_dataset(self, root, name, rows):
        columns = ['index', 'image', 'question', 'answer', 'image_path', 'ucsf_document_id', 'category']
        with (root / (name + '.tsv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)

    def test_materialization_matches_default_pil_jpeg_not_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = self.image_bytes((22, 90, 131), 'JPEG')
            result = data.materialize_image(raw, '.jpg', Path(temporary))
            expected = io.BytesIO()
            with Image.open(io.BytesIO(raw)) as image:
                image.save(expected, format='JPEG')
            self.assertEqual(Path(result['image_path']).read_bytes(), expected.getvalue())
            self.assertEqual(result['source_image_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(result['image_sha256'], hashlib.sha256(expected.getvalue()).hexdigest())

    def test_reference_rows_retained_and_exact_overlap_audited(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_training(root)
            raw = (run / 'data/fit.png').read_bytes()
            rows = [dict(index='10', image=base64.b64encode(raw).decode(), question='first?',
                         answer="['one', '1']", image_path='a.png', ucsf_document_id='fit', category=''),
                    dict(index='11', image='10', question='second?', answer="['two']",
                         image_path='a.png', ucsf_document_id='fit', category='')]
            self.write_dataset(root, 'DocVQA_VAL', rows)
            with patch.dict(data.EXPECTED_COUNTS, DocVQA_VAL=2):
                manifest = data.build_manifest(root / 'out', run, ('DocVQA_VAL',), source_root=root,
                                               history_root=root / 'missing_history')
                again = data.build_manifest(root / 'out', run, ('DocVQA_VAL',), source_root=root,
                                            history_root=root / 'missing_history')
            dataset = manifest['datasets']['DocVQA_VAL']
            self.assertEqual(manifest, again)
            self.assertEqual(dataset['num_items'], 2)
            self.assertEqual(dataset['num_unique_images'], 1)
            self.assertEqual(dataset['training_overlap_count'], 2)
            self.assertEqual(dataset['decontaminated_item_ids'], [])
            self.assertEqual(dataset['items'][1]['image_source_index'], '10')
            self.assertEqual(dataset['items'][1]['question'], 'second?')
            self.assertEqual(dataset['items'][0]['answers'], ['one', '1'])
            Path(dataset['items'][0]['image_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'benchmark input changed'):
                data.verify_manifest(manifest)

    def test_forward_image_reference_and_cycles(self):
        direct = dict(image_path='test', image_sha256='abc')
        refs = {'one': {'ref': 'two'}, 'two': direct}
        self.assertIs(data._resolve_metadata(refs, 'one'), direct)
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            data._resolve_metadata({'one': {'ref': 'two'}, 'two': {'ref': 'one'}}, 'one')
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            data._resolve_metadata({'one': {'ref': 'missing'}}, 'one')

    def test_cross_dataset_pixel_overlap_does_not_require_doc_id(self):
        item = dict(dataset='OCRBench', parent_id='unrelated', pixel_sha256='saved',
                    source_pixel_sha256='source', image_sha256='saved_file', source_image_sha256='source_file',
                    image_size=[200, 100], dhash64='00000000000000ff')
        refs = [dict(split='fit', parent_id='parent', page_id='page', pixel_sha256='source',
                     image_sha256='source_file', image_size=[400, 200], dhash64='00000000000000fe')]
        data._mark_overlap(item, refs)
        self.assertTrue(item['training_overlap'])
        self.assertTrue(item['near_duplicate_suspected'])
        self.assertEqual(item['training_overlap_reasons'][0]['matches'], ['rgb_pixels', 'image_bytes'])
        self.assertEqual(item['near_duplicate_matches'][0]['dhash_hamming'], 1)

    def test_answers_preserve_human_vote_multiplicity(self):
        self.assertEqual(data.parse_answers("['yes','yes','no']", 'TextVQA_VAL', '1'), ['yes', 'yes', 'no'])
        with self.assertRaises(ValueError):
            data.parse_answers('[]', 'TextVQA_VAL', '1')

    def test_history_marks_prior_indices_without_filtering(self):
        from openpyxl import Workbook
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = root / 'eadp/Qwen3-VL-8B-EADP-256-a0.5-b2.0'
            prefix.mkdir(parents=True)
            path = prefix / 'Qwen3-VL-8B-EADP-256-a0.5-b2.0_TextVQA_VAL.xlsx'
            book = Workbook()
            book.active.append(['index', 'prediction'])
            book.active.append([123, 'irrelevant old prediction'])
            book.save(path)
            ids, evidence = data._history_coverage(root, 'TextVQA_VAL')
            self.assertEqual(ids, {'123'})
            self.assertEqual(evidence['evidence_sha256'], data.sha256(path))


if __name__ == '__main__':
    unittest.main()
