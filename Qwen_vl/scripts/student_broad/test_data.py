"""CPU checks for official identities, MCQ rotations and image-free controls."""
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


class BroadDataTests(unittest.TestCase):
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
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (root / (name + '.tsv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)

    def test_scalar_and_list_answers_preserve_values_and_votes(self):
        for value in ('A', 'Yes', '14', '0', '-1.25', '1,234'):
            self.assertEqual(data.parse_answers(value, 'fixture', '1'), [value])
        self.assertEqual(data.parse_answers("['yes', 'yes', 'no']", 'fixture', '1'), ['yes', 'yes', 'no'])
        self.assertEqual(data.parse_answers("['pinterest']", 'fixture', '1'), ['pinterest'])
        for value in ('', '[]', '[1]', "['broken]"):
            with self.assertRaises(ValueError):
                data.parse_answers(value, 'fixture', '1')

    def test_chartqa_literal_brackets_and_numeric_lists_follow_official_parser(self):
        self.assertEqual(data.parse_answers('[Gambia, Niger]', 'ChartQA_TEST', '304'), ['[Gambia, Niger]'])
        self.assertEqual(data.parse_answers('[2014, 2016]', 'ChartQA_TEST', '349'), ['2014', '2016'])
        self.assertEqual(data.parse_answers("['a', 'b']", 'ChartQA_TEST', '1'), ['a', 'b'])

    def test_reference_rotation_options_hint_and_all_metadata_survive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_training(root)
            payload = base64.b64encode((source / 'data/fit.png').read_bytes()).decode()
            rows = [dict(index='1000007', image='7', question='first?', A='other', B='same',
                         answer='B', hint='context', category='cat', **{'l2-category': 'l2'}),
                    dict(index='7', image=payload, question='second?', A='same', B='other',
                         answer='A', hint='', category='cat', **{'l2-category': 'l2'})]
            self.write_dataset(root, 'MMBench_DEV_EN_V11', rows)
            with patch.dict(data.EXPECTED_COUNTS, MMBench_DEV_EN_V11=2):
                result = data.build_manifest(root / 'out', source, ('MMBench_DEV_EN_V11',),
                                             source_root=root, history_root=root / 'none')
                again = data.build_manifest(root / 'out', source, ('MMBench_DEV_EN_V11',),
                                            source_root=root, history_root=root / 'none')
            self.assertEqual(result, again)
            ds = result['datasets']['MMBench_DEV_EN_V11']
            self.assertEqual(ds['num_items'], 2)
            self.assertEqual(ds['num_unique_images'], 1)
            self.assertEqual(ds['training_overlap_count'], 2)
            self.assertEqual(ds['novel_item_ids'], [])  # unknown history is not unseen
            a, b = ds['items']
            self.assertEqual(a['image_source_index'], '7')
            self.assertEqual(a['g_index'], b['g_index'])
            self.assertEqual(a['options'], {'A': 'other', 'B': 'same'})
            self.assertEqual(a['hint'], 'context')
            self.assertEqual(a['metadata']['l2-category'], 'l2')
            self.assertEqual(a['answers'], ['B'])
            self.assertEqual(a['parent_id'], b['parent_id'])
            Path(a['image_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'benchmark input changed'):
                data.verify_manifest(result)

    def test_hallusion_text_controls_retained_without_synthetic_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_training(root)
            payload = base64.b64encode((source / 'data/fit.png').read_bytes()).decode()
            rows = [dict(index='VS_chart_0_4_0_0', image='', image_path='', question='text control?',
                         answer='No', category='VS', **{'l2-category': 'VS_chart'}),
                    dict(index='VS_chart_0_4_1_0', image=payload, image_path='example.png', question='image?',
                         answer='Yes', category='VS', **{'l2-category': 'VS_chart'})]
            self.write_dataset(root, 'HallusionBench', rows)
            with patch.dict(data.EXPECTED_COUNTS, HallusionBench=2):
                result = data.build_manifest(root / 'out', source, ('HallusionBench',),
                                             source_root=root, history_root=root / 'none')
            ds = result['datasets']['HallusionBench']
            self.assertEqual(ds['num_items'], 2)
            self.assertEqual(ds['num_text_only_items'], 1)
            self.assertEqual(ds['num_unique_images'], 1)
            a, b = ds['items']
            self.assertFalse(a['has_image'])
            self.assertTrue(all(a[k] is None for k in data.IMAGE_FIELDS))
            self.assertEqual(a['parent_id'], b['parent_id'])
            self.assertEqual(a['set_id'], '4')
            self.assertEqual(a['figure_id'], '0')
            self.assertEqual(a['question_id'], '0')
            data.verify_manifest(result)

    def test_mme_source_path_group_and_referenced_image(self):
        row = dict(index='2', question='yes?', answer='No', category='OCR',
                   image_path='OCR/page.jpg')
        item = data._row_metadata('MME', row, 0, set())
        self.assertEqual(item['mme_image_group'], 'OCR:OCR/page.jpg')
        self.assertEqual(item['metadata']['image_path'], 'OCR/page.jpg')
        self.assertEqual(item['source_image_path'], 'OCR/page.jpg')

    def test_original_mmstar_empty_correct_option_is_retained_and_flagged(self):
        row = dict(index='268', question='How many?', answer='A', A='', B='Two',
                   C='One', D='Cannot be determined')
        item = data._row_metadata('MMStar', row, 268, set())
        self.assertNotIn('A', item['options'])
        self.assertEqual(item['metadata']['A'], '')
        self.assertEqual(item['answers'], ['A'])
        self.assertEqual(item['annotation_warnings'], ['empty_correct_option:A'])

    def test_public_options_and_hint_follow_pandas_default_na_semantics(self):
        row = dict(index='28809', question='which?', answer='A', A='real', B='None',
                   C='NA', D=' ', hint='null')
        item = data._row_metadata('AI2D_TEST', row, 0, set())
        self.assertEqual(item['options'], {'A': 'real', 'D': ' '})
        self.assertEqual(item['hint'], '')
        self.assertEqual(item['metadata']['B'], 'None')

    def test_empty_noncontrol_images_and_invalid_mcq_labels_fail(self):
        row = dict(index='2', question='yes?', answer='C', A='yes', B='no')
        with self.assertRaisesRegex(ValueError, 'invalid label'):
            data._row_metadata('AI2D_TEST', row, 0, set())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_training(root)
            self.write_dataset(root, 'ChartQA_TEST', [dict(index='1', question='q?', answer='3', image='')])
            with self.assertRaisesRegex(ValueError, 'Empty image'):
                data.build_manifest(root / 'out', source, ('ChartQA_TEST',),
                                    source_root=root, history_root=root / 'none')

    def test_local_alias_and_duplicate_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for datasets in (('POPE_local',), ('POPE', 'POPE')):
                with self.assertRaisesRegex(ValueError, 'unique supported'):
                    data.build_manifest(Path(temporary), Path(temporary), datasets)


if __name__ == '__main__':
    unittest.main()
