"""Meaningful CPU checks for source-only panel selection and exclusion."""
import base64
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from . import data


def image_bytes(seed):
    rng = random.Random(seed)
    image = Image.frombytes('RGB', (27, 21), bytes(rng.randrange(256) for _ in range(27 * 21 * 3)))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


class DetailDataTests(unittest.TestCase):
    def make_sources(self, root):
        paths = []
        for dataset, name in [('InfoVQA_TRAIN', 'info'), ('TextVQA_TRAIN', 'text')]:
            path = root / name / 'train-00000-of-00001.parquet'
            path.parent.mkdir()
            rows = []
            for i in range(7):
                for q in range(2):
                    row = dict(question=f'original question {i} {q}?', answers=['yes'] * (10 if name == 'text' else 1),
                               image={'bytes': image_bytes(i + (0 if name == 'info' else 100)), 'path': f'{name}_{i}.png'})
                    if name == 'info':
                        row.update(questionId=str(i * 2 + q), image_url=f'https://fixture/{i}', data_split='train')
                    else:
                        row.update(question_id=i * 2 + q, image_id=f'{name}_{i}', image_width=27,
                                   image_height=21, set_name='train')
                    rows.append(row)
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
            paths.append(path)
        return paths

    def test_stable_disjoint_roles_one_question_per_image_and_all_votes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info, text = self.make_sources(root)
            manifest = data.build_manifest(root / 'run', info, text, 3, 1, 7, exclusion_manifests=())
            again = data.build_manifest(root / 'run', info, text, 3, 1, 7, exclusion_manifests=())
            self.assertEqual(manifest, again)
            all_items = []
            for name in data.DATASETS:
                ds = manifest['datasets'][name]
                self.assertEqual(len(ds['items']), 3)
                self.assertEqual(len(ds['dev']), 1)
                candidates, _ = data._source_rows(info if name.startswith('Info') else text, name, 7)
                self.assertEqual(ds['dev'][0]['source_image_id'], candidates[0]['source_image_id'])
                for item in ds['dev'] + ds['items']:
                    all_items.append(item)
                    self.assertEqual(item['source_question_count'], 2)
                    self.assertTrue(item['question'].startswith('original question'))
                    self.assertEqual(len(item['answers']), 10 if name.startswith('Text') else 1)
            self.assertEqual(len({i['parent_id'] for i in all_items}), 8)
            self.assertEqual(len({i['pixel_sha256'] for i in all_items}), 8)
            self.assertFalse(manifest['selection']['uses_answers_or_predictions_for_selection'])
            with self.assertRaisesRegex(ValueError, 'different selection'):
                data.build_manifest(root / 'run', info, text, 3, 1, 8, exclusion_manifests=())

    def test_exact_and_near_history_exclusion_replenishes_without_answer_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info, text = self.make_sources(root)
            candidates, _ = data._source_rows(info, 'InfoVQA_TRAIN', 7)
            selected = candidates[0]
            raw = data._image_payloads(info, [selected])[selected['source_row']]
            metadata = data.materialize_image(raw, '.png', root / 'history')
            history = root / 'previous_manifest.json'
            history.write_text(json.dumps(dict(datasets={'old': dict(items=[dict(
                metadata, dataset='old', item_id='old1', has_image=True)])})))
            manifest = data.build_manifest(root / 'run', info, text, 3, 1, 7, exclusion_manifests=(history,))
            ds = manifest['datasets']['InfoVQA_TRAIN']
            self.assertEqual(ds['counts']['rejected'], 1)
            self.assertEqual(ds['rejected_candidates'][0]['image_id'], selected['source_image_id'])
            self.assertIn('exact', ds['rejected_candidates'][0]['reason'])
            self.assertEqual(len(ds['items']), 3)
            first = ds['dev'][0]
            near = data._near_matches(first, [dict(identity='same', dataset='fixture',
                dhash64=first['dhash64'], image_size=first['image_size'])])
            self.assertEqual(near[0]['dhash_hamming'], 0)

    def test_modified_sources_and_images_fail_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info, text = self.make_sources(root)
            manifest = data.build_manifest(root / 'run', info, text, 2, 1, 9, exclusion_manifests=())
            image = Path(manifest['datasets']['InfoVQA_TRAIN']['items'][0]['image_path'])
            image.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'input image changed'):
                data.verify_manifest(manifest)
            with info.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                data.verify_manifest(manifest)

    def test_hf_snapshot_symlink_preserves_original_train_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info, text = self.make_sources(root)
            blob = root / 'extension_free_blob'
            info.rename(blob)
            info.symlink_to(blob)
            rows, inventory = data._source_rows(info, 'InfoVQA_TRAIN', 7)
            self.assertEqual(len(rows), 7)
            self.assertEqual(inventory['num_original_questions'], 14)


if __name__ == '__main__':
    unittest.main()
