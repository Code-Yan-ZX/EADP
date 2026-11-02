import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

path = Path(__file__).with_name('benchmark.py')
spec = importlib.util.spec_from_file_location('benchmark', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CheckpointTests(unittest.TestCase):
    def test_contiguous_prefix_and_damage(self):
        m = dict(prior_selection_match=True, injected=1, prior_response_match=True, prior_output_ids_match=True)
        manifest = [dict(position=0, index='a', **m), dict(position=1, index='b', **m)]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'out.jsonl'
            self.assertEqual(module.completed_prefix(p, manifest), 0)
            p.write_text(json.dumps(manifest[0]) + '\n')
            self.assertEqual(module.completed_prefix(p, manifest), 1)
            p.write_text(json.dumps(manifest[0]))
            with self.assertRaises(ValueError): module.completed_prefix(p, manifest)
            p.write_text(json.dumps(manifest[1]) + '\n')
            with self.assertRaises(ValueError): module.completed_prefix(p, manifest)
            p.write_text(json.dumps(dict(manifest[0], prior_selection_match=False)) + '\n')
            with self.assertRaises(ValueError): module.completed_prefix(p, manifest)
            p.write_text(json.dumps(dict(manifest[0], prior_output_ids_match=False)) + '\n')
            with self.assertRaises(ValueError): module.completed_prefix(p, manifest)

    def test_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'v'
            p.write_bytes(b'abc')
            self.assertEqual(module.sha(p), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')


if __name__ == '__main__':
    unittest.main()
