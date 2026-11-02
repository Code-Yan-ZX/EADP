import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze_and_score as m


class TerminalGateTests(unittest.TestCase):
    def fixture(self, root):
        def put(name, value):
            (root / name).write_text(json.dumps(value) + '\n')
        put('inputs.json', {'code_sha256': 'test'})
        (root / 'observations.jsonl').write_text('{"not":"a real run"}\n')
        complete = {'n': 1, 'observations_sha256': m.sha(root / 'observations.jsonl'),
                    'inputs': {'code_sha256': 'test'}}
        put('complete.json', complete)
        put('process_exit_1_2.json', {'container_exit_code': 0})
        proof = {'root_owned_exec_actual_exit_code': 0, 'container_exit_code': 0,
                 'root_session_id': 123, 'original_container_exited': True,
                 'complete_sha256': m.sha(root / 'complete.json'),
                 'process_exit_file': 'process_exit_1_2.json',
                 'process_exit_sha256': m.sha(root / 'process_exit_1_2.json'),
                 'observations_sha256': complete['observations_sha256']}
        put('root_exit_proof.json', proof)
        return proof

    def test_gate_accepts_complete_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); proof = self.fixture(root)
            self.assertEqual(m.gate(root, 1)['observations_sha256'], proof['observations_sha256'])
            (root / 'observations.jsonl').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'observations hash mismatch'): m.gate(root, 1)

    def test_incomplete_proof_never_opens_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); proof = self.fixture(root)
            proof['root_owned_exec_actual_exit_code'] = 1
            (root / 'root_exit_proof.json').write_text(json.dumps(proof))
            original_sha = m.sha
            def guarded_sha(path):
                if Path(path).name == 'observations.jsonl':
                    self.fail('opened observations before validating exit')
                return original_sha(path)
            with patch.object(m, 'sha', guarded_sha):
                with self.assertRaisesRegex(ValueError, 'root exec'): m.gate(root, 1)

    def test_missing_proof_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError): m.gate(Path(tmp), 5349)


class SummaryTests(unittest.TestCase):
    def test_comparisons_and_distributions(self):
        ids = list(range(256))
        manifest = [dict(position=0,index='a'), dict(position=1,index='b')]
        prior = [dict(position=0,index='a',response='yes'), dict(position=1,index='b',response='old')]
        prior_obs = [dict(position=0,index='a',output_ids=[5,6],ordered_ids=ids),
                     dict(position=1,index='b',output_ids=[8],ordered_ids=ids)]
        rows = [dict(position=0,index='a',response='yes',output_ids=[5,6],ordered_ids=ids,
                     seconds=1.0,prefill_gpu_ms=10.,peak_allocated_bytes=10,peak_reserved_bytes=20,
                     output_tokens=2,injected=1,eos_present=True,at_max_tokens=False,
                     prior_response_match=True,prior_output_ids_match=True,prior_selection_match=True),
                dict(position=1,index='b',response='new',output_ids=[9],ordered_ids=ids,
                     seconds=3.0,prefill_gpu_ms=20.,peak_allocated_bytes=12,peak_reserved_bytes=20,
                     output_tokens=1,injected=1,eos_present=False,at_max_tokens=True,
                     prior_response_match=False,prior_output_ids_match=False,prior_selection_match=True)]
        result = m.summarize(rows, manifest, prior, prior_obs)
        self.assertEqual(result['prior_response_matches'], 1)
        self.assertEqual(result['prior_output_id_matches'], 1)
        self.assertEqual(result['prior_selection_matches'], 2)
        self.assertEqual(result['max_tokens_without_eos'], 1)
        self.assertEqual(result['seconds']['p50'], 2.0)
        self.assertEqual(result['output_tokens_per_s'], .75)
        rows[0]['prior_output_ids_match'] = False
        with self.assertRaisesRegex(ValueError, 'stored comparison flag'): m.summarize(rows, manifest, prior, prior_obs)


if __name__ == '__main__':
    unittest.main()
