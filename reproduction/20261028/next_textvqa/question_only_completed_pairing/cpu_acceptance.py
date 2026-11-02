"""Synthetic complete-before-prefix and accepted-score arithmetic contracts."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
import pair_completed as pair


class PairContracts(unittest.TestCase):
    def test_missing_terminal_never_reads_metadata_or_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = dict(pair.NEW_SPEC, directory=Path(tmp))
            with mock.patch.object(pair, 'read', side_effect=AssertionError('No reads before complete metadata')):
                with self.assertRaisesRegex(RuntimeError, 'Incomplete terminal/root proof'):
                    pair.terminal_gate(spec, '0' * 64, {})

    def test_new_terminal_failure_precedes_both_row_sets(self):
        with mock.patch.object(pair, 'terminal_gate', side_effect=[{'accepted_old_metadata': True}, RuntimeError('new pending')]), \
             mock.patch.object(pair, 'freeze_sources', side_effect=AssertionError('No row hash before both gates')), \
             mock.patch.object(pair, 'rows', side_effect=AssertionError('No prefix rows')):
            with self.assertRaisesRegex(RuntimeError, 'new pending'):
                pair.main('0' * 64)

    def test_exact_known_launch_schemas_and_reused_PID_rejection(self):
        for spec in [pair.OLD_SPEC, pair.NEW_SPEC]:
            launch = dict(root_owned_exec_handle=spec['root_exec'], plan_sha256=spec['plan_sha256'],
                **{spec['root_allocation_field']: spec['note_sha256']})
            if spec['directory'] == pair.OLD:
                launch['processes'] = [dict(spec[role], role=role, boot_id=pair.BOOT) for role in ['controller', 'native_worker']]
            else:
                launch.update(controller=spec['controller'], native_worker=spec['native_worker'])
            pair.check_launch_identity(launch, spec)
            bad = copy.deepcopy(launch)
            if spec['directory'] == pair.OLD:
                bad['processes'][1]['start_ticks'] += 1
            else:
                bad['native_worker']['start_ticks'] += 1
            with self.assertRaisesRegex(RuntimeError, 'launch identity'):
                pair.check_launch_identity(bad, spec)

    def test_score_summary_preserves_negative_and_unchanged_rows(self):
        rows = []
        for old, new, answer_changed in [(0, 3, True), (3, 3, False), (10, 6, True)]:
            rows.append(dict(full_guidance_score_tenths=old, question_only_score_tenths=new,
                full_guidance_quotas=[64] * 5, question_only_quotas=[64] * 5,
                raw_answer_changed=answer_changed, output_token_IDs_changed=answer_changed,
                quota_vector_changed=False, retained_total_changed=False,
                final_mask_changed=False, effective_ordered_prefix_changed=False))
        summary = pair.summarize(rows)
        self.assertEqual(summary['n'], 3)
        self.assertEqual(summary['gained_tenths'], 3)
        self.assertEqual(summary['lost_tenths_signed'], -4)
        self.assertEqual(summary['net_tenths'], -1)
        for key in ['score_improved', 'score_declined', 'score_unchanged']:
            self.assertEqual(summary['counts'][key], 1)
        self.assertEqual(summary['counts']['raw_answer_changed'], 2)
        self.assertEqual(summary['counts']['final_mask_changed'], 0)
        with self.assertRaisesRegex(RuntimeError, 'integer tenths'):
            pair.units(.333)

    def test_actual_prefix_membership_and_physical_not_nominal_budget(self):
        quotas = [1, 1, 1, 1, 320]  # synthetic physical capacity example, total324
        masks = [list(range(n)) for n in quotas]
        observation = dict(position=0, input_shape=[1, 3], input_dtype='torch.int64', input_ids_sha256='inputSHA',
            crop_count=5, initial_quotas=[64] * 5, allocated_quotas=quotas, greedy_quotas=quotas,
            effective_mask_counts=quotas, effective_mask_shape=[5, 576], effective_mask_indices=masks,
            greedy_effective_prefixes=masks, returned_retained_tokens=324, importance_shape=[5, 576],
            importance_finite=True, projected_feature_shape=[5, 576, 4096], projected_feature_dtype='torch.float16',
            post_generation_tensor_reads_only=True)
        args = [dict(native_quotas=quotas), dict(actual_visual_tokens_retained=324), dict(n_tokens=3, token_ids_sha256='inputSHA'), 0]
        pair.validate_observation(observation, *args)
        bad = copy.deepcopy(observation); bad['greedy_effective_prefixes'][4][-1] = 999
        with self.assertRaisesRegex(RuntimeError, 'membership'):
            pair.validate_observation(bad, *args)
        bad = copy.deepcopy(observation); bad['input_ids_sha256'] = 'wrong'
        with self.assertRaisesRegex(RuntimeError, 'LLM IDs'):
            pair.validate_observation(bad, *args)

    def test_default_invocation_does_not_read_or_write_run_artifacts(self):
        process = subprocess.run([sys.executable, '-B', str(pair.OUT / 'pair_completed.py')], text=True, capture_output=True)
        self.assertEqual(process.returncode, 0)
        self.assertIn('no run artifacts read', process.stdout)
        self.assertNotIn('torch', sys.modules)
        self.assertFalse((pair.OUT / 'comparison_report.json').exists())
        self.assertFalse((pair.OUT / 'paired_rows.jsonl').exists())


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PairContracts))
    if not result.wasSuccessful():
        raise SystemExit(1)
    report = dict(status='passed', tests_run=result.testsRun,
        implementations={name: hashlib.sha256((pair.OUT / name).read_bytes()).hexdigest()
            for name in ['pair_completed.py', 'cpu_acceptance.py', 'README.md']},
        scope='Synthetic/mocked only; default invocation reads no run artifacts.',
        real_pairing_executed=False, new_prefix_read=False, scorer_invoked=False, model_loaded=False, GPU_started=False)
    (pair.OUT / 'CPU_acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
