"""Exercise the frozen comparison without reading any experimental scores."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).with_name('compare.py')
spec = importlib.util.spec_from_file_location('comparison_tested', SOURCE)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class SyntheticComparison(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.d = self.base / 'e6/pixel_t010_t020_comparison'
        self.r = self.base / 'e1_independent_review'
        self.d.mkdir(parents=True)
        self.r.mkdir()
        comparison.D, comparison.B, comparison.R = self.d, self.base, self.r
        self.tasks = ['DocVQA_VAL', 'TextVQA_VAL', 'AI2D_TEST']
        for tag, value in [('t010', .25), ('t020', .5)]:
            score = self.base / f'e6/pixelprune_{tag}_dev384_analysis/score001'
            score.mkdir(parents=True)
            rows = [{'task': t, 'qid': str(i), 'group_id': f'{i:04d}',
                     'arms': {f'pixel_{tag}': {'score': value}}}
                    for t in self.tasks for i in range(128)]
            (score / 'scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (score / 'analysis.json').write_text('{}')
            reviewer = self.r / f'review_pixel_{tag}_dev384_scores.py'
            reviewer.write_text('# synthetic reviewer\n')
            proof = {'status': 'passed', 'independent_review': True, 'questions': 384,
                     'errors': [], 'score_sha256': comparison.sha(score / 'scores.jsonl'),
                     'analysis_sha256': comparison.sha(score / 'analysis.json'),
                     'reviewer_sha256': comparison.sha(reviewer)}
            (self.r / f'pixel_{tag}_dev384_scores_review.json').write_text(json.dumps(proof))
            for monitor in [score.parent / 'monitor_score001', self.r / f'monitor_pixel_{tag}_dev384_scores']:
                monitor.mkdir()
                (monitor / 'started.json').write_text(json.dumps({'worker_pid': 1}))
                (monitor / 'terminal.json').write_text(json.dumps({'worker_pid': 1, 'reason': 'completed',
                    'worker_returncode': 0, 'remaining_process_group': []}))
        old = self.base / 'e6/pixelprune_t020_dev384_analysis/score001/scores.jsonl'
        plan = {'prior_t020_scores_sha256': comparison.sha(old), 'tasks': self.tasks,
                'limitations': ['synthetic only']}
        (self.d / 'plan001.json').write_text(json.dumps(plan))
        freeze = {'plan_sha256': comparison.sha(self.d / 'plan001.json'),
                  'script_sha256': comparison.sha(SOURCE)}
        (self.d / 'source_freeze001.json').write_text(json.dumps(freeze))

    def test_full_synthetic_pair(self):
        comparison.main()
        result = json.loads((self.d / 'result001.json').read_text())
        for task in self.tasks:
            self.assertEqual(result['results'][task]['mean_difference_pp'], -25)
            self.assertEqual(result['results'][task]['image_cluster_percentile_95_ci_pp'], [-25, -25])
            self.assertEqual(result['results'][task]['declined'], 128)

    def test_rejects_changed_reference(self):
        old = self.base / 'e6/pixelprune_t020_dev384_analysis/score001/scores.jsonl'
        old.write_text(old.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'old reference changed'):
            comparison.main()
        self.assertFalse((self.d / 'result001.json').exists())

    def test_rejects_failed_independent_review(self):
        path = self.r / 'pixel_t010_dev384_scores_review.json'
        proof = json.loads(path.read_text())
        proof['independent_review'] = False
        path.write_text(json.dumps(proof))
        with self.assertRaisesRegex(ValueError, 'proof missing'):
            comparison.main()
        self.assertFalse((self.d / 'result001.json').exists())


if __name__ == '__main__':
    unittest.main()
