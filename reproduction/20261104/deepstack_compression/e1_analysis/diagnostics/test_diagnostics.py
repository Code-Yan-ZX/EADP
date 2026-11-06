import tempfile,unittest
from pathlib import Path
from analyze import labels,analyze

class DiagnosticsTests(unittest.TestCase):
    def test_selected_label_only_ignores_invalid_nondev_bytes_both_index_positions(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.tsv'
            p.write_bytes(b'index\tcategory\tanswer\n1\tpartsOfA\tok\n2\tbad\t\xff\n')
            self.assertEqual(labels(p,'AI2D_TEST',{'1'},'category'),{'1':['partsOfA']})
            p.write_bytes(b'question_types\tanswer\tindex\n"[\'form\', \'layout\']"\tok\t1\ninvalid\t\xff\t2\n')
            self.assertEqual(labels(p,'DocVQA_VAL',{'1'},'question_types'),{'1':['form','layout']})
    def test_missing_duplicate_and_unparseable_selected_rows_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.tsv'; p.write_bytes(b'index\tcategory\tanswer\n1\tcat\tx\n1\tcat\ty\n')
            with self.assertRaises(ValueError):labels(p,'AI2D_TEST',{'1'},'category')
            with self.assertRaises(ValueError):labels(p,'AI2D_TEST',{'2'},'category')
    def test_sparse_groups_have_no_interval_and_counts_retained(self):
        arms=('P1-D0','P1-D1','P3-D0','P3-D1')
        row={'group_id':'g','scores':dict(zip(arms,[0,1,0,1])),'answers':dict(zip(arms,['x','y','x','y']))}
        result=analyze([row]);self.assertEqual(result['n_questions'],1)
        self.assertEqual(result['contrasts']['DS_at_P1']['difference_pp'],100)
        self.assertEqual(result['contrasts']['DS_at_P1']['answer_changed'],1)
        self.assertIsNone(result['contrasts']['DS_at_P1']['image_cluster_percentile_95_ci_pp'])
if __name__=='__main__':unittest.main()
