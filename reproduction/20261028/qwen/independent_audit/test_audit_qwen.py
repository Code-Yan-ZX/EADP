import copy
from pathlib import Path
import string
import unittest

import audit_qwen as audit


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.canonical=[dict(index=str(i),question='question '+str(i),answer='A',A='red',B='blue',C='green',D='NA')
                        for i in [1,2]]
        self.predictions=[dict(row,prediction='A',D='') for row in self.canonical]

    def test_valid_native_na_and_full_alignment(self):
        self.assertTrue(audit.validate_predictions(self.canonical,self.predictions,2))

    def test_subset_cannot_be_full_dataset(self):
        with self.assertRaises(AssertionError):
            audit.validate_predictions(self.canonical,self.predictions,5349)

    def test_duplicate_or_reordered_ids_rejected(self):
        for predictions in [self.predictions[::-1],self.predictions[:1]*2]:
            with self.subTest(predictions=predictions), self.assertRaises(AssertionError):
                audit.validate_predictions(self.canonical,predictions,2)

    def test_changed_question_gt_or_choice_rejected(self):
        for key,value in [('question','different'),('answer','B'),('A','different')]:
            prediction=copy.deepcopy(self.predictions);prediction[0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(AssertionError):
                audit.validate_predictions(self.canonical,prediction,2)

    def test_empty_answers_keep_coverage_and_full_denominator(self):
        predictions=copy.deepcopy(self.predictions);predictions[0]['prediction']=''
        report=audit.validate_predictions(self.canonical,predictions,2)
        self.assertTrue(report['coverage_complete'])
        self.assertEqual(report['n'],2)
        self.assertEqual(report['empty_prediction_count'],1)
        self.assertEqual(report['empty_count'],1)
        self.assertTrue(report['generation_valid'])
        self.assertFalse(report['planned_no_empty_gate_passed'])
        self.assertTrue(report['generation_without_failure_placeholders'])

    def test_failure_placeholders_are_flagged_without_deleting_rows(self):
        for failure in ['Failed to obtain answer via API.', 'Failed to obtain answer: RuntimeError synthetic']:
            predictions=copy.deepcopy(self.predictions);predictions[0]['prediction']=failure
            report=audit.validate_predictions(self.canonical,predictions,2)
            self.assertTrue(report['coverage_complete'])
            self.assertEqual(report['n'],2)
            self.assertEqual(report['failure_placeholder_count'],1)
            self.assertFalse(report['generation_without_failure_placeholders'])
            self.assertFalse(report['generation_valid'])
            self.assertTrue(report['planned_no_empty_gate_passed'])

    def test_empty_literal_and_native_pandas_scores_are_kept_distinct(self):
        # Synthetic case; no running prefix/ground truth is inspected or scored.
        self.assertEqual(audit.scoring_prediction(''),'nan')
        self.assertEqual(audit.independent_anls('',['none']),0.)
        self.assertEqual(audit.independent_anls(audit.scoring_prediction(''),['none']),.5)

    def test_distance_and_anls_thresholds(self):
        self.assertEqual(audit.independent_distance('kitten','sitting'),3)
        self.assertEqual(audit.independent_anls('ac',['ab']),.5)
        self.assertEqual(audit.independent_anls('zz',['ab']),0.)
        self.assertEqual(audit.independent_anls('  Foo  ',['foo']),1.)

    def test_mcq_semantics_against_author(self):
        helper=audit.author_helpers()['can_infer']
        choices=dict(A='red',B='blue',C='green',D='white')
        for prediction in ['A','(B).','The correct option is C.','blue','A or B','b','Cannot determine the answer']:
            with self.subTest(prediction=prediction):
                self.assertEqual(audit.independent_option(prediction,choices),
                                 helper(prediction,copy.deepcopy(choices)) or None)

    def test_real_old_full_docvqa_xlsx_and_pointwise_anls(self):
        # Regression input only: these are preserved old predictions, not new FA2 results.
        raw=Path('/media/disk2/YZX/research/audit_base_gap_20261003/archive_recovery/20261003_r2/old_predictions/old_DocVQA_VAL.xlsx')
        predictions=audit.read_xlsx(raw)
        canonical=audit.jsonl(audit.HERE/'DocVQA_VAL.canonical.jsonl')
        self.assertTrue(audit.validate_predictions(canonical,predictions,5349))
        helper=audit.author_helpers()['anls_compute'];scores=[]
        for row,pred in zip(canonical,predictions):
            gts=audit.parse_answers(row['answer']);text=audit.scoring_prediction(pred['prediction'])
            distance=min(helper(gt,text) for gt in gts)
            expected=1.-distance if 1.-distance>=.5 else 0.
            actual=audit.independent_anls(text,gts)
            self.assertAlmostEqual(actual,expected,places=12);scores.append(actual)
        self.assertAlmostEqual(100*sum(scores)/5349,61.135577708229846,places=9)
        self.assertEqual(sum(audit.scoring_prediction(r['prediction'])!=r['prediction'] for r in predictions),8)

    def test_real_old_full_ai2d_tsv_and_pointwise_mcq(self):
        raw=Path('/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/stage1_roundtrip_pilot/legacy_full/acc/L_E_GATHER/AI2D_TEST_pred.tsv')
        predictions=audit.read_table(raw)
        canonical=audit.jsonl(audit.HERE/'AI2D_TEST.canonical.jsonl')
        self.assertTrue(audit.validate_predictions(canonical,predictions,3088))
        helper=audit.author_helpers()['can_infer'];hits=[]
        for row,pred in zip(canonical,predictions):
            choices={k:audit.native_cell(row[k]) for k in string.ascii_uppercase
                     if k in row and audit.native_cell(row[k]) is not None}
            expected=helper(pred['prediction'],copy.deepcopy(choices)) or None
            actual=audit.independent_option(pred['prediction'],choices)
            self.assertEqual(expected,actual)
            hits.append(actual==row['answer'])
        self.assertAlmostEqual(100*sum(hits)/3088,74.54663212435233,places=9)


if __name__=='__main__':
    unittest.main()
