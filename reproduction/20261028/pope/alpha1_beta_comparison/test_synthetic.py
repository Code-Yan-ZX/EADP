"""Small synthetic-only tests; never open live experiment outputs."""
import copy
import unittest
from unittest.mock import patch
import pair_completed as p


class PairingTests(unittest.TestCase):
    def data(self):
        qs=[dict(question_id=1,text='q1',image='a',category='x'),
            dict(question_id=2,text='q2',image='b',category='x'),
            dict(question_id=3,text='q3',image='c',category='y')]
        old=[dict(question_id=1,prompt='q1',text='No.'),dict(question_id=2,prompt='q2',text='yes'),
             dict(question_id=3,prompt='q3',text='NO')]
        new=[dict(question_id=1,prompt='q1',text='Yes'),dict(question_id=2,prompt='q2',text='not here'),
             dict(question_id=3,prompt='q3',text='yes')]
        obs=[dict(position=i,question_id=q['question_id'],image=q['image'],text=a['text'])
             for i,(q,a) in enumerate(zip(qs,new))]
        return qs,old,new,obs

    def test_transitions_and_totals(self):
        result=p.paired_summary(*self.data(),{'x':2,'y':1})
        self.assertEqual(result['total'],dict(n=3,raw_text_changed=3,yes_no_changed=2,
            yes_to_yes=1,yes_to_no=1,no_to_yes=1,no_to_no=0))
        self.assertEqual(result['by_category']['x']['yes_no_changed'],2)

    def test_official_parser_edge_cases(self):
        self.assertFalse(p.official_yes('No, it is. Yes'))
        self.assertFalse(p.official_yes('not here'))
        self.assertTrue(p.official_yes('NO'))
        self.assertTrue(p.official_yes(''))
        self.assertTrue(p.official_yes('yes. No'))

    def test_bad_image_id_prompt_and_partial_rejected(self):
        for part,key,value in [(0,'image','wrong'),(2,'question_id',8),(1,'prompt','wrong')]:
            data=list(copy.deepcopy(self.data()));data[part][0][key]=value
            with self.assertRaises(AssertionError):p.paired_summary(*data,{'x':2,'y':1})
        data=list(self.data());data[2]=data[2][:-1]
        with self.assertRaises(AssertionError):p.paired_summary(*data,{'x':2,'y':1})

    def test_official_log_parsing(self):
        text=''.join(f'Category: {name}, # samples: {n}\nTP\tFP\tTN\tFN\t\n1\t2\t3\t4\nF1 score: 0.5\n'
                     for name,n in p.CATEGORIES.items())
        result=p.official_categories(text)
        self.assertEqual(result['random']['FP'],2)
        self.assertEqual(result['popular']['f1_percent'],50)
        with self.assertRaises(AssertionError):p.official_categories(text+text)

    def test_absent_root_proof_prevents_other_reads(self):
        with patch.object(p,'sha',side_effect=FileNotFoundError('missing root')) as sha, patch.object(p,'read') as read:
            with self.assertRaises(FileNotFoundError):p.terminal_gate('0'*64)
            self.assertEqual(sha.call_count,1)
            read.assert_not_called()

    def test_completed_old_metadata_only(self):
        # This intentionally reads only previously completed old metadata/log.
        # Fail immediately if the helper tries new-run files or raw predictions.
        allowed={'completion_and_release_proof.json','independently_verified_full_summary.json',
                 'full_result_summary.json','run_plan.json','official_score.log'}
        real_sha,real_read=p.sha,p.read
        def checked(fn,path):
            path=p.Path(path)
            self.assertEqual(path.parent,p.OLD)
            self.assertIn(path.name,allowed)
            return fn(path)
        with patch.object(p,'sha',side_effect=lambda path:checked(real_sha,path)), \
             patch.object(p,'read',side_effect=lambda path:checked(real_read,path)):
            proof=p.read(p.OLD/'completion_and_release_proof.json')
            final,categories=p.check_old_metadata(proof)
        self.assertEqual(final['n_predictions'],8910)
        self.assertEqual(final['official_scorer_exit_code'],0)
        self.assertEqual(final['predictions_sha256'],proof['predictions_sha256'])
        self.assertEqual({k:v['n'] for k,v in categories.items()},p.CATEGORIES)

    def test_old_metadata_false_verification_rejected(self):
        real_read=p.read
        def altered(path):
            value=real_read(path)
            if p.Path(path).name=='full_result_summary.json':
                value['gpu_worker_exit_verified']=False
            return value
        proof=real_read(p.OLD/'completion_and_release_proof.json')
        with patch.object(p,'read',side_effect=altered):
            with self.assertRaises(AssertionError):p.check_old_metadata(proof)


if __name__=='__main__':unittest.main()
