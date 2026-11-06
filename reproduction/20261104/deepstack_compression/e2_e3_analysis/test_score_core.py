import unittest
from copy import deepcopy
from paired import CONFIGS
from score_core import TASKS, inventory, native_predictions, score_rows


def fixtures():
    panel=[]; records=[]
    for i, task in enumerate(TASKS):
        row=dict(task=task,qid=str(i),group_id='g'+str(i),image_sha256='i'+str(i),message_sha256='m'+str(i))
        panel.append(row)
        for n, key in enumerate(CONFIGS):
            family,arm=key.split('/')
            records.append(dict(row,family=family,arm=arm,forced=False,answer='' if n==0 else '42',eos=True,truncated=False))
    return panel,records

class ScoreCoreTests(unittest.TestCase):
    def test_complete_inventory_xlsx_and_independent_scores(self):
        panel,records=fixtures(); grouped=inventory(panel,records,per_task=1)
        native=native_predictions(records,per_task=1)
        self.assertEqual(native[(TASKS[0],'0',CONFIGS[0])],'nan')
        self.assertEqual(native[(TASKS[0],'0',CONFIGS[1])],'42')
        truth={(r['task'],r['qid']):'42' for r in panel}
        def scorer(task,answer,gold,pred):
            value=float(answer==gold and pred=='42')
            return {'official':value,'independent':value}
        scored=score_rows(panel,grouped,truth,native,scorer)
        self.assertEqual(len(scored),3)
        self.assertEqual(scored[0]['arms'][CONFIGS[0]]['score'],0.)
        self.assertEqual(scored[0]['arms'][CONFIGS[1]]['score'],1.)
    def test_incomplete_reordered_or_changed_identities_fail(self):
        panel,records=fixtures()
        for broken in (records[:-1],records[1:2]+records[:1]+records[2:]):
            with self.assertRaises(ValueError):inventory(panel,broken,per_task=1)
        broken=deepcopy(records);broken[0]['image_sha256']='wrong'
        with self.assertRaises(ValueError):inventory(panel,broken,per_task=1)
        broken=deepcopy(panel);broken[1]['group_id']=broken[0]['group_id']
        with self.assertRaises(ValueError):inventory(broken,records,per_task=1)
    def test_disagreeing_or_nonfinite_scorers_fail(self):
        panel,records=fixtures();grouped=inventory(panel,records,per_task=1);native=native_predictions(records,per_task=1)
        truth={(r['task'],r['qid']):'x' for r in panel}
        for val in ({'official':1.,'independent':0.},{'official':float('nan'),'independent':float('nan')},{'official':1.,'independent':float('nan')},{'official':1.1,'independent':1.1}):
            with self.assertRaises(ValueError):score_rows(panel,grouped,truth,native,lambda *_:val)

if __name__=='__main__':unittest.main()
