import copy
import unittest
import score_core as c

class ScoringContract(unittest.TestCase):
    def rows(self):
        panel=[{'task':t,'qid':str(i),'group_id':f'g{i}','image_sha256':f'i{i}','message_sha256':f'm{i}'} for i,t in enumerate(c.TASKS)]
        records=[dict(r,arm=a,forced=False,answer=('012' if r['task']==c.TASKS[0] else 'x'),eos=True,truncated=False) for a in c.ARMS for r in panel]
        return panel,records
    def test_full_arm_major_inventory_and_tamper(self):
        p,r=self.rows();out=c.inventory(p,r,per_task=1);self.assertEqual(len(out),3)
        for field in ('arm','group_id','image_sha256','message_sha256','forced'):
            broken=copy.deepcopy(r);broken[1][field]='tampered'
            with self.assertRaises(ValueError):c.inventory(p,broken,per_task=1)
        with self.assertRaises(ValueError):c.inventory(p,r[:-1],per_task=1)
    def test_xlsx_conversion_and_score_disagreement(self):
        p,r=self.rows();native=c.native_predictions(r,per_task=1)
        self.assertEqual(native[(c.TASKS[0],'0',c.ARMS[0])],'12')
        grouped=c.inventory(p,r,per_task=1);truth={(x['task'],x['qid']):['x'] for x in p}
        ref={(x['task'],x['qid']):{'group_id':x['group_id'],'arms':{'P3-D1':{'score':1.,'independent_score':1.,'eos':True,'truncated':False}}} for x in p}
        good=lambda *args:{'official':0.5,'independent':0.5}
        self.assertEqual(len(c.score_rows(p,grouped,truth,native,ref,good)),3)
        with self.assertRaisesRegex(ValueError,'author and independent'):
            c.score_rows(p,grouped,truth,native,ref,lambda *args:{'official':0.5,'independent':0.4})
        ref[(c.TASKS[0],'0')]['group_id']='wrong'
        with self.assertRaisesRegex(ValueError,'group mismatch'):c.score_rows(p,grouped,truth,native,ref,good)
    def test_group_bootstrap_and_constant_difference(self):
        arms={a:{'score':v} for a,v in zip((*c.ARMS,c.REFERENCE),(0.2,0.4,0.6,0.5))}
        rows=[{'group_id':'a','arms':arms},{'group_id':'a','arms':arms},{'group_id':'b','arms':arms}]
        out=c.analyze(rows,replicates=100,seed=3);self.assertEqual(out['n_image_groups'],2)
        contrast=out['contrasts']['priortr_k256 - '+c.REFERENCE]
        self.assertAlmostEqual(contrast['difference_pp'],10.)
        for x in contrast['percentile_95_ci_pp']:self.assertAlmostEqual(x,10.)
        broken=copy.deepcopy(rows);broken[0]['arms'][c.ARMS[0]]['score']=float('nan')
        with self.assertRaisesRegex(ValueError,'invalid scores'):c.analyze(broken,replicates=3)
if __name__=='__main__':unittest.main()
