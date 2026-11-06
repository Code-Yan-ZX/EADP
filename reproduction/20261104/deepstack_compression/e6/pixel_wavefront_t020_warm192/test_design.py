import copy,unittest
import design
class DesignTest(unittest.TestCase):
 def fixtures(self):
  rule={'tasks':list(design.TASKS),'process_arm_order':[list(x) for x in design.ARMS],'GPU_authorized':False,'GT_opened':False,'seed':'unit'};rows=[];cohort={};refs={}
  for t in design.TASKS:
   m=[];w=[]
   for i in range(74):
    g=t+str(i);r={'task':t,'qid':str(i),'group_id':g,'split':'DEV','image_path':'/fake','image_sha256':'a'*64,'message_sha256':'b'*64};rows.append(r);refs[(t,str(i))]={k:r[k] for k in ('task','qid','group_id','image_sha256','message_sha256')}|{'processor':{},'image_start':2};(m if i<64 else w).append(g)
   cohort[t]={'measured':m,'warmup':w}
  return rule,cohort,rows,refs
 def test_fixed_complete_reproducible_and_balanced(self):
  args=self.fixtures();p=design.build(*args);self.assertEqual(p,design.build(*args));self.assertEqual((p['selected_questions'],p['measured_records'],p['warmup_records']),(192,384,8));self.assertEqual([x['arm'] for x in p['processes']],['author','wavefront','wavefront','author','wavefront','author','author','wavefront'])
  for b in p['blocks']:
   self.assertEqual({t:sum(x['task']==t for x in b) for t in design.TASKS},{t:16 for t in design.TASKS})
 def test_missing_or_mismatched_identity_fails(self):
  rule,cohort,rows,refs=self.fixtures();refs[(rows[0]['task'],rows[0]['qid'])]['image_sha256']='c'*64
  with self.assertRaises(ValueError):design.build(rule,cohort,rows,refs)
 def test_warmup_overlap_fails(self):
  rule,cohort,rows,refs=self.fixtures();cohort[design.TASKS[0]]['warmup'][0]=cohort[design.TASKS[0]]['measured'][0]
  with self.assertRaises(ValueError):design.build(rule,cohort,rows,refs)
if __name__=='__main__':unittest.main()
