import copy,unittest
import core
SEEDS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
def fixture():
 rows=[];measure=[];warm=[];panel={};fixed={};jobs=[]
 for t in SEEDS:
  for i in range(74):
   key=str(i);base={'task':t,'qid':key,'group_id':t+'-'+key,'image_path':'/fake','image_sha256':'a'*64,'message_sha256':'b'*64,'split':'DEV','processor_reference':{'hash':'p'},'image_start_reference':3};(measure if i<64 else warm).append(base);rows.append({**base,'processor_expected':base['processor_reference'],'image_start':3})
   for k in core.KS:
    d={'task':t,'qid':key,'group_id':base['group_id'],'image_sha256':base['image_sha256'],'message_sha256':base['message_sha256'],'processor':base['processor_reference'],'image_start':3,'budget':k,'ds_on':True,'generation_scalars':{'max_new_tokens':2048},'eos_ids':[99],'output_ids':[99],'answer':'ok','eos':True,'truncated':False};panel[(t,key,k)]=d;fixed[(t,key,k)]={**d,'protocol':'fixed32','output_ids':[1]*31+[99],'min_new_tokens':32,'generation_scalars':{'max_new_tokens':32}}
 for phase,items,reps in (('warmup',warm,1),('measure',measure,5)):
  for rep in range(reps):
   for r in items:
    for k in core.KS:
     for p in core.PROTOCOLS:jobs.append({'phase':phase,'replicate':rep,'task':r['task'],'qid':r['qid'],'group_id':r['group_id'],'budget':k,'protocol':p,'order':len(jobs)})
 return {'status':'proposal_frozen','measurement_identities':measure,'warmup_identities':warm,'jobs':jobs,'GT_opened':False,'GPU_authorized':False},rows,panel,fixed
def execute(row,k,p,panel,raw):
 rec={x:panel[x] for x in ('task','qid','group_id','budget','output_ids','answer','eos_ids','generation_scalars','eos','truncated')};rec['protocol']=p;rec['min_new_tokens']=None
 if p=='fixed32':rec.update(output_ids=[1]*31+[99],min_new_tokens=32,generation_scalars={'max_new_tokens':32})
 raw({x:rec[x] for x in ('task','qid','group_id','budget','protocol','output_ids','eos_ids','generation_scalars','min_new_tokens')});return rec
class Tests(unittest.TestCase):
 def test_all_7920_complete(self):
  p,rows,panel,fixed=fixture();events=[];r=core.run(p,rows,panel,fixed,execute,events.append);self.assertEqual((r['jobs'],r['measured'],r['fixed32_measure_keys'],len(events)),(7920,7680,768,23760))
 def test_changed_fixed_and_wrong_reference_refused(self):
  p,rows,panel,fixed=fixture();one=next(iter(fixed));fixed[one]['output_ids']=[99]*32
  with self.assertRaises(ValueError):core.prepare(p,rows,panel,fixed)
  p,rows,panel,fixed=fixture();one=next(iter(panel));panel[one]['processor']={'hash':'other'}
  with self.assertRaises(ValueError):core.prepare(p,rows,panel,fixed)
 def test_schedule_leak_and_replicate_refused(self):
  for kind in ('dup','wrong-rep','overlap'):
   p,rows,panel,fixed=fixture()
   if kind=='dup':p['jobs'][1]={**p['jobs'][0],'order':1}
   if kind=='wrong-rep':p['jobs'][-1]['replicate']=9
   if kind=='overlap':p['measurement_identities'][0]=p['warmup_identities'][0]
   with self.subTest(kind=kind),self.assertRaises(ValueError):core.prepare(p,rows,panel,fixed)
 def test_failed_after_generated_is_retained(self):
  p,rows,panel,fixed=fixture();events=[]
  def fail(row,k,p,ref,raw):raw({'output_ids':[1]});raise RuntimeError('after generate')
  with self.assertRaisesRegex(RuntimeError,'after generate'):core.run(p,rows,panel,fixed,fail,events.append)
  self.assertEqual([e['event'] for e in events],['job_started','raw_generation','job_failed'])
 def test_natural_or_fixed_wrong_output_retained_and_stops(self):
  p,rows,panel,fixed=fixture();events=[]
  def bad(row,k,protocol,ref,raw):
   rec=execute(row,k,protocol,ref,raw);rec['answer']='mismatch';return rec
  with self.assertRaises(ValueError):core.run(p,rows,panel,fixed,bad,events.append)
  self.assertEqual([x['event'] for x in events],['job_started','raw_generation','job_completed','job_failed'])
if __name__=='__main__':unittest.main()
