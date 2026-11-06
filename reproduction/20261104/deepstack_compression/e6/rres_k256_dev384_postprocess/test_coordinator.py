import importlib.util,json,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('coordinator',P/'coordinator.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
class TestCoordinator(unittest.TestCase):
 def simulate(self,fail=None,collision=None,bad_generation=False):
  with tempfile.TemporaryDirectory() as tmp:
   base=Path(tmp);d=base/'e6/post';d.mkdir(parents=True);run=base/'e6/rres_k256_dev384/monitor_run001';write(run/'terminal.json',{})
   scripts=[];steps=[]
   for name in ('snapshot','formal','independent_result','score','independent_score'):
    worker=base/(name+'.py');worker.write_text(name);scripts.append(worker);mon=base/('mon_'+name);product=base/('product_'+name+'.json');out=product
    steps.append({'name':name,'worker':str(worker),'monitor':str(mon),'product':str(product),'reserved':str(product),'out_arg':str(out),'status':'passed','required':{},'timeout':20,'grace':1,'command':['fake',str(worker),'--out',str(out)]})
   if collision:write(base/('mon_'+collision)/'started.json',{})
   plan={'coordinator_sha256':c.sha(c.__file__),'sources':{str(x):c.sha(x) for x in scripts},'budget_end_unix':time.time()+1000,'steps':steps};write(d/'plan001.json',plan);calls=[]
   def child(command,**kwargs):
    name=Path(command[1]).stem;calls.append(name);step=next(x for x in steps if x['name']==name);mon=Path(step['monitor'])
    write(mon/'started.json',{'cwd':str(base),'worker_pid':123,'command':['fake',step['worker'],'--out',step['out_arg']]})
    write(mon/'terminal.json',{'worker_pid':123,'reason':'completed' if name!=fail else 'worker_failed','worker_returncode':0 if name!=fail else 1,'remaining_process_group':[]})
    if name!=fail:write(step['product'],{'status':'passed'})
    class Child:
     pid=456
     def wait(self,timeout=None):return 1 if name==fail else 0
    return Child()
   def gen():
    if bad_generation:raise ValueError('bad actual container')
    return {'container_id':'synthetic'}
   with patch.object(c,'B',base),patch.object(c,'D',d),patch.object(c,'generation',side_effect=gen),patch.object(c.subprocess,'Popen',side_effect=child),patch('sys.argv',['coordinator','--plan-sha256',c.sha(d/'plan001.json')]),patch('pathlib.Path.cwd',return_value=base):
    if fail or collision or bad_generation:
     with self.assertRaises((RuntimeError,FileExistsError,ValueError)):c.main()
    else:c.main()
   events=[json.loads(x) for x in (d/'audit001.jsonl').read_text().splitlines()]
   return calls,events[-1]['event']
 def test_all_stages(self):self.assertEqual(self.simulate()[0],['snapshot','formal','independent_result','score','independent_score'])
 def test_failed_formal_never_scores(self):self.assertEqual(self.simulate(fail='formal')[0],['snapshot','formal'])
 def test_existing_monitor_never_starts_that_stage(self):self.assertEqual(self.simulate(collision='independent_result')[0],['snapshot','formal'])
 def test_bad_generation_never_starts(self):self.assertEqual(self.simulate(bad_generation=True)[0],[])
 def test_reject_bad_supervisor(self):
  with tempfile.TemporaryDirectory() as t:
   m=Path(t);write(m/'started.json',{'cwd':str(c.B),'command':['python',str(c.B/'x.py')],'worker_pid':1});write(m/'terminal.json',{'worker_pid':1,'reason':'completed','worker_returncode':0,'remaining_process_group':[2]})
   with self.assertRaises(ValueError):c.clean(m,c.B/'x.py')
if __name__=='__main__':unittest.main()
