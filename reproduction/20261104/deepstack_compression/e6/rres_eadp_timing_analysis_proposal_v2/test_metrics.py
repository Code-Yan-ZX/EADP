import copy,importlib.util,unittest
from pathlib import Path
import numpy as np
import metrics
B=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('timing_schedule_cpu',B/'e6/rres_eadp_timing_pilot_v4/schedule.py');schedule_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(schedule_module)

def fixtures():
 measured=[];warm=[]
 for task,n in zip(metrics.TASKS,(11,11,10)):
  measured.extend({'task':task,'qid':f'm{i}','group_id':f'{task}-m{i}'} for i in range(n))
  warm.extend({'task':task,'qid':f'w{i}','group_id':f'{task}-w{i}'} for i in range(10))
 procs=schedule_module.build(measured,warm)
 schedule={'measured_inputs':measured,'warmup_inputs':warm,'processes':procs};process_records=[]
 for process in procs:
  output=[]
  for index,req in enumerate(process['requests']):
   j=int(req['qid'][1:]);base=(j+1)*1000000;mult=2 if process['arm']=='rres' and j==0 else 1
   t={'t0_ns':100,'first_ns':100+base*mult,'generate_end_ns':100+2*base*mult,'response_end_ns':100+3*base*mult,'raw_persisted_after_all_endpoints':True,'criteria_calls':2}
   output.append({'event':'completed','index':index,'request':req,'arm':process['arm'],'times':t,'checks':{k:True for k in metrics.CHECKS},'output_ids':[7,2],'generation':{'effective_generation_config':{'eos_token_id':[2]}}})
  process_records.append(output)
 return schedule,process_records

class Tests(unittest.TestCase):
 def test_full_and_fixed_bootstrap(self):
  schedule,records=fixtures();a=metrics.summarize(schedule,records);b=metrics.summarize(schedule,records)
  self.assertEqual(a,b);self.assertEqual(a['counts'],{'images':32,'processes':12,'warmups':60,'measured':320});self.assertEqual(len(a['per_job']),380);self.assertEqual(len(a['per_image']),32)
  task=a['tasks']['DocVQA_VAL']['metrics']['ttft_ms'];self.assertEqual(task['paired_ratio_median'],1.0);self.assertEqual(task['paired_difference_median_ms'],0.0)
 def test_median_of_ratios_not_ratio_of_medians(self):
  schedule,records=fixtures()
  for process,recs in zip(schedule['processes'],records):
   if process['task']!='DocVQA_VAL':continue
   for rec in recs:
    if rec['request']['phase']!='measured':continue
    idx=int(rec['request']['qid'][1:]);base=(idx+1 if process['arm']=='eadp' else (1 if idx<6 else 100))*1000000
    rec['times'].update(first_ns=100+base,generate_end_ns=100+2*base,response_end_ns=100+3*base)
  task=metrics.summarize(schedule,records)['tasks']['DocVQA_VAL']['metrics']['ttft_ms']
  self.assertAlmostEqual(task['paired_ratio_median'],1.0)
  self.assertAlmostEqual(task['rres_median_ms']/task['eadp_median_ms'],1/6)
 def test_missing_duplicate_wrong_process_or_order(self):
  for action in ('missing','duplicate','rep','arm','order','failed','check','badtime'):
   schedule,records=fixtures()
   if action=='missing':records[0].pop()
   if action=='duplicate':records[0][-1]=copy.deepcopy(records[0][-2])
   if action=='rep':records[0][-1]['request']=dict(records[0][-1]['request'],replicate=9)
   if action=='arm':records[0][0]['arm']='rres'
   if action=='order':schedule['processes'][0]['order']=2
   if action=='failed':records[0][0]['event']='failed'
   if action=='check':records[0][0]['checks']['tokens_match']=False
   if action=='badtime':records[0][0]['times']['first_ns']=0
   with self.subTest(action=action),self.assertRaises(ValueError):metrics.summarize(schedule,records)
 def test_repeated_group_rejected(self):
  schedule,records=fixtures();schedule['warmup_inputs'][0]['group_id']=schedule['measured_inputs'][0]['group_id']
  with self.assertRaises(ValueError):metrics.summarize(schedule,records)
 def test_early_non_eos_and_cross_arm_block_rejected(self):
  s,r=fixtures();r[0][0]['output_ids']=[7,8]
  with self.assertRaisesRegex(ValueError,'incomplete natural'):metrics.summarize(s,r)
  s,r=fixtures()
  matching=[p for p in s['processes'] if p['task']=='DocVQA_VAL' and p['arm']=='rres']
  matching[0]['block'],matching[1]['block']=matching[1]['block'],matching[0]['block']
  with self.assertRaisesRegex(ValueError,'different blocks'):metrics.summarize(s,r)
 def test_seed_not_changeable(self):
  s,r=fixtures()
  with self.assertRaises(ValueError):metrics.summarize(s,r,seed=42)
if __name__=='__main__':unittest.main()
