import json,unittest
from pathlib import Path
from p3_reconstruct import reconstruct
B=Path(__file__).resolve().parents[2];S=B/'e6/pixelprune_t020_smoke/run_smoke001/observations';MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
class Tests(unittest.TestCase):
 def test_known_simple_mapping(self):
  x=reconstruct([7,8]+[99]*1024+[6],99,8,77,[0,31,32,1023]);self.assertEqual(x['position_ids'],[[[0,1,2,2,2,2,34]],[[0,1,2,2,3,33,34]],[[0,1,2,33,2,33,34]]]);self.assertEqual(x['rope_delta'],[[-992]])
 def test_reject_invalid_or_noncontiguous(self):
  for ids,selected in (([8]+[99]*1023, [0]),([8]+[99]*1024,[3,3]),([8]+[99]*1024+[77],[0])):
   with self.assertRaises(ValueError):reconstruct(ids,99,8,77,selected)
 def test_three_previously_accepted_smoke_records(self):
  config=json.loads((MODEL/'config.json').read_text())
  for task,qid in (('DocVQA_VAL','16539'),('TextVQA_VAL','39421'),('AI2D_TEST','368100')):
   with self.subTest(task=task):
    obs=json.loads((S/(task+'_'+qid+'.json')).read_text());raw=json.loads((S/(task+'_'+qid+'_generated.json')).read_text());ids=raw['full_sequence_ids'][:raw['prompt_length']]
    got=reconstruct(ids,config['image_token_id'],config['vision_start_token_id'],config['video_token_id'],obs['selected_indices'])
    self.assertEqual(got['position_ids'],obs['trace'][0]['position_ids']);self.assertEqual(got['rope_delta'],obs['original_rope_delta']);self.assertEqual(got['original_max_position'],obs['original_max_position'])
if __name__=='__main__':unittest.main(verbosity=2)
