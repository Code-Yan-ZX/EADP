"""Actual Transformers generation loop on a tiny random CPU-only model."""
import unittest
import torch
from transformers import GPT2Config,GPT2LMHeadModel,StoppingCriteriaList
import paired_runner as runner
import generation_capture
class TestHF(unittest.TestCase):
 def test_real_input_ids_and_inputs_embeds_boundaries(self):
  torch.manual_seed(1);model=GPT2LMHeadModel(GPT2Config(vocab_size=31,n_positions=16,n_embd=16,n_layer=1,n_head=2,bos_token_id=1,eos_token_id=None,pad_token_id=0)).eval();model.generation_config.eos_token_id=None;prompt=torch.tensor([[1,3,4]])
  for arm in ('rres','eadp'):
   stored=[]
   def invoke(b):
    kwargs={'input_ids':prompt} if arm=='rres' else {'inputs_embeds':model.transformer.wte(prompt)}
    raw=model.generate(**kwargs,max_new_tokens=2,do_sample=False,stopping_criteria=StoppingCriteriaList([b]));b.generation_returned(raw);return 'decoded'
   with torch.inference_mode():response,b=runner.measure(invoke,lambda:None,persist=lambda x:stored.append(x.cpu().tolist()[0]))
   tokens=runner.normalize(arm,stored[0],prompt.tolist()[0]);self.assertEqual(len(tokens),2);self.assertEqual(b['criteria_calls'],2);self.assertEqual(b['first_criteria_length'],4 if arm=='rres' else 1);self.assertEqual(response,'decoded')
  self.assertFalse(torch.cuda.is_initialized())
 def test_actual_effective_generation_configuration(self):
  model=GPT2LMHeadModel(GPT2Config(vocab_size=31,n_positions=16,n_embd=16,n_layer=1,n_head=2,eos_token_id=2,pad_token_id=0)).eval()
  prompt=torch.tensor([[1,3,4]]);kwargs={'input_ids':prompt,'max_new_tokens':2,'do_sample':False,'top_k':23,'temperature':1.0,'top_p':1.0,'repetition_penalty':1.0}
  with generation_capture.capture(model,kwargs) as capture:
   model.generate(**kwargs)
  self.assertEqual(capture['explicit_kwargs']['top_k'],23)
  self.assertEqual(capture['effective_generation_config']['top_k'],23)
  self.assertEqual(capture['effective_generation_config']['max_new_tokens'],2)
  self.assertEqual(capture['effective_generation_config']['eos_token_id'],2)
  self.assertEqual(capture['explicit_kwargs']['input_ids']['shape'],[1,3])
  self.assertNotIn('_prepare_generation_config',model.__dict__)
  with self.assertRaises(RuntimeError):
   with generation_capture.capture(model,kwargs): pass
  self.assertNotIn('_prepare_generation_config',model.__dict__)
if __name__=='__main__':unittest.main()
