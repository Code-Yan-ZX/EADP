import copy
import unittest
from contract import validate_arm, validate_modes, validate_timing


def arm(mode='C-ON', k=256):
    ids=list(range(k))
    return {'task':'DocVQA_VAL','qid':'1','group_id':'g','image_sha256':'i','message_sha256':'m',
            'processor_sha256':'p','mode':mode,'k':k,'n_input_visual':1024,'n_retained_visual':k,
            'ordered_ids':ids,'spatial_ids':ids.copy(),'selection_mode':'keep_all' if k==1024 else 'eadp_greedy',
            'scorer_calls':0 if k==1024 else 1,'greedy_calls':0 if k==1024 else 1,
            'main_merger_calls':1,'vision_blocks_executed':27,'vision_block_count':27,
            'ds_original_merger_calls':[0,0,0] if mode=='C-SKIP' else [1,1,1],
            'ds_injected_rows':[k,k,k] if mode=='C-ON' else [0,0,0],
            'position_mode':'P3','full_delta':-20,'decode_delta':-20+1024-k,
            'vision_mask_count':k,'original_length':1050,'prefill_length':26+k,
            'visual_sha256':'v','prefill_embed_sha256':'e','attention_mask_sha256':'a','position_sha256':'p',
            'output_ids':[42,43],'first_logits_argmax':42,'logits_comparison_passed':True,
            'eos':True,'truncated':False}


def timing():
    return {'batch':1,'gpu_exclusive':True,'prompt_callback_index':0,'first_token_callback_index':1,
            'callback_kind':'generated_token_after_cpu_transfer','stream_output_ids':[42,43],
            'plain_output_ids':[42,43],'callback_token_id':42,'ttft_scope':'before_image_read_to_first_token_cpu',
            'total_scope':'before_image_read_to_end_no_streamer','observer_sync_inside_ttft':'first_token_transfer_only',
            'ttft_s':0.3,'unstreamed_total_s':0.6,'warmup_complete':True,'all_gpu_processes_recorded':True}


class E4ContractTests(unittest.TestCase):
    def test_all_budget_endpoints_and_three_modes(self):
        for k in (128,256,512,1024):
            for mode in ('C-ON','C-DROP','C-SKIP'):
                with self.subTest(k=k,mode=mode):self.assertTrue(validate_arm(arm(mode,k)))
        self.assertTrue(validate_modes([arm(m) for m in ('C-ON','C-DROP','C-SKIP')]))

    def test_drop_is_not_merger_skip_and_backbone_must_run(self):
        x=arm('C-DROP');x['ds_original_merger_calls']=[0,0,0]
        with self.assertRaises(ValueError):validate_arm(x)
        x=arm('C-SKIP');x['vision_blocks_executed']=24
        with self.assertRaises(ValueError):validate_arm(x)

    def test_keep_all_is_not_facility(self):
        x=arm(k=1024);x['greedy_calls']=1
        with self.assertRaises(ValueError):validate_arm(x)
        x=arm(k=1024);x['ordered_ids']=list(reversed(x['ordered_ids']))
        with self.assertRaises(ValueError):validate_arm(x)

    def test_same_image_and_selector_in_each_mode(self):
        x=[arm(m) for m in ('C-ON','C-DROP','C-SKIP')]
        x[2]['image_sha256']='other'
        with self.assertRaises(ValueError):validate_modes(x)
        x=[arm(m) for m in ('C-ON','C-DROP','C-SKIP')]
        x[2]['output_ids']=[99]
        with self.assertRaises(ValueError):validate_modes(x)

    def test_p3_full_delta_and_ds_mask(self):
        x=arm();x['decode_delta']+=1
        with self.assertRaises(ValueError):validate_arm(x)
        x=arm('C-ON');x['ds_injected_rows']=[256,256,255]
        with self.assertRaises(ValueError):validate_arm(x)

    def test_prompt_callback_is_not_first_generated_token(self):
        x=timing();self.assertTrue(validate_timing(x))
        x['first_token_callback_index']=0
        with self.assertRaises(ValueError):validate_timing(x)
        x=timing();x['callback_token_id']=99
        with self.assertRaises(ValueError):validate_timing(x)

    def test_streamed_total_and_different_outputs_are_rejected(self):
        x=timing();x['total_scope']='streamed_total'
        with self.assertRaises(ValueError):validate_timing(x)
        x=timing();x['plain_output_ids']=[42,44]
        with self.assertRaises(ValueError):validate_timing(x)
        x=timing();x['observer_sync_inside_ttft']='each_layer'
        with self.assertRaises(ValueError):validate_timing(x)


if __name__=='__main__':unittest.main()
