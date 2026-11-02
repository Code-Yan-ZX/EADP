"""CPU acceptance-gate tests, including decoding real generated panel IDs."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

import repro_full_default as full


class CompletenessGateTests(unittest.TestCase):
    def setUp(self):
        self.questions = [dict(question_id='same-image', image='same-image.jpg', text=s)
                          for s in ['question one', 'question two']]
        self.answers = [dict(question_id=q['question_id'], prompt=q['text'], text='answer')
                        for q in self.questions]
        self.runtime = [dict(position=i, question_id=q['question_id'], image=q['image'],
            prompt_sha256=hashlib.sha256(q['text'].encode()).hexdigest(),
            actual_visual_tokens_retained=637, image_tensor_shape=[1,5,3,336,336],
            output_token_ids=[[1,2]]) for i,q in enumerate(self.questions)]

    def test_distinct_questions_on_same_image_are_valid(self):
        result = full.validate_rows(self.questions, self.answers, self.runtime, expected_n=2)
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['retained_count_histogram'], {637:2})

    def test_subset_cannot_pass_full_denominator(self):
        with self.assertRaises(AssertionError):
            full.validate_rows(self.questions, self.answers, self.runtime)

    def test_truncated_runtime_is_rejected(self):
        with self.assertRaises(AssertionError):
            full.validate_rows(self.questions, self.answers, self.runtime[:1], expected_n=2)

    def test_answer_reordering_is_rejected(self):
        with self.assertRaises(AssertionError):
            full.validate_rows(self.questions, self.answers[::-1], self.runtime, expected_n=2)

    def test_duplicate_answer_is_rejected(self):
        with self.assertRaises(AssertionError):
            full.validate_rows(self.questions, self.answers[:1]*2, self.runtime, expected_n=2)

    def test_wrong_prompt_or_crop_shape_or_count_is_rejected(self):
        for key, value in [('prompt_sha256','bad'), ('image_tensor_shape',[1,1,3,336,336]),
                           ('actual_visual_tokens_retained',637.5)]:
            runtime = copy.deepcopy(self.runtime)
            runtime[1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                full.validate_rows(self.questions, self.answers, runtime, expected_n=2)

    def test_active_alternative_arch_is_rejected_before_import(self):
        with patch.dict(full.os.environ,{'USE_LLAVA_ARCH_HIPRUNE':'1'}):
            with self.assertRaisesRegex(AssertionError,'Alternative pruning arch'):
                full.author_imports()

    def test_actual_panel_ids_decode_to_all_raw_answers_without_cuda(self):
        import torch
        from transformers import AutoTokenizer
        self.assertFalse(torch.cuda.is_initialized())
        runtime = json.loads((full.HERE/'panel64_author_wait/runtime.json').read_text())
        answers = full.rows(full.HERE/'panel64_author_wait/answers.jsonl')
        tokenizer = AutoTokenizer.from_pretrained(full.MODEL, use_fast=False)
        decoded = tokenizer.batch_decode([r['output_token_ids'][0] for r in runtime],
                                         skip_special_tokens=True)
        self.assertEqual(len(decoded),64)
        self.assertEqual([s.strip() for s in decoded],[a['text'] for a in answers])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
