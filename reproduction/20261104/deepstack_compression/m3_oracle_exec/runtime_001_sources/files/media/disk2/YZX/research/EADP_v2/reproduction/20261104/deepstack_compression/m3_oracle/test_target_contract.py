import unittest
from target_contract import encode_first_reference

class Tokenizer:
    def __init__(self): self.calls = []
    def encode(self, text, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        return list(range(len(text)))

class TargetTests(unittest.TestCase):
    def test_first_reference_is_verbatim_and_no_eos_is_added(self):
        tokenizer = Tokenizer()
        result = encode_first_reference('DocVQA_VAL', {'answers':[' A ', 'alternate']}, tokenizer)
        self.assertEqual(tokenizer.calls, [(' A ', False)])
        self.assertEqual(result['token_ids'], [0, 1, 2])
        self.assertFalse(result['truncated'])
        self.assertEqual(encode_first_reference('AI2D_TEST', {'answer':'B'}, tokenizer)['token_ids'], [0])

    def test_limit_and_no_fallback(self):
        tokenizer = Tokenizer()
        result = encode_first_reference('TextVQA_VAL', {'answers':['x'*130]}, tokenizer)
        self.assertEqual(len(result['token_ids']), 128)
        self.assertEqual(result['original_token_count'], 130)
        self.assertTrue(result['truncated'])
        for task, truth in (('DocVQA_VAL', {'answers':[' ', 'valid']}), ('TextVQA_VAL', {'answers':[]}),
                            ('AI2D_TEST', {'answer':''}), ('UNKNOWN', {'answers':['x']})):
            with self.assertRaises(ValueError): encode_first_reference(task, truth, tokenizer)

if __name__ == '__main__':
    unittest.main()
