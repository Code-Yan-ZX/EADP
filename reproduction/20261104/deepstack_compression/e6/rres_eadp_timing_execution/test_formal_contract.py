import copy
import unittest
import formal_contract as c


def fixture(arm='rres'):
    item = {'phase': 'measured', 'replicate': 0, 'task': 'T', 'qid': '1', 'group_id': 'g'}
    job = {'arm': arm, 'requests': [item]}
    key = (arm, 'T', '1')
    output = [8, 9]
    raw = [2, 3, *output] if arm == 'rres' else output
    scalars = {'do_sample': False, 'max_new_tokens': 2048}
    reference = {'output_ids': output, 'answer': 'a', 'processor': {'x': 'hash'}, 'generation_scalars': scalars, 'eos_ids': [4, 5]}
    clock = {'t0_ns': 10, 'first_ns': 20, 'generate_end_ns': 30, 'response_end_ns': 40, 'first_criteria_length': 3 if arm == 'rres' else 1, 'criteria_calls': 2, 'raw_persisted_after_all_endpoints': True}
    generation = {'prepare_calls': 1, 'explicit_kwargs': scalars.copy(), 'effective_generation_config': {**scalars, 'eos_token_id': [4, 5], 'num_beams': 1, 'num_return_sequences': 1, 'use_cache': True}}
    start = {'event': 'started', 'index': 0, 'request': item}
    done = {'event': 'completed', 'index': 0, 'arm': arm, 'request': item, 'output_ids': output, 'response': 'a', 'processor': reference['processor'], 'image_sha256': 'img', 'times': clock, 'generation': generation, 'checks': dict.fromkeys(c.CHECKS, True)}
    return job, [start, done], {key: reference}, {0: {'arm': arm, 'request': item, 'full_raw_sequence': [raw]}}, {key: raw}, {('T', '1'): {'image_sha256': 'img'}}


class Tests(unittest.TestCase):
    def test_each_arm_and_denominator(self):
        for arm in ('eadp', 'rres'):
            self.assertEqual(c.check_job(*fixture(arm))['measured'], 1)
    def test_same_arm_raw_and_processor_tamper(self):
        for mutate in ('raw', 'processor', 'image', 'check', 'request'):
            args = copy.deepcopy(fixture())
            if mutate == 'raw': args[3][0]['full_raw_sequence'] = [[99, 3, 8, 9]]
            if mutate == 'processor': args[1][1]['processor'] = {'x': 'other'}
            if mutate == 'image': args[1][1]['image_sha256'] = 'other'
            if mutate == 'check': args[1][1]['checks']['tokens_match'] = False
            if mutate == 'request': args[1][1]['request'] = {**args[1][1]['request'], 'qid': 'wrong'}
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): c.check_job(*args)
    def test_boundaries_and_actual_config_reject(self):
        for field in ('first', 'length', 'calls', 'eos', 'explicit', 'beam', 'return', 'cache'):
            args = copy.deepcopy(fixture())
            row = args[1][1]
            if field == 'first': row['times']['first_ns'] = 10
            if field == 'length': row['times']['first_criteria_length'] = 1
            if field == 'calls': row['times']['criteria_calls'] = 1
            if field == 'eos': row['generation']['effective_generation_config']['eos_token_id'] = [0]
            if field == 'explicit': row['generation']['explicit_kwargs']['do_sample'] = True
            if field == 'beam': row['generation']['effective_generation_config']['num_beams'] = 2
            if field == 'return': row['generation']['effective_generation_config']['num_return_sequences'] = 2
            if field == 'cache': row['generation']['effective_generation_config']['use_cache'] = False
            with self.subTest(field=field), self.assertRaises(ValueError): c.check_job(*args)
    def test_query_overlap_and_unknown(self):
        ns = 'time:[1]'; t = {'t0_ns': 10, 'response_end_ns': 20}
        w = lambda a,b,n=ns: {'start_monotonic_ns': a, 'end_monotonic_ns': b, 'time_namespace': n}
        self.assertEqual(c.known_overlaps(t, ns, [w(1,10),w(11,12),w(20,21),w(21,22)]), [1,2])
        with self.assertRaises(ValueError): c.known_overlaps(t, ns, [w(1,2,'time:[2]')])
        with self.assertRaises(ValueError): c.known_overlaps(t, ns, [w(1,1)])


if __name__ == '__main__': unittest.main()
