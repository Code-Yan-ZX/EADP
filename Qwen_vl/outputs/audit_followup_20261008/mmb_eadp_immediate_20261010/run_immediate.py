"""One user-authorized EADP MMB-EN run, reusing the frozen runtime/score validator."""
import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path('/media/disk2/YZX/research/EADP_amp')
BASE = ROOT / 'Qwen_vl/outputs/audit_followup_20261008'
OUT = Path(__file__).resolve().parent
FROZEN = BASE / 'post_batch_followup_20261009/registered_followup/followup_worker.py'
JOB = 'v15_mmben_E128'


def worker():
    spec = importlib.util.spec_from_file_location('immediate_frozen_worker', FROZEN)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    loaded.OUT = OUT
    loaded.INVENTORY = OUT / 'inventory.json'
    loaded.job_specs = lambda: json.loads(loaded.INVENTORY.read_text())['specs']
    return loaded


def prepare(w):
    OUT.mkdir(parents=True, exist_ok=True)
    original = json.loads((BASE / 'streamwait_continuation_20261008/next_mmben_LRMAIN025_streamwait_attempt2.control.protocol.json').read_text())
    parameters = dict(original['parameters'], model_path='/media/disk2/YZX/doct/FastV/llava-v1.5-7b', anchorzip=False, lang='en')
    answers = ROOT / 'LLaVA/playground/data/eval/anchorzip_p3/mmben/LRMAIN025.jsonl'
    origin = copy.deepcopy(original['origin'])
    origin.update(model='v15', arm='EADP', jobid=JOB, original_prediction=str(answers), original_prediction_sha256=w.sha(answers), parameters=parameters)
    command = [w.PYTHON, str(ROOT / 'LLaVA/llava/eval/model_vqa_mmbench.py'),
               '--model-path', parameters['model_path'], '--question-file', parameters['question_file'],
               '--answers-file', str(OUT / (JOB + '_streamwait.jsonl')), '--visual_token_num', '128',
               '--beta', '2.0', '--alpha', '0.5', '--single-pred-prompt', '--temperature', '0',
               '--conv-mode', 'vicuna_v1', '--num-chunks', '1', '--chunk-idx', '0', '--num_beams', '1', '--lang', 'en']
    origin['selected_generate_command'] = command
    origin['historical_reference_only'] = 'Original AnchorZip supplies question/image identity and configuration only; this is a new EADP run, with stream wait and no AnchorZip installation.'
    spec = dict(key=JOB, job_id=JOB, model='v15', task='mmben', method='EADP', budget=128, n=4876,
                parameters=parameters, origin=origin, declared_generate_command=command,
                min_free_memory_mib=22500, exclusive_gpu=True)
    w.dump(OUT / 'inventory.json', {'specs': {JOB: spec}, 'authorization': 'User explicitly requested one fastest complete EADP MMB reproduction, waiting for its result.', 'only_one_model_run': True})
    w.dump(OUT / 'POLICY.json', {'method': 'EADP', 'dataset': 'MMBench_DEV_EN_V11', 'n': 4876, 'groups': 1292,
           'model': 'LLaVA-1.5-7B', 'budget': 128, 'alpha': 0.5, 'beta': 2.0, 'lang': 'en',
           'max_new_tokens': 1024, 'temperature': 0, 'single_pred_prompt': True, 'stream_wait': True,
           'math_changes': False, 'frozen_worker': str(FROZEN), 'frozen_worker_sha256': w.sha(FROZEN),
           'comparison_limit': 'Existing v15 AnchorZip MMB-EN answers precede stream-wait repair. This EADP run diagnoses a same-dataset/budget baseline; strict paired method attribution requires both methods with the same execution repair. Paper dataset/argv remain unverified.'})
    protocol, paths = w.prepare(JOB)
    protocol['generate_command'] = [w.PYTHON, str(Path(__file__).resolve()), '--run']
    protocol['source_and_input_sha256'][str(Path(__file__).resolve())] = w.sha(__file__)
    protocol['comparison_limit'] = json.loads((OUT / 'POLICY.json').read_text())['comparison_limit']
    protocol['authorization'] = 'Immediate user request, one complete EADP MMB-EN run; older blanket baseline completion remains cancelled.'
    w.dump(paths['control_protocol'], protocol)
    print(json.dumps({'prepared': True, 'job': JOB, 'n': 4876, 'generate_command': protocol['generate_command']}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--validate', action='store_true')
    args = ap.parse_args()
    if sum([args.prepare, args.run, args.validate]) != 1:
        ap.error('Choose exactly one action')
    w = worker()
    if args.prepare:
        prepare(w)
        return
    paths = w.paths(JOB)
    protocol = json.loads(paths['control_protocol'].read_text())
    if args.run:
        w.run(protocol, paths)
    else:
        result = w.validate_finished(protocol)
        w.dump(OUT / 'validation.json', {'success': True, 'n': result['score']['n'], 'score': result['score'], 'files': result['files']})
        print(json.dumps({'success': True, 'value': result['score']['value'], 'n': result['score']['n']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
