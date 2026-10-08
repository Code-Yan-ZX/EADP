"""Run the tested TextVQA wrapper unchanged and record ID-aligned token counts.

The adapter observes the existing model.generate return value. It returns the
same object to the wrapper and changes no model, prompt, generation setting or
CUDA stream. Use the original wrapper's command line arguments.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import runpy
import sys


def option(name):
    return sys.argv[sys.argv.index(name) + 1]


def main():
    from llava.model import builder

    wrapper = Path(__file__).with_name('llava_eval_arm_model_vqa.py')
    questions_path = Path(option('--question-file'))
    questions = [json.loads(line) for line in questions_path.open()]
    wrapper_sha256 = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    input_sha256 = hashlib.sha256(questions_path.read_bytes()).hexdigest()
    if '--num-chunks' in sys.argv and int(option('--num-chunks')) != 1:
        raise ValueError('Runtime trace currently requires one complete shard')
    if '--chunk-idx' in sys.argv and int(option('--chunk-idx')) != 0:
        raise ValueError('Runtime trace currently requires chunk index zero')
    answers = Path(option('--answers-file'))
    trace_path = Path(str(answers) + '.runtime.jsonl')
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace = trace_path.open('x')
    loader = builder.load_pretrained_model
    runtime = dict(position=0)

    def traced_loader(*args, **kwargs):
        loaded = loader(*args, **kwargs)
        model = loaded[1]
        generate = model.generate

        def traced_generate(*generation_args, **generation_kwargs):
            returned = generate(*generation_args, **generation_kwargs)
            if not isinstance(returned, tuple) or len(returned) != 2:
                raise RuntimeError('Expected original generate output/count tuple')
            row = questions[runtime['position']]
            images = generation_kwargs.get('images')
            trace.write(json.dumps(dict(
                question_id=row['question_id'], image=row['image'],
                question_position=runtime['position'],
                prompt_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                actual_visual_tokens_retained=int(returned[1]),
                visual_token_budget_parameter=int(option('--visual_token_num')),
                image_tensor_shape=list(images.shape) if images is not None else None,
                image_sizes=generation_kwargs.get('image_sizes'),
                source='unchanged model.generate returned visual-token count',
                source_wrapper_sha256=wrapper_sha256,
                input_sha256=input_sha256,
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
            )) + '\n')
            trace.flush()
            runtime['position'] += 1
            return returned

        model.generate = traced_generate
        return loaded

    builder.load_pretrained_model = traced_loader
    try:
        runpy.run_path(str(wrapper), run_name='__main__')
        if runtime['position'] != len(questions):
            raise RuntimeError('Incomplete runtime trace')
    finally:
        trace.close()


if __name__ == '__main__':
    main()
