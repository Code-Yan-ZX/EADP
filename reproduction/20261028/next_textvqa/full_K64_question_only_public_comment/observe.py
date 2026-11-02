"""Reuse native K64 post-generation budget observer; observe actual texts hash."""
import hashlib
import importlib.util
import json
from pathlib import Path
import guidance_patch

BASE=Path(__file__).resolve().parents[1]/'full_default_K64_public/observe.py'
BASE_SHA='f5c6b8bac254875b050b4e63b36ae3a905a7eff5f962607e0768567d34975cd2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def install_observer(model,observed,run):
    assert sha(BASE)==BASE_SHA
    spec=importlib.util.spec_from_file_location('frozen_completed_K64_budget_observer',BASE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.install_observer(model,observed,run)
    observed['K64_observer'].update(path=str(Path(__file__).resolve()),sha256=sha(__file__),
        reused_observer_path=str(BASE),reused_observer_sha256=BASE_SHA)
    actual=guidance_patch.EXECUTED_PROOF
    assert actual is not None and {k:actual[k] for k in guidance_patch.proof()}==guidance_patch.proof()
    observed['guidance_entry_transform']=dict(actual)
    questions=[json.loads(s) for s in (Path(__file__).resolve().parents[1]/'author_linked_textvqa_questions.jsonl').read_text().splitlines()]
    references=[json.loads(s) for s in (Path(__file__).resolve().parents[1]/'guidance_question_boundary_exact_token_rows.jsonl').read_text().splitlines()]
    assert len(questions)==len(references)==5000
    original=model.generate
    position=0
    path=run/'guidance_observations.jsonl';path.touch(exist_ok=False)
    def generate(*args,**kwargs):
        nonlocal position
        texts=kwargs['texts']
        returned=original(*args,**kwargs)
        q=questions[position];reference=references[position]
        expected=q['text'].split(guidance_patch.SEPARATOR)[0].replace(guidance_patch.SUFFIX,'')
        assert texts==expected and texts.strip()
        text_sha=hashlib.sha256(texts.encode()).hexdigest()
        assert text_sha==reference['question_only_guidance']['guidance_sha256']
        record=dict(position=position,question_id=q['question_id'],prompt_sha256=hashlib.sha256(q['text'].encode()).hexdigest(),
            guidance_sha256=text_sha,expected_transform='split_before_Reference_OCR_then_remove_exact_answer_suffix',post_generation_observation=True)
        with path.open('a') as stream:stream.write(json.dumps(record)+'\n')
        position+=1
        return returned
    model.generate=generate
