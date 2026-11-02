"""Complete8910 input/output audit, then the existing qualified POPE scorer.

CPU tokenizer use may import Torch; no model, CUDA context or selector is loaded.
The existing parser/category macro and author CLI are reused without changes.
"""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import common as c


def rows(path):
    raw = Path(path).read_bytes()
    assert raw.endswith(b'\n'), ('Incomplete last record', str(path))
    return [json.loads(line) for line in raw.splitlines()]


def audit_loaded_runtime(plan,loaded):
    assert [loaded['actual_alpha'],loaded['actual_beta'],loaded['actual_K']] == [1.,2.,32]
    assert loaded['model_dtype'] == 'torch.float16' and loaded['backend'] == 'sdpa'
    assert loaded['component_dtypes'] == plan['expected_component_dtypes']
    assert loaded['torch'] == '2.1.2+cu118' and loaded['transformers'] == '4.37.2'
    assert loaded['CUDA'] == '11.8' and loaded['GPU'] == 'NVIDIA A40'
    assert loaded['plan_sha256'] == c.sha(c.PLAN) and loaded['worker_sha256'] == c.sha(c.HERE/'worker.py')
    assert loaded['unmodified_fresh_eval_model'] is True and plan['generation_entry_body_unchanged'] is True
    assert loaded['entry_source_sha256'] == c.sha(plan['entry'])
    assert Path(loaded['original_eval_model_code_filename']) == Path(plan['entry'])
    assert loaded['original_eval_model_AST_sha256'] == plan['eval_model_AST_sha256']
    assert loaded['same_process_compiled_code_equal'] is True
    assert loaded['actual_eval_model_structural_fingerprint'] == plan['eval_model_structural_fingerprint']
    assert loaded['source_sha256'] == plan['source_sha256']
    assert set(loaded['bound_methods']) == {'generate','encode_images','greed_select','prepare_inputs_labels_for_multimodal'}
    for entry in loaded['bound_methods'].values():
        assert c.sha(entry['path']) == entry['sha256'] == plan['source_sha256'][entry['path']]
    observer = loaded['observer']
    assert Path(observer['path']) == c.HERE/'observe.py' and observer['sha256'] == c.sha(c.HERE/'observe.py')
    assert observer['pre_generation_tensor_reads'] is False and observer['native_numerical_operations_replaced'] is False
    assert loaded['backend_flags']['matmul_TF32'] is False
    args = c.read(c.OUT/'native_args.json')
    expected = dict(model_path=plan['model_path'],model_base=None,image_folder=plan['image_folder'],
        question_file=plan['question_file'],answers_file=str(c.OUT/'predictions.jsonl'),conv_mode='vicuna_v1',
        num_chunks=1,chunk_idx=0,temperature=0.,top_p=None,num_beams=1,max_new_tokens=128,
        visual_token_num=32,alpha=1.,beta=2.)
    assert args == expected


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    score_dir = c.OUT/'official_and_reference_score'
    report_path = c.OUT/'independent_verification.json'
    assert not score_dir.exists() and not report_path.exists(), 'Do not repeat a scoring attempt'
    assert not (c.HERE/'failure.json').exists() and not (c.HERE/'worker_failure.json').exists()
    state = c.read(c.HERE/'state.json')
    assert state['status'] == 'native_exited' and state['native_exit_code'] == 0
    from run import identity,gpu_pids
    assert identity(state['native_worker']['pid']) != state['native_worker']
    assert state['native_worker']['pid'] not in gpu_pids()
    plan = c.read(c.PLAN); c.check(plan)
    claim = c.read(c.HERE/'launch.claim.json'); own = c.read(c.OUT/'process_identity.json')
    assert claim['controller'] == state['controller']
    assert {key:own[key] for key in ['pid','start_ticks','boot_id']} == state['native_worker']
    assert own['plan_sha256'] == claim['plan_sha256'] == state['plan_sha256'] == c.sha(c.PLAN)
    assert own['allocation_note_sha256'] == claim['allocation_note_sha256'] == state['allocation_note_sha256']
    complete = c.read(c.OUT/'native_complete.json')
    assert complete['success'] is True and complete['n_predictions'] == complete['n_runtime_records'] == 8910
    assert complete['panel_beta2_anchor_checks'] == 16 and complete['plan_sha256'] == c.sha(c.PLAN)
    assert complete['GT_read'] is False and complete['scoring_started'] is False
    assert complete['all8910_generation_completed_before_scoring'] is True
    assert complete['raw_native_outputs_saved_before_anchor_checks'] is True
    assert complete['post_generation_tensor_reads_only'] is True
    for key,name in [('predictions','predictions.jsonl'),('runtime_observations','runtime_observations.jsonl')]:
        entry = complete[key]
        assert Path(entry['path']) == c.OUT/name and c.sha(entry['path']) == entry['sha256']
        assert Path(entry['path']).stat().st_size == entry['bytes']
    questions = rows(plan['question_file']); predictions = rows(c.OUT/'predictions.jsonl')
    runtime = rows(c.OUT/'runtime_observations.jsonl')
    assert len(questions) == len(predictions) == len(runtime) == 8910
    assert len({q['question_id'] for q in questions}) == len({p['question_id'] for p in predictions}) == 8910
    assert [p['question_id'] for p in predictions] == [q['question_id'] for q in questions]
    assert [p['prompt'] for p in predictions] == [q['text'] for q in questions]
    loaded = c.read(c.OUT/'loaded_runtime.json'); audit_loaded_runtime(plan,loaded)
    patch = c.read(c.OUT/'wait_patch_proof.json')
    assert patch['runtime_patch_installed'] is True and patch['restoration_ast_identical'] is True
    assert patch['source_files_modified'] is False and patch['record_stream_added'] is False
    assert {key:patch[key] for key in plan['wait_patch_prepared_proof']} == plan['wait_patch_prepared_proof']
    assert patch['inserted_nodes'] == ['caller_stream = torch.cuda.current_stream(device=self.device)',
        'image_stream.wait_stream(caller_stream)','text_stream.wait_stream(caller_stream)']
    assert patch['production_source_sha256'] == plan['source_sha256'][patch['production_source']]
    anchor_file = c.read(c.HERE/'beta2_panel_anchors.json')
    assert anchor_file['accepted_beta2_native16'] is True and anchor_file['GT_or_accuracy_used'] is False
    assert anchor_file['fixed_original_panel'] is True and anchor_file['acceptance_not_based_on_correctness_or_score'] is True
    anchors = {(str(a['question_id']),a['prompt']):a for a in anchor_file['rows']}
    assert len(anchors) == 16
    # Full raw/runtime identity, observation and fixed-panel gates precede any
    # tokenizer use, GT-label payload access or score invocation.
    matched = []
    for position,(question,prediction,observation) in enumerate(zip(questions,predictions,runtime)):
        assert isinstance(prediction['text'],str) and prediction['text'] == observation['text']
        assert prediction['model_id'] == 'llava-v1.5-7b'
        c.check_observation(observation,question,position)
        key = (str(question['question_id']),question['text'])
        if key in anchors:
            anchor = anchors[key]; c.check_anchor(observation,anchor['expected'])
            accepted = anchor['accepted_panel_record']
            assert c.sha(accepted['path']) == accepted['sha256']
            panel_record = c.read(accepted['path'])
            assert panel_record['actual_alpha_beta_K'] == [1.,2.,32] and panel_record['checks_status'] == 'all_passed'
            assert observation['output_token_ids'] == panel_record['output_token_ids'][0]
            assert observation['text'] == panel_record['text']
            assert observation['input_ids_sha256'] == panel_record['actual_LLM_input_ids_sha256']
            assert observation['quotas'] == panel_record['selection']['greedy_quotas']
            assert observation['effective_ordered_prefix'] == panel_record['selection']['greedy_effective_prefixes'][0]
            assert observation['effective_mask_indices'] == panel_record['selection']['effective_mask_indices'][0]
            matched.append(dict(position=position,question_id=question['question_id'],accepted_panel_record=accepted))
    assert len(matched) == len({x['question_id'] for x in matched}) == 16
    sys.path.insert(0,plan['llava_root'])
    from transformers import AutoTokenizer
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import DEFAULT_IMAGE_TOKEN,IMAGE_TOKEN_INDEX
    import torch
    assert not torch.cuda.is_initialized() and IMAGE_TOKEN_INDEX == -200
    tokenizer = AutoTokenizer.from_pretrained(plan['model_path'],use_fast=False,local_files_only=True)
    assert type(tokenizer).__name__ == loaded['tokenizer_class']
    for module_name in ['llava.conversation','llava.mm_utils','llava.constants']:
        path = Path(sys.modules[module_name].__file__).resolve()
        assert path == (Path(plan['llava_root'])/Path(*module_name.split('.'))).with_suffix('.py').resolve()
        assert c.sha(path) == plan['source_sha256'][str(path)], str(path)
    assert Path(inspect.getfile(tokenizer_image_token)).resolve() == Path(sys.modules['llava.mm_utils'].__file__).resolve()
    model_config = c.read(Path(plan['model_path'])/'config.json')
    generation = c.read(Path(plan['model_path'])/'generation_config.json')
    assert model_config['mm_use_im_start_end'] is False
    assert model_config['bos_token_id'] == generation['bos_token_id'] == loaded['bos_token_id'] == 1
    assert model_config['eos_token_id'] == generation['eos_token_id'] == loaded['eos_token_id'] == 2
    for question,observation in zip(questions,runtime):
        conv = conv_templates['vicuna_v1'].copy()
        conv.append_message(conv.roles[0],DEFAULT_IMAGE_TOKEN+'\n'+question['text'])
        conv.append_message(conv.roles[1],None)
        expected_ids = tokenizer_image_token(conv.get_prompt(),tokenizer,IMAGE_TOKEN_INDEX)
        assert expected_ids == observation['input_token_ids']
        assert c.int64_sha(expected_ids) == observation['input_ids_sha256']
        assert all(0 <= token < model_config['vocab_size'] for token in observation['output_token_ids'])
    decoded = tokenizer.batch_decode([r['output_token_ids'] for r in runtime],skip_special_tokens=True)
    assert [text.strip() for text in decoded] == [p['text'] for p in predictions]
    assert [text.strip() for text in decoded] == [r['text'] for r in runtime]
    assert not torch.cuda.is_initialized()
    c.check(plan)
    # First access to pinned label payloads and first score call: all8910
    # generated rows and CPU input/output/panel checks have now passed.
    for path,digest in plan['score_only_sha256'].items():
        assert c.sha(path) == digest, path
    helper_path = c.POPE/'verify_complete_predictions.py'
    assert c.sha(helper_path) == plan['score_only_sha256'][str(helper_path)]
    spec = importlib.util.spec_from_file_location('qualified_existing_pope_score_helper',helper_path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    assert helper.QPATH == Path(plan['question_file'])
    assert helper.SCORER == Path(plan['llava_root'])/'llava/eval/eval_pope.py'
    result = helper.verify(c.OUT/'predictions.jsonl',score_dir)
    assert result['complete'] is True and result['n_predictions'] == 8910 and result['official_scorer_exit_code'] == 0
    assert result['all_question_ids_order_verified'] and result['all_raw_prompts_verified'] and result['all_pinned_image_question_positions_verified']
    assert {name:row['n'] for name,row in result['categories'].items()} == {'random':2910,'popular':3000,'adversarial':3000}
    assert abs(result['official_macro_f1_percent']-result['independent_macro_f1_percent']) < 1e-12
    assert result['model_ids'] == {'llava-v1.5-7b':8910}
    assert result['predictions_sha256'] == c.sha(c.OUT/'predictions.jsonl')
    assert result['canonical_questions_sha256'] == c.sha(plan['question_file'])
    c.check(plan); assert not torch.cuda.is_initialized()
    for path,digest in plan['score_only_sha256'].items():
        assert c.sha(path) == digest, path
    names = ['predictions.jsonl','runtime_observations.jsonl','native_complete.json','loaded_runtime.json',
             'native_args.json','process_identity.json','wait_patch_proof.json']
    report = dict(success=True,complete=True,n_predictions=8910,actual_guidance_checks=8910,input_ID_checks=8910,
        output_redecode_checks=8910,budget_mask_prefix_checks=8910,panel_beta2_anchor_checks=16,
        official_macro_f1_percent=result['official_macro_f1_percent'],independent_macro_f1_percent=result['independent_macro_f1_percent'],
        categories=result['categories'],same_input_repeated_groups=result['same_input_repeated_groups'],
        same_input_yes_no_conflict_groups=result['same_input_yes_no_conflict_groups'],
        empty_answers_preserved_in_official_denominator=result['empty_answers_preserved_in_official_denominator'],
        model_ids=result['model_ids'],full_denominator_retained=True,score_direction_gate=False,paper_argv_confirmed=False,
        plan_sha256=c.sha(c.PLAN),native_complete_sha256=c.sha(c.OUT/'native_complete.json'),
        predictions_sha256=result['predictions_sha256'],official_scorer_exit_code=0,
        official_score_log=c.describe(score_dir/'official_score.log'),
        qualified_metric_verification=c.describe(score_dir/'full_prediction_verification.json'),
        qualified_score_helper=c.describe(helper_path),accepted_beta2_panel_anchors=matched,
        artifacts={name:c.describe(c.OUT/name) for name in names},
        actual_native_exit=dict(original=state['native_worker'],original_exited=True,GPU_pid_absent=True),
        root_actual_controller_exit_required=True,CPU_only=True,torch_CPU_imported=True,CUDA_initialized=False,model_loaded=False,
        tokenizer_only_loaded=True,selector_executed=False,all_input_output_gates_before_this_verifier_GT_or_scoring=True,
        EOS_count=sum(r['EOS_present'] for r in runtime),non_EOS_max128_count=sum(not r['EOS_present'] and r['new_tokens']==128 for r in runtime),
        all_actual_visual_counts=32,full_feature_or_prepared_embedding_arrays_saved=False,
        limits='One complete alpha1/beta2 single control. Existing author parser,full random2910/popular3000/adversarial3000 macro and all outcomes retained. CPU tokenizer reconstructs original conversation IDs and redecodes outputs; no model or GPU. Full-run features/gains/prepared embeddings were not stored. Paper argv and reproduction success are not inferred.')
    assert not report_path.exists()
    with report_path.open('x') as stream:
        json.dump(report,stream,indent=2); stream.write('\n')
    print(json.dumps({key:report[key] for key in ['success','n_predictions','official_macro_f1_percent','independent_macro_f1_percent','panel_beta2_anchor_checks']}))


if __name__ == '__main__':
    main()
