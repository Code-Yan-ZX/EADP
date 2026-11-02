"""Freeze exactly one full8910 beta2 hypothesis after accepted native32 evidence."""
import ast
import marshal
from pathlib import Path
import sys
import common as c


def prepare():
    assert not c.PLAN.exists() and not c.OUT.exists() and not (c.HERE/'launch.claim.json').exists()
    c.check_predecessor()
    panel=c.read(c.PANEL/'plan.json')
    baseline=c.read(c.POPE/'run03_single_alpha05_beta1_wait_k32/run_plan.json')
    # The qualified old full guard already binds every model payload and image
    # to its prior actual SHA. Require those exact current stats, not a fresh
    # unqualified snapshot of potentially changed payloads.
    for group in ['model_stat_guards','image_stat_guards']:
        for path,guard in baseline[group].items():
            assert c.stat(path)=={k:v for k,v in guard.items() if k!='prior_verified_sha256'},path
    canonical=c.POPE/'inputs/llava_pope_test.jsonl'
    questions=[__import__('json').loads(line) for line in canonical.read_text().splitlines()]
    assert len(questions)==len({q['question_id'] for q in questions})==8910
    rows=[];panel_references=[]
    for i,p in enumerate(panel['panel']):
        file=c.PANEL/'execution'/('beta2_%02d.json'%i);record=c.read(file)
        assert record['checks_status']=='all_passed' and record['actual_alpha_beta_K']==[1.,2.,32]
        selection=record['selection']
        # Native32 B1 contract preserves source-space mask order and the full
        # selected32 prefix. No answers, labels or correctness choose a row.
        expected=dict(question_id=p['question_id'],image=p['image'],full_prompt_sha256=record['full_LLM_prompt_sha256'],
            actual_guidance_sha256=record['guidance_sha256'],input_ids_sha256=record['actual_LLM_input_ids_sha256'],
            quotas=selection['greedy_quotas'],effective_ordered_prefix=selection['greedy_effective_prefixes'][0],
            effective_mask_indices=selection['effective_mask_indices'][0],output_token_ids=record['output_token_ids'][0],text=record['text'])
        match=[q for q in questions if q['question_id']==p['question_id'] and q['text']==p['prompt']]
        assert len(match)==1 and match[0]['image']==p['image']
        rows.append(dict(question_id=p['question_id'],prompt=p['prompt'],expected=expected,accepted_panel_record=c.describe(file)))
        panel_references.append(file)
    c.save(c.HERE/'beta2_panel_anchors.json',dict(rows=rows,accepted_beta2_native16=True,
        generated_answers_used_only_for_consistency=True,GT_or_accuracy_used=False,
        fixed_original_panel=True,acceptance_not_based_on_correctness_or_score=True))
    sources=dict(panel['source_sha256'])
    for name in ['llava/mm_utils.py','llava/constants.py','llava/utils.py']:
        path=Path(panel['llava_root'])/name;sources[str(path)]=c.sha(path)
    entry=Path(panel['llava_root'])/'llava/eval/model_vqa_loader.py'
    node=next(n for n in ast.parse(entry.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='eval_model')
    code=next(o for o in compile(entry.read_text(),str(entry),'exec').co_consts if isinstance(o,type(compile('','', 'exec'))) and o.co_name=='eval_model')
    sys.path.insert(0,str(c.POPE))
    from wait_only_patch import build
    wait_proof=build()[1]
    refs=[c.POPE/'wait_only_patch.py',c.POPE/'canonical_image_manifest.json',c.POPE/'model_identity/model_identity_audit.json',
        c.POPE/'run02_source_wait_only_k32/run_plan.json',c.POPE/'run02_source_wait_only_k32/full_result_summary.json',
        c.POPE/'run02_source_wait_only_k32/completion_and_release_proof.json',
        c.POPE/'run03_single_alpha05_beta1_wait_k32/run_plan.json',c.HERE/'beta2_panel_anchors.json',canonical]
    refs += [c.PANEL/name for name in c.PREDECESSOR]+panel_references
    refs += [c.PANEL/'RESULT.md',c.PANEL/'execution/loaded_runtime.json']
    refs += [c.POPE/'q03_fixed_cache_mechanism/public_sources'/name for name in [
        'sections/methodology.tex','sections/experiments.tex','tables/main_ablations.tex']]
    refs += [Path(path) for path in c.read(c.POPE/'run02_source_wait_only_k32/run_plan.json')['model_small_files']]
    files=['common.py','observe.py','worker.py','run.py','prepare_plan.py','verify_terminal.py','cpu_acceptance.py','README.md']
    # Label and metric guards are verified only by the post-generation CPU
    # scorer; generation/default check never opens label payloads.
    score_guards={p:h for p,h in baseline['small_file_guards'].items() if 'pinned_pope_labels' in p}
    metric=c.POPE/'verify_complete_predictions.py';score_guards[str(metric)]=c.sha(metric)
    scorer=Path(panel['llava_root'])/'llava/eval/eval_pope.py'
    score_guards[str(scorer)]=c.sha(scorer)
    plan=dict(run_id=c.RUN_ID,prepared_utc=c.UTC(),status='prepared_not_launched',python=c.PYTHON,n_questions=8910,
        parameters=c.PARAMETERS,llava_root=panel['llava_root'],model_path=panel['model_path'],entry=str(entry),
        question_file=str(canonical),image_folder=str(Path(panel['panel'][0]['image_path']).parent),
        source_sha256=sources,reference_sha256={str(p):c.sha(p) for p in refs},
        implementation_sha256={str(c.HERE/name):c.sha(c.HERE/name) for name in files},
        score_only_sha256=score_guards,model_stat_guards=baseline['model_stat_guards'],image_stat_guards=baseline['image_stat_guards'],
        expected_component_dtypes=panel['expected_component_dtypes'],
        eval_model_AST_sha256=c.text_sha(ast.dump(node,include_attributes=False)),
        eval_model_code_sha256=__import__('hashlib').sha256(marshal.dumps(code)).hexdigest(),
        wait_patch_prepared_proof=wait_proof,
        generation_entry_body_unchanged=True,sole_intervention='Native builder beta1 -> beta2; alpha1/K32/q.2 and all source arithmetic, full prompts, images and greedy128 unchanged.',
        public_basis='Released POPE alpha1 source shell plus paper beta>1 and explicit beta2 ablation; fixed single mechanism hypothesis, not known main-paper argv, repair or complete2x2.',
        predecessor=dict(root_proof=c.describe(c.PANEL/'root_terminal_process_proof.json'),peer=c.describe(c.PANEL/'peer_terminal_review.json'),
            source_native16_all_passed=True,actual_visual_input_changed16=True,output_changed1=True,correctness_not_examined=True),
        generation_outputs='Original predictions.jsonl plus post-generation actual guidance SHA/full LLM IDs+SHA/output IDs/native decode/actual K32 mask and effective ordered32 prefix. No full features/gain/embeddings saved for8910.',
        scoring_policy='Only after all8910 generation, complete raw/runtime, source/model/image guards and actual native exit: unchanged author eval_pope CLI and qualified independent parser/counts. Macro of random2910/popular3000/adversarial3000; preserve all outcomes and denominator.',
        old_source_results_preserved=True,score_direction_used_for_acceptance=False,paper_argv_confirmed=False,
        launch_policy=dict(explicit_root_allocation_required=True,single_attempt=True,automatic_retry=False,queue_or_watcher=False,max_GPU_models_including_loading=2),
        limitations=['One fixed beta test, not a parameter sweep or a known reproduction repair.',
            'The16 consistency anchors are generated beta2 answers and native selection, not GT or an accuracy direction gate.',
            'Observer hooks retain references only; all tensor content reads and serialization happen after native generate returns.',
            'Max128 non-EOS completions remain valid bounded generations and retain the official full denominator.',
            'Current controller and native worker must actually exit; root-owned exec proof is separately required for formal ledger inclusion.'])
    c.save(c.PLAN,plan);c.check(plan)
    c.save(c.HERE/'READY.json',dict(status='prepared_not_launched',plan_sha256=c.sha(c.PLAN),model_loaded=False,GPU_started=False,queue_started=False))


if __name__=='__main__':
    assert sys.argv[1:]==['prepare'];prepare()
