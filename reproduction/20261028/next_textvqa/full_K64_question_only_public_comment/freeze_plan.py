"""Freeze only after the completed32-case source/comment input mechanism gate."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import worker

OUT,NEXT=worker.OUT,worker.NEXT
PANEL=NEXT/'question_only_guidance_panel'
BASE=NEXT/'full_default_K64_public'
read=lambda p:json.loads(Path(p).read_text())
sha=worker.sha
PANEL_LOCKED={
    PANEL/'root_terminal_process_proof.json':'272ae626ed2bc36b03ea69d38bb21c9f11a6cb9a82df4b0f87f297693a23e356',
    PANEL/'execution/independent_verification.json':'f9d9076891883ea98d8569d9907e43e056a34303211339fe8fd6728c8ee8479b',
    PANEL/'RESULT.md':'7e59616cc220f49a01db970bcaa9bba71403207fb7de88651c79633ad7696fa2',
    PANEL/'plan.json':'812c39f98bb238c0f8bf513c0b730fb85a41c5ee3603510f82b2db16592dfe52',
}


def check_panel_metadata(root,report):
    assert root['root_owned_exec_handle']==76915 and root['root_owned_exec_actual_exit_code']==0
    assert root['all_original_processes_exited'] is True and root['n_generations']==32 and root['source_anchors_verified']==16
    assert report['success'] and report['n_generations']==32 and report['source_anchors_verified']==report['variant_inputs_verified']==16
    assert report['full_LLM_IDs_and_nonvisual_embeddings_unchanged'] is True
    assert report['real_projected_image_embeds_visualsim_pixels_equal'] is True
    assert report['source_native_feature_and_fullK64_selection_output_anchors'] is True
    assert report['GT_or_accuracy_used'] is False and report['new_benchmark_row'] is False
    assert len(report['changes'])==16 and any(r['final_masks_changed'] for r in report['changes'])


def strict_stat(path):
    p=Path(path);s=p.stat()
    return dict(resolved=str(p.resolve()),bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def check_prerequisite(plan):
    for path,digest in PANEL_LOCKED.items():assert sha(path)==digest,path
    root=read(PANEL/'root_terminal_process_proof.json');report=read(PANEL/'execution/independent_verification.json')
    check_panel_metadata(root,report)
    assert sha(PANEL/'execution/release.json')==root['release_sha256']
    assert sha(PANEL/'execution/independent_verification.json')==root['independent_verification_sha256']
    for path,guard in plan['strict_model_stat_guards'].items():
        assert strict_stat(path)=={k:v for k,v in guard.items() if k!='prior_verified_sha256'},path
    assert plan['input_hypothesis']['known_paper_argv'] is False


def prepare():
    assert not worker.PLAN.exists() and not worker.RUN.exists() and not (OUT/'launch.claim.json').exists()
    assert sha(BASE/'plan.json')=='9ef9b4c69727d7baf324c5e3b3d93fd8c7d23a899d466f35e8c2389f05405d0d'
    base=read(BASE/'plan.json');panel=read(PANEL/'plan.json')
    plan=copy.deepcopy(base);full,proof=worker.adapted_module()
    plan.update(run_id=worker.RUN_ID,prepared_utc=datetime.now(timezone.utc).isoformat(),status='prepared_not_launched',
        purpose='Full5000 single input hypothesis supported by the inactive public loader comment; retain every result regardless of score.',
        worker=dict(path=str(OUT/'worker.py'),sha256=sha(OUT/'worker.py')),worker_adapter=proof,
        parent_completed_K64_plan=dict(path=str(BASE/'plan.json'),sha256=sha(BASE/'plan.json')),
        strict_model_stat_guards=panel['model_stat_guards'],
        input_hypothesis=dict(transform="question.split('\\nReference OCR token')[0] before existing suffix removal, only for texts=",
            source_comment='Fresh model_vqa_loader.py:113 inactive line',known_paper_argv=False,
            full_LLM_OCR_prompt_or_IDS_changed=False,images_or_pruning_math_changed=False,
            parameters_changed=False,full_denominator=5000,metric_or_parameter_selection=False),
        prerequisite=dict(root_proof_sha256=PANEL_LOCKED[PANEL/'root_terminal_process_proof.json'],
            independent_report_sha256=PANEL_LOCKED[PANEL/'execution/independent_verification.json'],
            source_anchors=16,generations=32,real_native_input_change_verified=True,
            independent_peer='latest_version_audit complete terminal PASS confirmed by root before this freeze; no accuracy selection'),
        baseline_comparator=dict(path=str(BASE/'execution/independent_verification.json'),
            sha256=sha(BASE/'execution/independent_verification.json'),scope='Completed same public K64 full-guidance control; preserve both complete results'),
        commands=dict(check=['/home/dell/miniconda3/envs/llava_pruner/bin/python','-B',str(OUT/'run.py'),'check'],
            run_requires_separate_root_allocation=['/home/dell/miniconda3/envs/llava_pruner/bin/python','-B',str(OUT/'run.py'),'run','--allocation-note','FUTURE_ROOT_NOTE.json'],
            verify_CPU=['/home/dell/miniconda3/envs/llava_pruner/bin/python','-B',str(OUT/'verify.py')]),
        limits='Single question-only pruning-guidance input hypothesis, not known paper argv or a confirmed fix. Entire LLM OCR prompt, all5000 input IDs, images, parameters, pruning and generation math remain. All complete results retained; no score-direction gate, retry or multiple-budget sweep.')
    paths=[OUT/n for n in ['worker.py','guidance_patch.py','observe.py','run.py','verify.py','freeze_plan.py','cpu_acceptance.py','README.md']]
    paths += [BASE/n for n in ['worker.py','observe.py','run.py','verify.py','plan.json','root_terminal_process_proof.json','execution/independent_verification.json']]
    paths += [NEXT/n for n in ['repro_full_default.py','repro_panel.py','llm_tokens_current.json','llm_tokens_author_pins.json',
        'guidance_question_boundary_exact_token_rows.jsonl','guidance_question_boundary_exact_tokenization.json']]
    paths += list(PANEL_LOCKED)+[PANEL/'execution/release.json']
    plan['additional_frozen_sources']=[dict(path=str(p.resolve()),sha256=sha(p)) for p in paths]
    check_prerequisite(plan)
    worker.PLAN.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    from run import check
    check(plan)
    (OUT/'READY.json').write_text(json.dumps(dict(status='prepared_not_launched',plan_sha256=sha(worker.PLAN),
        model_loaded=False,GPU_started=False,watcher_started=False,root_allocation_required=True),indent=2)+'\n')
    assert 'torch' not in sys.modules


if __name__=='__main__':prepare()
