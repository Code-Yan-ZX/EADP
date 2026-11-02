"""Freeze candidate; CPU-only. Never launches the controller or loads a model."""
import sys
import common as c


def main():
    assert not c.PLAN.exists() and not c.OUT.exists() and not (c.HERE/'launch.claim.json').exists()
    m=c.read(c.HERE/'inputs/manifest.json')
    assert m['n_rows']==60 and m['n_groups']==16 and not m['all_rounds']
    assert not m['GT_columns_parsed'] and m['source_TSV_bytes_read_including_GT']
    assert len({x['index'] for x in m['rows']})==60
    previous=c.read(c.HERE.parent/'full_token_baseline_control/protocol.json')
    source=c.SOURCE.parents[2]
    paths=[c.HERE/x for x in ('common.py','observer.py','worker.py','controller.py','cpu_container.py','prepare_plan.py','cpu_preflight.py','cpu_preflight_container.py','test_observer.py','test_controller.py','test_source_identity.py','test_sanitized_inputs.py','verifier.py','test_verifier.py','prepare_inputs.py','test_prepare_inputs.py','README.md','inputs/manifest.json','inputs/panel.tsv')]
    paths += [c.HERE.parent/'question_only_guidance_panel_design/source_boundary.py',c.HERE.parent/'question_only_guidance_panel_design/cpu_design_report.json',
              c.HERE.parent/'streamwait_control/patch_only.py',c.HERE.parent/'streamwait_control/runtime_patch_validation.json',
              c.HERE.parent/'streamwait_control/observed/runtime.jsonl',c.HERE.parent/'streamwait_control/v15_EADP_K128_oldEN_beta1_alpha05_streamwait.jsonl',
              c.HERE.parent/'question_only_guidance_cpu_audit_20261030/report.json',c.HERE.parent/'question_only_guidance_cpu_audit_20261030/token_fingerprints.jsonl',
              c.HERE.parent/'cpu_inputs_current.json',c.HERE.parent/'cpu_inputs_author_pins.json',
              c.HERE.parent/'cpu_llm_tokens_current.json',c.HERE.parent/'cpu_llm_tokens_author_pins.json',c.HERE.parent/'image_manifest.json']
    paths += [source/x for x in ('llava/eval/model_vqa_mmbench.py','llava/model/llava_arch.py','llava/model/language_model/llava_llama.py','llava/model/builder.py','llava/model/multimodal_encoder/clip_encoder.py','llava/mm_utils.py','llava/conversation.py','llava/constants.py')]
    paths += [__import__('pathlib').Path(r['image']['path']) for r in m['rows']]
    paths=list(dict.fromkeys(paths))
    plan=dict(run_id='MMB_fixed16groups_source_vs_question_only',prepared_only=True,n_rows=60,n_groups=16,
        files=[dict(path=str(p),sha256=c.sha(p)) for p in paths],model_stat=previous['model_stat_guards'],
        panel_sha256=c.sha(c.HERE/'inputs/panel.tsv'),manifest_sha256=c.sha(c.HERE/'inputs/manifest.json'),
        parameters=dict(K=128,alpha=.5,beta=1.,temperature=0.,top_p=None,num_beams=1,max_new_tokens=1024,conv_mode='vicuna_v1',all_rounds=False),
        arms=['source','question_only'],source_gate='All 60 original author results and existing historical text/runtime plus CPU input references before any variant.',
        intervention='Only late CLIP texts assignment uses row question instead of full guidance. Full LLM conversation unchanged.',
        limitations=['Inactive source comment is a hypothesis, not a demonstrated paper recipe or bug.','Question-only removes hints and options, changes meaning and text segmentation; it is not a pure length control.',
          'Previous GPU records preserve responses and runtime shape, not feature arrays, masks or output token IDs; image and token fingerprints were independently rebuilt on CPU.',
          'Circular options are pre-expanded; do not require group answer letters to match. No GT, scores or accuracy in worker or verifier.',
          'Input preparation read source TSV bytes containing GT, but did not parse or use its answer column. Runtime reads sanitized input only.',
          '120 calls, one model; old source timing suggests about 32 seconds of generation only but full elapsed time, GPU peak and disk usage are uncertain. Projected arrays alone are about 540 MiB uncompressed.'],
        launch_policy=dict(root_explicit_allocation_required=True,single_attempt=True,automatic_retry=False,automatic_queue=False,one_model=True))
    c.save_new(c.PLAN,plan);c.check(plan)
    print('Frozen plan:',c.sha(c.PLAN),'files:',len(paths))

if __name__=='__main__':main()
