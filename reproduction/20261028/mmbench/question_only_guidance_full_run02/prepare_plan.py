"""Freeze full MMB candidate. CPU only; never starts model, devices or scoring."""
from pathlib import Path
import common as c

PANEL=c.HERE.parent/'question_only_guidance_panel'
FAILED=c.HERE.parent/'question_only_guidance_full'

def main():
    assert not c.PLAN.exists() and not c.OUT.exists() and not (c.HERE/'launch.claim.json').exists()
    m=c.read(c.HERE/'inputs/manifest.json');rows=m['rows'];assert m['n_rows']==4377 and m['n_groups']==1176 and not m['all_rounds'] and len(rows)==4377
    assert [r['position'] for r in rows]==[r['original_position'] for r in rows]==list(range(4377))
    assert len({r['index'] for r in rows})==4377 and len({r['group_id'] for r in rows})==1176
    assert m['source_TSV_bytes_read_including_GT'] and not m['GT_columns_parsed']
    panel_plan=c.read(PANEL/'plan.json');previous=c.read(PANEL/'inputs/manifest.json');old=previous['rows'];assert len(old)==60
    root=c.read(PANEL/'root_exit_proof.json');state=c.read(PANEL/'execution/state.json');ver=c.read(PANEL/'execution/independent_verification.json')
    assert root['root_pty_actual_exit_code']==0 and root['verifier_container_not_running'] and all(x['original_process_absent'] for x in root['process_checks'])
    assert root['state_sha256']==c.sha(PANEL/'execution/state.json') and root['verifier_sha256']==c.sha(PANEL/'execution/independent_verification.json')
    assert state['status']=='complete' and state['native_exit_code']==state['verifier_exit_code']==0
    assert ver['success'] and ver['plan_sha256']==c.sha(PANEL/'plan.json') and ver['n_rows']==60 and ver['n_generations']==120
    assert ver['native_complete_sha256']==c.sha(PANEL/'execution/native_complete.json') and ver['native_release_sha256']==c.sha(PANEL/'execution/native_release.json')
    assert all(rows[r['original_position']]['index']==r['index'] for r in old)
    failed_proof=c.read(FAILED/'root_failure_exit_proof.json');failed_state=c.read(FAILED/'execution/state.json')
    assert failed_proof['root_pty_actual_exit_code']==1 and failed_proof['root_pty_exit_observed_by']=='functions.write_stdin'
    assert failed_state['status']=='failed' and failed_state['native_exit_code']==1 and failed_state['original_worker_exited'] and failed_state['GPU_pid_absent']
    assert failed_proof['state_sha256']==c.sha(FAILED/'execution/state.json') and failed_proof['worker_log_sha256']==c.sha(FAILED/'execution/worker.log')
    source=c.SOURCE.parents[2]
    names=('common.py','worker.py','observer.py','controller.py','cpu_container.py','cpu_preflight.py','cpu_preflight_container.py','prepare_plan.py','prepare_inputs.py','score_terminal.py','verify_generation.py','test_verify_generation.py','test_full_inputs.py','test_source_identity.py','test_observer.py','test_controller.py','test_dispatch.py','test_score_terminal.py','DESIGN.md','inputs/manifest.json','inputs/panel.tsv')
    paths=[c.HERE/x for x in names]
    paths += [Path(x['path']) for x in panel_plan['files']]
    paths += [PANEL/x for x in ('plan.json','root_exit_proof.json','execution/state.json','execution/native_complete.json','execution/native_release.json','execution/independent_verification.json','execution/runtime.json','execution/source_anchor_gate.json')]
    paths += [FAILED/x for x in ('plan.json','cpu_preflight.json','worker.py','root_failure_exit_proof.json','execution/state.json','execution/worker.log','execution/worker_failure.json','execution/worker_identity.json','execution/cases/source_0023.raw.json','execution/cases/source_0023.output_ids.json','execution/cases/source_0023.capture.json','execution/cases/source_0023.observation_failure.json','execution/anchors/source_00.npz')]
    for arm in ('source','question_only'):
        for i in range(60):
            rpath=PANEL/'execution'/('%s_%02d.json'%(arm,i));r=c.read(rpath)
            for key in ('raw','output_ids_file','archive'):
                p=Path(r[key]['path']);assert c.sha(p)==r[key]['sha256'];paths.append(p)
            paths.append(rpath)
    paths.extend(Path(r['image']['path']) for r in rows)
    paths=list(dict.fromkeys(paths))
    # Original GT TSV is deliberately not opened or hashed in preparation of this plan.
    gt=(c.HERE.parent/'data/mmbench_dev_20230712.tsv').resolve()
    assert all(p.resolve()!=gt for p in paths)
    matching=c.HERE.parents[3]/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/utils/matching_util.py'
    mcq=c.HERE.parents[3]/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/dataset/utils/multiple_choice.py'
    plan=dict(run_id='MMB_full_source_vs_question_only_run02',prepared_only=True,n_rows=4377,n_groups=1176,
        files=[dict(path=str(p),sha256=c.sha(p)) for p in paths],model_stat=panel_plan['model_stat'],
        panel_sha256=c.sha(c.HERE/'inputs/panel.tsv'),manifest_sha256=c.sha(c.HERE/'inputs/manifest.json'),
        score_only_files={str(p):c.sha(p) for p in (matching,mcq)},GT_file_expected_sha256='52d45f29876e56138d3d4feb169dc9a7323f6ed2ee3f439ae7e4af94192b3324',
        arms=['source','question_only'],parameters=dict(K=128,alpha=.5,beta=1.,temperature=0.,top_p=None,num_beams=1,max_new_tokens=1024,conv_mode='vicuna_v1',all_rounds=False),
        prior_panel_root_proof_sha256=c.sha(PANEL/'root_exit_proof.json'),prior_panel_verification_sha256=c.sha(PANEL/'execution/independent_verification.json'),
        prior_failed_run_root_proof_sha256=c.sha(FAILED/'root_failure_exit_proof.json'),
        source_order='First60 fixed panel pairs with old actual arrays/IDs/text anchors; remaining4317 source then variant per original row; output restored to original order.',
        limits=['Input preparation read GT-bearing TSV bytes but only parsed input columns; generation and verifier read sanitized inputs only.',
                'Only 120 anchor captures retain full high-dimensional arrays; other high-dimensional features are runtime hashes and cannot be independently reconstructed from summaries.',
                'Author source late CLIP guidance and previously accepted three wait statements are in-memory AST patches. Full LLM prompt remains unchanged. Question-only changes semantics and token segmentation.',
                'Public exact-matching helpers and independent local circular score are separately gated and are not external official-server results.'],
        launch_policy=dict(root_explicit_allocation_required=True,single_attempt=True,automatic_retry=False,automatic_queue=False,one_model=True))
    c.save_new(c.PLAN,plan);c.check(plan);print('Frozen plan',c.sha(c.PLAN),'guarded files',len(paths))
if __name__=='__main__':main()
