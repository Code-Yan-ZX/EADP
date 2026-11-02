"""Frozen native-panel metadata and guards; imports no model/tensor runtime."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from contracts import ARMS,POSITIONS,sha,read,save,stat
import gain_observer

HERE=Path(__file__).resolve().parent
POPE=HERE.parent
PANEL=POPE/'gpu_formula_facility_panel'
HISTORY=POPE/'run02_source_wait_only_k32'
PLAN=HERE/'plan.json'
OUT=HERE/'execution'
PYTHON='/home/dell/miniconda3/envs/llava_pruner/bin/python'
RUN_ID='POPE_fixed16_native_alpha1_beta1_vs_beta2_K32'
UTC=lambda:datetime.now(timezone.utc).isoformat()
LOCKED={
    PANEL/'run_plan.json':'355f020bb2bcaf9ac37f4daa061f504ec532e155c0cb61be5aa966fc3e2df4ff',
    PANEL/'execution/per_image.jsonl':'1948d294bb12027872995f1d6dd0d4e711bf9b4a64c40fdd00bfaf4a5b053ff1',
    PANEL/'execution/loaded_runtime.json':'32fa78e5ec98a69b15dfa42568b6a2b3eef00c81546098568aaaeff8d249c10a',
    PANEL/'independent_verification.json':'ef6e790c133b9f247130f841ed7b93be6f41fc43844a3e4da8eed539b8ef7d46',
    PANEL/'root_terminal_process_proof.json':'c690cc0d0f7f24999031d4b7a90b9b6f3f3211e351cf89600b8b7743387cf581',
    HISTORY/'run_plan.json':'83bd2829d421871e120641b9910d31f4a488864ee007534a35a5ba5c497f7ece',
    HISTORY/'full_result_summary.json':'15f84c054ab366216cc5cfc2271fd3a423d6c698366883b22a828699daa9b82f',
    HISTORY/'predictions.jsonl':'d23f3c0c7abe3233ec22ee38c8816f3b0ec047a5d0d665a69c5d90982e351233',
}


def describe(path):
    path=Path(path);return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def check(plan):
    assert Path(sys.executable).resolve()==Path(PYTHON).resolve()
    assert plan['run_id']==RUN_ID and plan['arms']==list(ARMS)
    assert plan['parameters']==dict(K=32,q=.2,alpha=1.,source_beta=1.,variant_beta=2.,temperature=0.,top_p=None,
        num_beams=1,max_new_tokens=128,conv_mode='vicuna_v1',use_cache=True)
    for group in ['source_sha256','implementation_sha256','reference_sha256']:
        for path,digest in plan[group].items():assert sha(path)==digest,path
    for path,guard in plan['model_stat_guards'].items():assert stat(path)=={k:v for k,v in guard.items() if k!='prior_verified_sha256'},path
    for row in plan['panel']:
        assert sha(row['image_path'])==row['image_sha256'] and sha(row['cache']['path'])==row['cache']['sha256']
    assert len(plan['panel'])==len({r['image'] for r in plan['panel']})==16
    assert [r['index'] for r in plan['panel']]==POSITIONS
    assert plan['gain_observer_proof']==gain_observer.transformed()[1]
    fixture=read(HERE/'source_anchor_fixture.json')['rows']
    assert len(fixture)==16 and [r['position'] for r in fixture]==POSITIONS
    for row,panel in zip(fixture,plan['panel']):
        assert row['question_id']==panel['question_id'] and row['prompt']==panel['prompt'] and row['image']==panel['image']
        assert 'expected_source_output_token_ids' not in row
    assert 'torch' not in sys.modules


def prepare():
    assert not PLAN.exists() and not OUT.exists() and not (HERE/'launch.claim.json').exists()
    for path,digest in LOCKED.items():assert sha(path)==digest,path
    prior=read(PANEL/'run_plan.json');old=read(HISTORY/'run_plan.json');summary=read(HISTORY/'full_result_summary.json')
    assert summary['n_predictions']==8910 and summary['all_8910_question_ids_order_verified'] and summary['gpu_worker_exit_verified']
    assert summary['official_scorer_exit_code']==0 and summary['predictions_sha256']==LOCKED[HISTORY/'predictions.jsonl']
    assert old['parameters']['alpha']==old['parameters']['beta']==1. and old['parameters']['visual_token_num']==32
    assert old['parameters']['max_new_tokens']==128 and old['parameters']['source_caller_stream_wait'] is True
    raw=[json.loads(line) for line in (HISTORY/'predictions.jsonl').read_text().splitlines()]
    canonical=[json.loads(line) for line in Path(prior['question_file']).read_text().splitlines()]
    assert len(raw)==len(canonical)==8910
    index={(str(r['question_id']),r['prompt']):r for r in raw};assert len(index)==8910
    questions={str(r['question_id']):r for r in canonical};assert len(questions)==8910
    caches=[json.loads(line) for line in (PANEL/'execution/per_image.jsonl').read_text().splitlines()]
    panels=[];fixture=[]
    for i,(panel,cache) in enumerate(zip(prior['panel'],caches)):
        assert panel==cache['panel'] and panel['index']==cache['index']==i
        q=questions[str(panel['question_id'])];assert q['text']==panel['prompt'] and q['image']==panel['image']
        answer=index[(str(panel['question_id']),panel['prompt'])]
        panels.append(dict(panel,cache=dict(path=cache['cache_path'],sha256=cache['cache_sha256'])))
        fixture.append(dict(position=i,question_id=panel['question_id'],image=panel['image'],prompt=panel['prompt'],expected_source_text=answer['text']))
    save(HERE/'source_anchor_fixture.json',dict(rows=fixture,history_raw=describe(HISTORY/'predictions.jsonl'),
        source_text_consistency_only=True,historical_output_IDs_available=False,historical_input_IDs_available=False,
        historical_prepared_embeddings_available=False,GT_or_scores_used_for_selection=False))
    reference_paths=list(LOCKED)+[HERE/'source_anchor_fixture.json',HISTORY/'completion_and_release_proof.json',POPE/'wait_only_patch.py',
        POPE/'alpha1_beta2_existing_metadata_audit_20261029.json',
        POPE/'q03_fixed_cache_mechanism/public_sources/sections/methodology.tex',
        POPE/'q03_fixed_cache_mechanism/public_sources/sections/experiments.tex',
        POPE/'q03_fixed_cache_mechanism/public_sources/tables/main_ablations.tex']
    reference_paths += [Path(prior['question_file']),POPE/'model_identity/model_identity_audit.json']
    reference_paths += [Path(p) for p in old['model_small_files']]
    files=['common.py','contracts.py','gain_observer.py','worker.py','run.py','verify_terminal.py','cpu_acceptance.py','README.md']
    runtime=read(PANEL/'execution/loaded_runtime.json')
    plan=dict(run_id=RUN_ID,prepared_utc=UTC(),status='prepared_not_launched',python=PYTHON,
        parameters=dict(K=32,q=.2,alpha=1.,source_beta=1.,variant_beta=2.,temperature=0.,top_p=None,num_beams=1,
            max_new_tokens=128,conv_mode='vicuna_v1',use_cache=True),arms=list(ARMS),
        llava_root=prior['llava_root'],pope_root=str(POPE),model_path=prior['model_path'],panel=panels,
        source_sha256=prior['source_sha256'],implementation_sha256={str(HERE/n):sha(HERE/n) for n in files},
        reference_sha256={str(p):sha(p) for p in reference_paths},model_stat_guards=prior['model_stat_guards'],
        expected_component_dtypes=dict(vision_body=runtime['dtype']['vision'],visual_projection=runtime['dtype']['projection'],
            text_tower=runtime['dtype']['text'],mm_projector=runtime['dtype']['mm_projector']),
        gain_observer_proof=gain_observer.transformed()[1],
        source_anchors=['Original actual native1/1 projected/input/global/dense/visualsim/importance/ordered32/finalmask caches.',
            'Old same-AST alpha1/beta1 GPU helper complete gain trace; not an old native internal gain capture.',
            'Qualified8910 source1/1 decoded text at exact unique(questionID,fullprompt) keys; no historical outputIDs/inputIDs/prepared embeddings exist.'],
        sole_intervention='Set actual native model.beta from1 to2 between complete source16 and variant16; record actual self alpha/beta/K at encode. All weights, source math, q.2, full LLM/guidance prompt, image processing and greedy128 arguments remain.',
        source_first='All16 source raw attempts saved and all native/cache/history-text gates passed before any beta2 generation. Any failure saves the attempted data then stops, without retuning/retry.',
        observations='Reference-only wrappers and one inverse-verified native greedy gain_refs.append after selected-mask assignment. All content reads, stacks and serialization occur only after generate returns.',
        saved_arrays=['projected_fp16','image_embeds_fp32','text_global_fp32','text_dense_fp32','global_fp32','dense_fp32',
            'visual_sim_fp32','importance_fp32','internal_ids','quotas','gain_trace_fp32','final_mask','actual_input_fp16',
            'actual_LLM_input_ids','prepared_embeddings_fp16'],
        public_basis='Released POPE shell alpha1/beta1 source control plus paper methodology beta>1 and an explicit beta2 ablation. This is one fixed mechanism hypothesis, not main paper argv, known bug, full2x2 or a comparable historical.5/2 runtime claim.',
        backend='sdpa',dtype='torch.float16',no_GT_or_accuracy=True,paper_argv_confirmed=False,
        launch_policy=dict(root_GPU_allocation_required=True,single_attempt=True,automatic_retry=False,queue_or_watcher=False,max_GPU_models=2),
        limits=['Native prepare/generate bodies unchanged. New gain observer is explicitly disclosed, and its removal restores complete original greedy AST.',
            'Both arms newly save actual outputIDs and native decoded text. Historical run02 has text only, never a fabricated historical ID gate.',
            'B1/576/32 visual rows use source spatial mask order. Actual prepared FP16 visual region and nonvisual prefix/suffix are independently byte-checked.',
            'Text is decoded by the worker native tokenizer; CPU terminal audit checks IDs/source/horizon metadata without independent tokenizer re-decoding.',
            'Max128 without EOS is a bounded observation; preserve EOS/horizon flags. No answer normalization, GT score, accuracy or score-direction selection.',
            'Fixed16 mechanism only; no paper reproduction, new full benchmark, parameter sweep, timing-efficiency claim or automatic next run.'])
    save(PLAN,plan);check(plan)
    save(HERE/'READY.json',dict(status='prepared_not_launched',plan_sha256=sha(PLAN),GPU_started=False,model_loaded=False,queue_started=False))


if __name__=='__main__':
    assert sys.argv[1:]==['prepare'];prepare()
