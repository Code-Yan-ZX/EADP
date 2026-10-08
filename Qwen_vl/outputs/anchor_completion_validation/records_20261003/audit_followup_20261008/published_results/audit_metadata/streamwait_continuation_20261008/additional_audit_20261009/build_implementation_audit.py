"""Read-only source/environment audit. Writes only this diagnostic namespace."""
from pathlib import Path
import ast
from collections import Counter
import datetime
import hashlib
import importlib.metadata as metadata
import importlib.machinery
import json
import subprocess

ROOT=Path('/media/disk2/YZX/research/EADP_amp')
OUT=Path(__file__).resolve().parent
STABILITY=ROOT/'Qwen_vl/outputs/audit_followup_20261008/next_gap_diagnosis_20261008/stability'
LLAVA=ROOT/'LLaVA'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def evidence(file,line,meaning):
    return dict(file=str(ROOT/file),line=line,meaning=meaning)


packages=['torch','torchvision','transformers','tokenizers','safetensors','flash-attn','accelerate','huggingface_hub','Pillow','numpy','sentencepiece','ftfy']
installed={}
for package in packages:
    try: installed[package]=metadata.version(package)
    except metadata.PackageNotFoundError: installed[package]=None
requirements={}
for line in (LLAVA/'requirements.txt').read_text().splitlines():
    if '==' in line:
        name,version=line.split('==',1)
        requirements[name.lower().replace('_','-')]=version
version_rows=[dict(package=name,installed=version,released_requirement=requirements.get(name.lower().replace('_','-')),matches_released=version==requirements.get(name.lower().replace('_','-'))) for name,version in installed.items()]
processes=[]
allowed=['HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE','TORCH_HOME','CUDA_VISIBLE_DEVICES','PYTHONPATH','USE_LLAVA_ARCH_CDPRUNER','USE_LLAVA_ARCH_DIVPRUNE','USE_LLAVA_ARCH_HIPRUNE','USE_LLAVA_ARCH_ABLATION','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE']
for pid in (2451780,2452469):
    proc=Path('/proc')/str(pid)
    if not proc.exists():
        processes.append(dict(pid=pid,present=False));continue
    env=dict(x.split('=',1) for x in (proc/'environ').read_bytes().decode().split('\x00') if '=' in x)
    libs=sorted({s.split()[-1] for s in (proc/'maps').read_text().splitlines() if '/' in s and any(k in s for k in ('libcublas','libcudnn','libtorch_cuda','libcuda.so','libnvrtc','flash_attn'))})
    processes.append(dict(pid=pid,present=True,cwd=str((proc/'cwd').resolve()),argv=(proc/'cmdline').read_bytes().decode().strip('\x00').split('\x00'),restricted_environment={k:env[k] for k in allowed if k in env},mapped_cuda_libraries=libs))
protocols=[]
for arm in ('EADP_beta2','AZ_beta2','EADP_beta1'):
    stem=STABILITY/f'next_text_K32_{arm}_streamwait'
    control=Path(str(stem)+'.control.protocol.json');cp=read(control)
    native=Path(str(stem)+'.jsonl.protocol.json');rt=Path(str(stem)+'.jsonl.runtime.jsonl')
    traces=[]
    if rt.exists():
        for line in rt.read_bytes().splitlines():
            try: traces.append(json.loads(line))
            except json.JSONDecodeError: break
    changed=[]
    for source,digest in cp['source_and_input_sha256'].items():
        if Path(source).stat().st_size>50*1024*1024:
            raise RuntimeError('Do not re-read model payload via frozen source table')
        if sha(source)!=digest:changed.append(source)
    protocols.append(dict(arm=arm,parameters=cp['parameters'],control_protocol=str(control),control_protocol_sha256=sha(control),native_protocol=read(native) if native.exists() else None,
        n_runtime_prefix=len(traces),actual_token_histogram=dict(Counter(r['actual_visual_tokens_retained'] for r in traces)),
        image_shape_histogram=dict(Counter(str(r['image_tensor_shape']) for r in traces)),
        runtime_budget_parameter_values=sorted({r['visual_token_budget_parameter'] for r in traces}),
        frozen_sources_checked=len(cp['source_and_input_sha256']),changed_sources=changed,
        actual_parameter_observation_boundary='Returned visual token counts and input shapes are dynamically logged. Alpha/beta/lambda follow frozen argv/source routes; exact loaded object attributes/class/dtypes/TF32 were not dynamically logged.',
        AST_validation=cp['AST_validation']))
summary=read(ROOT/'Qwen_vl/outputs/audit_followup_20261008/rerun_batch/repaired_results_summary.json')
full=[r for r in summary['rows'] if r['task']=='textvqa' and r['arm']=='FULL']
local_clip=read(OUT/'clip_weight_local_sha256.json');remote_clip=read(OUT/'clip_official_hf_revision_metadata.json');small_clip=read(OUT/'clip_small_file_identity.json')
remote_bin=next(s for s in remote_clip['siblings'] if s['rfilename']=='pytorch_model.bin')
clip_matches=local_clip['sha256']==remote_bin['lfs']['sha256'] and local_clip['size']==remote_bin['size']
models={name:read(Path('/media/disk2/YZX/doct/FastV')/name/'config.json') for name in ('llava-v1.5-7b','llava-v1.6-vicuna-7b')}
routes={}
for name,search in [('llava',[str(LLAVA)]),('llava.model',[str(LLAVA/'llava')]),('llava.model.builder',[str(LLAVA/'llava/model')]),('llava_arch_anchorzip',[str(LLAVA/'llava/model')]),('amp_common',[str(ROOT/'Qwen_vl/scripts/anchor_merge_pilot')])]:
    spec=importlib.machinery.PathFinder.find_spec(name,search);routes[name]=spec.origin if spec else None
mapping=read(OUT/'az_mapping_cpu_counterexamples.json')
report=dict(schema_version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Read-only additional implementation/environment audit. No GPU initialization, production model imports, new generation, downloads of model weights, changed frozen sources/parameters/environments/queues/protocols/predictions. One 1.712GB shared CLIP payload hashed once; existing 14.1GB NeXT identity proof reused.',
    git=dict(live_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),live_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),handoff_lines=906,handoff_sha256=sha(ROOT/'docs/project_handoff.md'),handoff_read='complete in this audit'),
    full_baseline=full,full_baseline_inference='v15 58.226 vs paper58.2; NeXT60.370 vs paper60.3. This narrows global model/input/decoding faults but cannot prove pruning correctness or author hyperparameters. FULL bypasses CLIP text/scoring.',
    current_processes=processes,current_wait_control_protocols=protocols,
    import_routes_without_importing_model_packages=routes,
    environment_packages=version_rows,
    confirmed_bugs=[dict(id='AZ_COMPLETION_SCOPE',classification='Confirmed preregistered mapping deviation; AZ-only, NeXT multi-crop',finding='Actual Completion flattens all crops of each image and allows cross-crop dropped-to-anchor assignment. Preregistration and implementation comments say per crop.',
        evidence=[evidence('docs/anchorzip_llava_port_prereg.md',51,'Explicit NeXT per-crop Completion commitment'),evidence('LLaVA/llava/model/llava_arch_anchorzip.py',16,'Implementation header claims per image crop'),evidence('LLaVA/llava/model/llava_arch.py',654,'split_sizes=number of crops per image, therefore [5] for these traces'),evidence('LLaVA/llava/model/llava_arch_anchorzip.py',158,'Group is a whole split_sizes image; flatten at167 before assignment')],
        cpu_counterexample=str(OUT/'az_mapping_cpu_counterexamples.json'),cross_crop_examples=mapping['Completion_cross_crop_prereg_deviation']['cross_crop_assignments'],max_abs_feature_difference_vs_per_crop=mapping['Completion_cross_crop_prereg_deviation']['max_abs_difference'],
        causal_boundary='Mathematically different grouping is proved, no downstream accuracy effect measured; does not explain EADP reproduction gap, does not affect v15 one-crop grouping.',
        candidate='Prepare a separate per-crop Completion implementation/arm/output/protocol, preserve existing global-per-image method and all wait-only controls; no current algorithm edited.')],
    confirmed_differences_and_behavior=[dict(id='RTG_SEGMENT0',status='Confirmed input coverage behavior; unproven method bug',finding='RTG reads only CLIP segment0 global/local embeddings. Current repaired TextVQA has881/5000 multi-segment guidance; shipped SQA_IMG has552/2017.',
        evidence=[evidence('LLaVA/llava/model/llava_arch_anchorzip.py',116,'Explicit index0 for A and g'),evidence('Qwen_vl/scripts/stage1_roundtrip_pilot/rtg_common.py',160,'Original Qwen RTG documented shape has exactly one text row M=1'),evidence('docs/anchorzip_llava_port_prereg.md',43,'Does not specify multi-segment LLaVA mapping')],
        cpu_result='Changing only segment1 leaves current RTG fused scores bit-exact unchanged',
        causal_boundary='Difference from EADP averaging all segments does not alone prove incorrect method. Intent for M>1 is unspecified; cannot cause EADP-only paper gap.',
        candidate='Explicitly document current segment0 scope; register a separate all-segment RTG comparison only if scientifically authorized, no silent change to current frozen method.'),
        dict(id='ENVIRONMENT_LOCK_DRIFT',status='Confirmed installed package differences from byte-identical released requirements',versions=[r for r in version_rows if not r['matches_released']],causal_boundary='Installed packages differ, but author executed versions unknown. No measured accuracy effect. Missing flash-attn/ftfy do not directly affect the active default SDPA/fast-tokenizer route; do not update live env.')],
    suspicious_not_proved=[
        dict(id='BETA_MAIN_TABLE_IDENTITY',finding='Frozen beta2 differs published shell default beta1; q=.2 main call differs helper default .5 and paper panels do not pin a unique author main-table configuration.',evidence=[evidence('LLaVA/scripts/v1_6/eval/textvqa.sh',11,'beta1 shell default'),evidence('LLaVA/llava/model/llava_arch.py',586,'q=.2 actual call')],causal_boundary='Not a new source mismatch. Existing separately registered beta1 control remains pending; no q/beta search performed.'),
        dict(id='LOADED_OBJECT_PROVENANCE',finding='Current frozen native protocol records requested args/config hashes and versions; runtime logs actual retained tokens/input shapes, not actual loaded class, tensor dtype, module training, alpha/beta/lambda or per-worker TF32 flags.',causal_boundary='Source and import routing support expected defaults but cannot replace absent dynamic object traces. No live-worker instrumentation added.'),
        dict(id='PACKAGE_NUMERICS',finding='Pillow12.3 vs10.4, tokenizers.15.2 vs.15.1, safetensors.8 vs.7, sentencepiece.2.2 vs.1.99 are candidates for small input/token differences compared with author environment.',causal_boundary='Official reference comparisons performed under same local libraries cannot prove author pixels/token IDs. Baselines close to paper weaken gross fault hypothesis but do not exclude per-question churn; no controlled cross-version effect measured.'),
        dict(id='ALLOCATOR_LIFETIME',finding='CLIP has no record_stream calls, but input/concat references remain live and original forward globally synchronizes both streams before returning.',evidence=[evidence('LLaVA/llava/model/llava_arch.py',656,'concat_images retained in caller stack through encode return'),evidence('LLaVA/llava/model/multimodal_encoder/clip_encoder.py',114,'global synchronize before returned tensors leave forward')],causal_boundary='No specific use-after-free path demonstrated. Missing record_stream alone is not a confirmed extra bug. Existing wait-only dependency repair remains frozen; cannot claim every allocator risk excluded.')],
    excluded_candidates=[
        dict(id='SHARED_CLIP_LOCAL_CORRUPTION',status='Excluded at file identity level',model_repo='openai/clip-vit-large-patch14-336',revision=remote_clip['sha'],file='pytorch_model.bin',size=local_clip['size'],local_sha256=local_clip['sha256'],official_lfs_sha256=remote_bin['lfs']['sha256'],payload_matches=clip_matches,stat_unchanged=local_clip['stat_unchanged'],small_files_matches=all(f['matches_official_blob'] for f in small_clip['files']),files=[f['name'] for f in small_clip['files']],
             route='Both model configs name this repo; load_model CLIPVisionModel, load_text_tower CLIPVisionModelWithProjection.visual_projection and CLIPTextModelWithProjection/CLIPTokenizerFast all read it.',
             checkpoint_selector='Transformers4.37.2 default use_safetensors=None tries absent main safe file, then main bin; cached PR20 safetensors is not selected by this route.',
             evidence=[evidence('LLaVA/llava/model/multimodal_encoder/clip_encoder.py',32,'Shared pretrained source'),evidence('LLaVA/llava/model/multimodal_encoder/clip_encoder.py',41,'Shared projection and text pretrained source')],
             causal_boundary='Confirms shared CLIP payload/config/tokenizer files match the official HF revision. Does not prove paper used this exact revision, actual in-memory objects, or v15 full LLM payload identity. NeXT LLM/projector payload identity only reuses earlier proof.',official_metadata_url=small_clip['url']),
        dict(id='CLIP_METADATA_RED_HERRINGS',finding='CLIP tokenizer_config name_or_path says base-patch32, but its entire file matches official large336 repo and vocab. CLIP config eos2 uses legacy argmax pooling in4.37, tokenizer EOS49407 is maximum ID. logit_scale is not used by this negated normalized cosine scoring route.',causal_boundary='Metadata naming/EOS convention alone does not prove wrong tokenizer or pooled embeddings.'),
        dict(id='CORE_CODE_OR_PROJECTOR_ROUTE_DRIFT',extended_source_identity=str(OUT/'extended_source_identity.json'),identical_files=13,interpretation='Official clone and local builder/arch/vision/projector builders/conversation/utils/loader/environment/requirements files all byte-identical; no local formula substitution for EADP detected.'),
        dict(id='BUDGET_GEOMETRY',finding='Text controls actual prefix shape always[1,5,3,336,336], per-crop parameter32, resulting156–159. Official importance allocation floor/min1 may leave nominal160 unspent, and different scorers can receive different actual counts.',causal_boundary='Expected released allocation behavior, not missing global crop or mistaken total32. No budget or allocator changed.'),
        dict(id='DECODING_PROMPT_CONFIG_DRIFT',finding='New explicit official Text wrapper removes only score-text suffix, keeps full OCR LLM chat, vicuna_v1, greedy/do_sampleFalse, beams1, top_pNone, max_new128/use_cacheTrue; eos2/bos1/pad0 match checkpoint and release.',causal_boundary='Old model_vqa1024/suffix drift is already repaired by separate outputs; exact author argv still unavailable.'),
        dict(id='FLASH_ATTENTION_CONFIG_MISREAD',finding='NeXT generation_config attn_implementation=flash_attention_2 is arbitrary generation metadata. Model config attention internal isNone; builder use_flash_attnFalse does not request external Flash2. Transformers4.37.2 Llama supports default SDPA, expected active implementation is SDPA.',evidence=[evidence('LLaVA/llava/model/builder.py',26,'use_flash_attn defaultFalse'),evidence('LLaVA/llava/model/builder.py',45,'only explicitTrue requests flash2')],causal_boundary='Do not infer actual Flash2 from generation_config. Exact live attention object not recorded; current route matches release default under this torch/transformers.'),
        dict(id='CUDA12_LIBRARY_CONFLICT',finding='Actual mapped worker libraries are torch-bundled libcublas.so.11, libcudnn.so.8 and CUDA11 nvrtc; no external flash_attn library mapping. Requirements list unused CUDA12 distributions but maps show active cu118 path.',causal_boundary='No detected worker loaded CUDA12 CUBLAS/cuDNN conflict; driver/hardware/author low-level versions not matched here.'),
        dict(id='HISTORICAL_CROSS_QUESTION_CACHE',finding='Similarity debug cache is write-only disabled by default; active RTG/uniform has no RNG; greed_select allocates coverage/masks each call. Original scalar/gather route source unchanged.',causal_boundary='Static analysis excludes these specific persistent-state candidates, not all CUDA nondeterminism. Synchronization instability already demonstrated in separate prior full/panel controls.')],
    artifacts=dict(clip_payload=str(OUT/'clip_weight_local_sha256.json'),clip_official_metadata=str(OUT/'clip_official_hf_revision_metadata.json'),clip_small_files=str(OUT/'clip_small_file_identity.json'),method_cpu_counterexamples=str(OUT/'az_mapping_cpu_counterexamples.json'),extended_source_identity=str(OUT/'extended_source_identity.json'),prior_NeXT_payload_identity=str(ROOT/'Qwen_vl/outputs/audit_followup_20261008/next_gap_diagnosis_20261008/input_identity/weight_payload_identity.json')),
    final_causal_limit='No newly proved EADP-paper residual root cause. Shared CLIP and extended local source drift candidates narrowed; new confirmed algorithm mapping deviation is AZ-only. Wait-only controls and beta1 predefined comparison remain unchanged, current prefixes not scored as complete runs.')
adapter=OUT/'per_crop_completion_adapter.py'
adapter_validation=OUT/'per_crop_completion_adapter_cpu_validation.json'
if adapter.exists() and adapter_validation.exists():
    av=read(adapter_validation)
    report['prepared_per_crop_completion_candidate']=dict(
        adapter=str(adapter),adapter_sha256=sha(adapter),validation=str(adapter_validation),
        validation_sha256=sha(adapter_validation),CPU_gates=av['gates'],
        installed=False,registered_in_existing_queue=False,measured_GPU_accuracy=False,
        implementation='Factory wraps the unchanged original Completion callable, groups flattened tokens in576-token crops, calls original assignment/merge independently, preserves ascending global gather order.',
        untouched='RTG segment0 scope, alpha/beta/lambda, visual features, facility/quota, keep IDs and every existing source/queue/protocol/prediction/environment.',
        next_decision='Review this concrete isolated candidate, then independently register a paired comparison if requested; no silent switch of the55 current publication groups.')
(OUT/'implementation_environment_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(dict(created_utc=report['created_utc'],clip_official_match=clip_matches,new_confirmed_bug_ids=[x['id'] for x in report['confirmed_bugs']],package_differences=[r for r in version_rows if not r['matches_released']],protocol_source_checks=[dict(arm=p['arm'],checked=p['frozen_sources_checked'],changed=p['changed_sources'],n_runtime_prefix=p['n_runtime_prefix']) for p in protocols],output=str(OUT/'implementation_environment_audit.json')),indent=2))
