"""CPU exact image-input tiers for two completed beta2 runs; no generation."""
import ast
import base64
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import PIL
from PIL import Image
import torch
import transformers
from transformers import CLIPImageProcessor

import compare_beta_and_v11 as first

BASE = first.BASE
OUT = BASE / "v11_image_overlap_stage2"
CLIP = Path("/home/dell/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1")
EXPECTED_PREPROCESSOR = "d253881f65322dc546df59cf925a408e5538b8ecb5a1b496cdd36af9992686d4"


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def payloads(data):
    bases = {int(row["index"]) % 1000000: row["image"] for _, row in data.iterrows()
             if isinstance(row["image"], str) and len(row["image"]) > 100}
    return {int(row["index"]): row["image"] if isinstance(row["image"], str) and len(row["image"]) > 100
            else bases[int(row["index"]) % 1000000] for _, row in data.iterrows()}


def main():
    first.complete_gate()
    assert not torch.cuda.is_initialized()
    assert PIL.__version__ == "10.4.0" and "vendor_author_pins" in PIL.__file__, "Use the isolated author image pins"
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    stage1_path = BASE / "beta1_vs_beta2_and_v11_full/comparison.json"
    stage1 = first.read(stage1_path)
    assert stage1["success"] and stage1["full4377_beta_comparison"]["n_rows"] == 4377
    for record in stage1["sources"]:
        assert first.sha(record["path"]) == record["sha256"], record["path"]
    assert first.sha(stage1["paired_rows"]["path"]) == stage1["paired_rows"]["sha256"]
    rows = [json.loads(line) for line in Path(stage1["paired_rows"]["path"]).read_text().splitlines()]
    assert len(rows) == 4377
    plan = first.read(first.BETA2 / "protocol.json")
    old_plan = first.read(first.OLD / "v15_mmben_E128_streamwait.control.protocol.json")
    old_score = first.read(first.OLD / "v15_mmben_E128_streamwait.official.score.json")
    new_data_path, old_data_path = Path(plan["dataset"]["path"]), Path(old_plan["parameters"]["question_file"])
    new_data, old_data = pd.read_table(new_data_path), pd.read_table(old_data_path)
    assert first.sha(new_data_path) == plan["dataset"]["sha256"] and first.sha(old_data_path) == old_plan["origin"]["question_sha256"]
    old_raw_path = first.OLD / "v15_mmben_E128_streamwait.jsonl"
    old_raw = first.jsonl(old_raw_path, 4876, first.EXPECTED_OLD_RAW)
    old_gt = {int(item["index"]): item["answer"] for _, item in old_data.iterrows()}
    old_by_id = {int(row["question_id"]): dict(row, GT=old_gt[int(row["question_id"])]) for row in old_raw}
    renderer = first.conversation_renderer(Path(old_plan["origin"]["native_entry"]).parents[1] / "conversation.py", "stage2_old_conversation")
    candidates = [row for row in rows if row["index"] in old_by_id
                  and row["prompt"] == old_by_id[row["index"]]["prompt"]
                  and row["options"] == old_by_id[row["index"]]["options"]
                  and row["GT"] == old_by_id[row["index"]]["GT"]
                  and row["rendered_full_prompt"] == renderer(old_by_id[row["index"]]["prompt"])]
    assert len(candidates) == stage1["V11_same_beta2_descriptive_overlap"]["input_matching_stages"]["plus_exact_GT"]
    utils = first.ROOT / "_reproduction/author_eadp/LLaVA/llava/mm_utils.py"
    namespace = dict(Image=Image, torch=torch)
    tree = ast.parse(utils.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("expand2square", "process_images")]
    assert len(nodes) == 2
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(utils)+"[unchanged CPU helpers]","exec"),namespace)
    config_path = Path("/media/disk2/YZX/doct/FastV/llava-v1.5-7b/config.json")
    config = SimpleNamespace(**first.read(config_path)); assert config.image_aspect_ratio == "pad"
    assert first.sha(CLIP / "preprocessor_config.json") == EXPECTED_PREPROCESSOR
    processor = CLIPImageProcessor.from_pretrained(CLIP,local_files_only=True)
    assert processor.to_dict() == first.read(first.BETA2 / "observed/loaded_model.json")["image_processor"]
    verified_path = BASE / "cpu_inputs_author_pins.json"
    verified = first.read(verified_path)
    assert verified["image_helper_source_sha256"] == first.sha(utils)
    assert verified["dataset_sha256"] == first.sha(new_data_path) and verified["versions"]["Pillow"] == PIL.__version__
    maps = [payloads(data) for data in (old_data,new_data)]
    cache = {}; pair_rows=[]
    def fingerprint(payload):
        raw = base64.b64decode(payload); encoded = hashlib.sha256(raw).hexdigest()
        if encoded not in cache:
            image = Image.open(io.BytesIO(raw)); image.load()
            tensor = namespace["process_images"]([image],processor,config)[0].unsqueeze(0)
            assert tensor.dtype == torch.float32 and list(tensor.shape) == [1,3,336,336] and not tensor.is_cuda
            cache[encoded]=dict(encoded_sha256=encoded,size=list(image.size),mode=image.mode,
                                decoded_RGB_sha256=digest(np.asarray(image.convert("RGB"))),
                                FP32_sha256=digest(tensor.numpy()),FP16_sha256=digest(tensor.half().numpy()),shape=list(tensor.shape))
        return cache[encoded]
    for row in candidates:
        idx=row["index"]; old,new=[fingerprint(mapping[idx]) for mapping in maps]
        reference=verified["images"][new["encoded_sha256"]]
        assert (new["size"],new["mode"],new["decoded_RGB_sha256"],new["FP32_sha256"],new["FP16_sha256"],new["shape"]) == (reference["size"],reference["mode"],reference["decoded_rgb_sha256"],reference["float32_sha256"],reference["float16_sha256"],reference["shape"])
        same_size=old["size"]==new["size"]
        pair_rows.append(dict(index=idx,base_group=idx%1000000,GT=row["GT"],
            native_prompt_sha256=hashlib.sha256(row["prompt"].encode()).hexdigest(),
            full_prompt_sha256=hashlib.sha256(row["rendered_full_prompt"].encode()).hexdigest(),
            old=old,new=new,encoded_match=old["encoded_sha256"]==new["encoded_sha256"],
            decoded_RGB_and_size_match=same_size and old["decoded_RGB_sha256"]==new["decoded_RGB_sha256"],
            FP32_and_shape_and_original_size_match=same_size and old["shape"]==new["shape"] and old["FP32_sha256"]==new["FP32_sha256"],
            FP16_and_shape_and_original_size_match=same_size and old["shape"]==new["shape"] and old["FP16_sha256"]==new["FP16_sha256"],
            old_beta2_text=old_by_id[idx]["text"],new_beta2_text=row["beta2_text"],
            old_correct=first.letter(old_by_id[idx]["text"])==row["GT"],new_correct=row["beta2_correct"]))
    all_new,all_old=defaultdict(set),defaultdict(set)
    for row in rows:all_new[row["base_group"]].add(row["index"])
    for idx in old_by_id:all_old[idx%1000000].add(idx)
    tiers={}
    for key in ("encoded_match","decoded_RGB_and_size_match","FP32_and_shape_and_original_size_match","FP16_and_shape_and_original_size_match"):
        matches=[row for row in pair_rows if row[key]];ids={row["index"] for row in matches}
        complete=[group for group,indices in all_new.items() if indices==all_old.get(group) and indices<=ids]
        by_id={row["index"]:row for row in matches}
        groups=[dict(base_group=group,n_rotations=len(all_new[group]),
                     old_all_correct=all(by_id[i]["old_correct"] for i in all_new[group]),
                     new_all_correct=all(by_id[i]["new_correct"] for i in all_new[group])) for group in complete]
        tiers[key]=dict(matched_rows=len(matches),complete_identical_rotation_groups=len(complete),
                        exact_indices=sorted(ids),row_correctness_transitions=first.transitions([r["old_correct"] for r in matches],[r["new_correct"] for r in matches]),
                        raw_answer_changed_rows=sum(r["old_beta2_text"]!=r["new_beta2_text"] for r in matches),
                        old_correct_groups=sum(g["old_all_correct"] for g in groups),new_correct_groups=sum(g["new_all_correct"] for g in groups),complete_groups=groups)
    assert tiers["encoded_match"]["matched_rows"]==stage1["V11_same_beta2_descriptive_overlap"]["strict_matched_rows"]
    assert tiers["encoded_match"]["complete_identical_rotation_groups"]==stage1["V11_same_beta2_descriptive_overlap"]["strict_complete_circular_groups"]
    OUT.mkdir(parents=True,exist_ok=True)
    pair_path=OUT/"candidate_pair_fingerprints.jsonl"
    pair_path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in pair_rows))
    sources=[Path(__file__),Path(first.__file__),stage1_path,Path(stage1["paired_rows"]["path"]),verified_path,
             new_data_path,old_data_path,old_raw_path,utils,config_path,CLIP/"preprocessor_config.json"]
    report=dict(success=True,CPU_only=True,GPU_initialized=torch.cuda.is_initialized(),model_loaded=False,
        completed_utc=datetime.now(timezone.utc).isoformat(),new_full_rows=4377,old_full_rows=4876,
        prior_ID_prompt_options_GT_gate=stage1["V11_same_beta2_descriptive_overlap"]["input_matching_stages"],
        exact_ID_native_and_rendered_prompt_options_GT_candidates=len(candidates),image_input_tiers=tiers,
        unique_image_payloads_actually_processed=len(cache),all_new_candidate_fingerprints_match_prior_author_pinned_full_input_audit=True,
        reencoded_but_exact_decoded_RGB_size_rows=sum(not row["encoded_match"] and row["decoded_RGB_and_size_match"] for row in pair_rows),
        different_decoded_RGB_but_exact_native_FP16_rows=sum(not row["decoded_RGB_and_size_match"] and row["FP16_and_shape_and_original_size_match"] for row in pair_rows),
        same_FP16_but_different_FP32_rows=sum(not row["FP32_and_shape_and_original_size_match"] and row["FP16_and_shape_and_original_size_match"] for row in pair_rows),
        versions=dict(Pillow=PIL.__version__,PIL_module=PIL.__file__,torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__),
        official_CLIP_revision=CLIP.name,processor_config_sha256=EXPECTED_PREPROCESSOR,
        candidate_pair_fingerprints=dict(path=str(pair_path),sha256=first.sha(pair_path)),
        sources=[dict(path=str(path),sha256=first.sha(path)) for path in sources],
        limits=["Only exact full ID/prompt/options/GT candidates enter each image tier; image-size arguments and tensor shape are retained.",
                "FP16 refers to the actual native transfer dtype, reconstructed with unchanged source pad/CLIP preprocessing and isolated author Pillow10.4. No model or CUDA is initialized.",
                "The historical runtime did not record pixel-tensor bytes; this is fixed-source input reconstruction, not proof of every transitive historical environment detail.",
                "Complete circular subsets require the same full rotation-ID set in both datasets; partial groups never become full benchmark scores.",
                "Neither whole-dataset score difference nor matched-subset output difference alone is a causal dataset estimate or paper reproduction."])
    assert not report["GPU_initialized"]
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("sources","image_input_tiers")},ensure_ascii=False,indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a not in ("exact_indices","complete_groups")} for k,v in tiers.items()},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
