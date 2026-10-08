import argparse
import torch
import os
import sys
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

LLAVA_ROOT = "/media/disk2/YZX/research/EADP_amp/LLaVA"


def generation_text_settings(cur_prompt, official_textvqa=False):
    """Match the official TextVQA loader without changing the LLM prompt."""
    if official_textvqa:
        return cur_prompt.replace(
            "\nAnswer the question using a single word or phrase.", ""), 128
    return cur_prompt, 1024


def write_official_textvqa_protocol(args):
    """Refuse to overwrite an existing prediction; record this new run."""
    from pathlib import Path
    import hashlib
    import datetime
    import subprocess

    answers = Path(os.path.expanduser(args.answers_file))
    sidecar = Path(str(answers) + ".protocol.json")
    if answers.exists() or sidecar.exists():
        raise FileExistsError(
            f"Official TextVQA needs a new artifact path; existing file: {answers}")
    question_path = Path(os.path.expanduser(args.question_file))
    questions = [json.loads(s) for s in question_path.open()]
    suffix = "\nAnswer the question using a single word or phrase."
    if not questions or not all("Reference OCR token: " in r.get("text", "")
                                and r["text"].endswith(suffix) for r in questions):
        raise ValueError("--official-textvqa requires the official TextVQA OCR question file")
    commit = subprocess.check_output(
        ["git", "-C", LLAVA_ROOT, "rev-parse", "HEAD"], text=True).strip()
    metadata = dict(
        protocol="official_textvqa_loader", task="TextVQA_VAL",
        model_path=os.path.realpath(os.path.expanduser(args.model_path)),
        question_file=str(question_path.resolve()),
        question_file_sha256=hashlib.sha256(question_path.read_bytes()).hexdigest(),
        question_rows=len(questions), num_chunks=args.num_chunks, chunk_idx=args.chunk_idx,
        image_folder=os.path.realpath(args.image_folder),
        answers_file=str(answers.resolve()), conv_mode=args.conv_mode,
        score_text_transform="remove exact single-word-or-phrase answer suffix",
        llm_prompt_transform="unchanged full TextVQA prompt, including OCR and answer suffix",
        max_new_tokens=128, temperature=args.temperature, top_p=args.top_p,
        num_beams=args.num_beams, visual_token_num=args.visual_token_num,
        alpha=args.alpha, beta=args.beta, anchorzip=bool(args.anchorzip),
        git_commit=commit,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__, transformers_version=__import__("transformers").__version__,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for basename in ("config.json", "generation_config.json"):
        config_path = Path(metadata["model_path"]) / basename
        if config_path.is_file():
            metadata[basename + "_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    source_paths = ["llava/model/llava_arch.py", "llava/model/builder.py",
                    "llava/model/multimodal_encoder/clip_encoder.py",
                    "llava/mm_utils.py", "llava/conversation.py"]
    if args.anchorzip:
        source_paths.append("llava/model/llava_arch_anchorzip.py")
        source_paths.append("../Qwen_vl/scripts/anchor_merge_pilot/amp_common.py")
    metadata["source_sha256"] = {
        rel: hashlib.sha256((Path(LLAVA_ROOT) / rel).read_bytes()).hexdigest()
        for rel in source_paths}
    answers.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(sidecar) + ".tmp")
    tmp.write_text(json.dumps(metadata, indent=2))
    os.replace(tmp, sidecar)
    return metadata


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    official_textvqa = getattr(args, "official_textvqa", False)
    protocol = write_official_textvqa_protocol(args) if official_textvqa else None
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        visual_token_num=args.visual_token_num,
        beta=args.beta,
        alpha=args.alpha,
    )

    if getattr(args, "anchorzip", False):
        sys.path.insert(0, LLAVA_ROOT + "/llava/model")
        sys.path.insert(0, "/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/anchor_merge_pilot")
        import llava_arch_anchorzip as _AZ
        _AZ.MODE = "rtg"
        _AZ.install()

    # Data
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # Configure tqdm to be less verbose
    data_bar = tqdm(questions)
    for line in data_bar:
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        question, max_new_tokens = generation_text_settings(cur_prompt, official_textvqa)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids, visual_token_num = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                texts=question,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=max_new_tokens,
                use_cache=True)
            if hasattr(model.model, 'visual_token_num'):
                visual_token_num = model.model.visual_token_num
            data_bar.set_postfix(vtn=f"{visual_token_num}")

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": ({"protocol": protocol["protocol"],
                                                 "max_new_tokens": 128,
                                                 "score_text_suffix_removed": True}
                                                if protocol is not None else {})}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--visual_token_num", type=int, default=576)
    parser.add_argument("--anchorzip", action="store_true")
    parser.add_argument("--official-textvqa", action="store_true",
                        help="Use official TextVQA score texts and 128-token decoding; requires a new answers path")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    eval_model(args)
