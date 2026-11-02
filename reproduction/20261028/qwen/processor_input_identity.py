"""Real processor-only fingerprints, with no model construction or CUDA access."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import sys
import types


HERE = Path(__file__).resolve().parent
MODEL = Path("/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
SMOKE = HERE / "author_fa2_smoke/answers.jsonl"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ast_sha(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def source_preprocessing(root, torch, Image):
    """Execute source preprocessing, returning before its model-specific device move.

    Real qwen_vl_utils, PIL and AutoProcessor calls are unchanged. The CPU-only
    extraction records references to the two intermediate PIL lists; it creates
    no dummy model, fake processor, synthetic image, or replacement tensor.
    """
    base = root / "VLMEvalKit/vlmeval/vlm/qwen3_vl"
    fixed, model = base / "model_fixed_res.py", base / "model.py"
    tree = ast.parse(fixed.read_text())
    expand = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "expand2square")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3VLChatFixedRes")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    original = methods["_processor_inputs"]
    extracted = copy.deepcopy(original)
    assert ast.unparse(extracted.body[-1]) == "return self._move_inputs_to_model(inputs)"
    extracted.args.args.append(ast.arg(arg="stages"))
    # Preserve the exact arithmetic and order. Only post-call reference captures
    # and the final return boundary differ from the complete generation wrapper.
    restored = copy.deepcopy(extracted)
    restored.args.args.pop()
    assert ast.dump(restored) == ast.dump(original)
    marker_positions = []
    for index, node in enumerate(extracted.body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Tuple) for t in node.targets):
            if isinstance(node.value, ast.Call) and getattr(node.value.func, "id", None) == "process_vision_info":
                marker_positions.append((index + 1, ast.parse("stages['after_process_vision_info'] = images").body[0]))
        if isinstance(node, ast.If) and ast.unparse(node.test) == "images is not None":
            marker_positions.append((index + 1, ast.parse("stages['after_square_1024_resize'] = images").body[0]))
    assert len(marker_positions) == 2
    for index, node in reversed(marker_positions):
        extracted.body.insert(index, node)
    extracted.body[-1] = ast.parse("return inputs, text").body[0]
    inverse = copy.deepcopy(extracted)
    inverse.body = [n for n in inverse.body if not (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript) and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "stages")]
    inverse.body[-1] = copy.deepcopy(original.body[-1]); inverse.args.args.pop()
    assert ast.dump(inverse) == ast.dump(original), "Removing observations must restore the original preprocessing AST"
    model_tree = ast.parse(model.read_text())
    ensure = [n for n in model_tree.body if isinstance(n, ast.FunctionDef) and n.name in ("ensure_image_url", "ensure_video_url")]
    namespace = dict(Image=Image, torch=torch, logging=logging, os=os, QWEN3_FIXED_RESOLUTION=1024)
    code = ast.Module(body=[copy.deepcopy(expand), *copy.deepcopy(ensure),
                           copy.deepcopy(methods["_prepare_content"]), copy.deepcopy(methods["_build_messages"]), extracted], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code),str(fixed)+"[processor-only original statements]","exec"),namespace)
    proof = dict(source=str(fixed), source_sha256=sha(fixed), model_source_sha256=sha(model),
                 original_processor_AST_sha256=ast_sha(original), extracted_processor_AST_sha256=ast_sha(extracted),
                 observation_removed_restores_original_AST=True,
                 changes=["Two references to intermediate PIL image lists", "Return actual processor inputs before model-device move; CPU BF16 conversion separately uses real BatchFeature.to"])
    return namespace, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--environment-label",required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only execution is mandatory"
    import numpy as np
    import torch
    import PIL
    from PIL import Image
    from transformers import AutoProcessor
    import qwen_vl_utils
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    assert not torch.cuda.is_initialized()
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    namespace, proof = source_preprocessing(args.source_root,torch,Image)
    # A real processor context, without constructing the generation/model class.
    context = types.SimpleNamespace(processor=processor,min_pixels=1048576,max_pixels=1048576,system_prompt=None,fps=2)
    for name in ("_prepare_content","_build_messages","_processor_inputs"):
        setattr(context,name,types.MethodType(namespace[name],context))
    records=[json.loads(line) for line in SMOKE.read_text().splitlines()]
    assert len(records)==4 and {(r['dataset'],r['position']) for r in records}=={('DocVQA_VAL',0),('DocVQA_VAL',2674),('AI2D_TEST',0),('AI2D_TEST',1544)}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report=dict(CPU_only=True,model_loaded=False,scope="Fixed four existing smoke questions; current dependency stack, not a historical environment reconstruction",
                environment_label=args.environment_label,python=sys.executable,created_utc=datetime.now(timezone.utc).isoformat(),
                versions={name:importlib.metadata.version(name) for name in ('torch','transformers','qwen-vl-utils','Pillow','tokenizers','numpy')},
                processor_class=type(processor).__name__,image_processor_class=type(processor.image_processor).__name__,
                image_processor_config=processor.image_processor.to_dict(),source_AST_proof=proof,
                model_processor_files={p.name:sha(p) for p in MODEL.iterdir() if p.is_file() and p.suffix in ('.json','.jinja') and p.stat().st_size<30*1024*1024},
                loaded_module_files={name:getattr(sys.modules[name],'__file__',None) for name in ('torch','transformers','qwen_vl_utils','PIL')},
                processor_source={name:dict(path=inspect.getfile(type(obj)),sha256=sha(inspect.getfile(type(obj)))) for name,obj in [('processor',processor),('image_processor',processor.image_processor),('tokenizer',processor.tokenizer)]},
                smoke_file=dict(path=str(SMOKE),sha256=sha(SMOKE)),records=[])
    def array_record(array):
        value=np.ascontiguousarray(array)
        return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    def image_record(image):
        rgb=np.asarray(image.convert('RGB'))
        return dict(size=list(image.size),mode=image.mode,RGB=array_record(rgb))
    for number,record in enumerate(records):
        for path,expected in record['image_sha256'].items():assert sha(path)==expected
        stages={}; messages=context._build_messages(record['message'],dataset=record['dataset'])
        inputs,text=context._processor_inputs(messages,stages)
        arrays={key:value.detach().cpu().contiguous().numpy() for key,value in inputs.items() if isinstance(value,torch.Tensor)}
        raw_metadata={key:array_record(value) for key,value in arrays.items()}
        # Native model is BF16; BatchFeature.to preserves integer IDs/grid/mask.
        converted=inputs.to('cpu').to(torch.bfloat16)
        native_metadata={}
        for key,value in converted.items():
            if isinstance(value,torch.Tensor):
                actual=value.detach().cpu().contiguous()
                if actual.dtype==torch.bfloat16:
                    bits=actual.view(torch.uint16).numpy();arrays['native_'+key+'_BF16_bits']=bits
                    native_metadata[key]=dict(dtype=str(actual.dtype),shape=list(actual.shape),sha256=hashlib.sha256(bits.tobytes()).hexdigest())
                else:
                    native_metadata[key]=dict(dtype=str(actual.dtype),**{k:v for k,v in array_record(actual.numpy()).items() if k!='dtype'})
        destination=args.output_dir/('question_%02d.npz'%number)
        np.savez_compressed(destination,**arrays)
        item=dict(dataset=record['dataset'],position=record['position'],index=record['index'],
                  message=record['message'],source_messages=messages,rendered_chat_template=text,
                  rendered_chat_template_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  intermediate_images={key:[image_record(image) for image in images] for key,images in stages.items()},
                  raw_processor_tensors=raw_metadata,native_CPU_BF16_tensors=native_metadata,
                  tensor_arrays=dict(path=str(destination),sha256=sha(destination)))
        report['records'].append(item)
        print(json.dumps(dict(completed=len(report['records']),dataset=record['dataset'],position=record['position'])),flush=True)
    report.update(CUDA_initialized=torch.cuda.is_initialized(),FA2_modules_imported=[name for name in sys.modules if name.startswith('flash_attn') or name=='flash_attn_2_cuda'])
    assert not report['CUDA_initialized'] and not report['FA2_modules_imported']
    report['script_sha256']=sha(__file__)
    (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='complete',n=4,output=str(args.output_dir/'report.json'),CUDA_initialized=False,FA2_imported=False)))


if __name__=='__main__':main()
