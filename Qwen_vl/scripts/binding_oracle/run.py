"""Resumable generation pilot; no training and no selection on student errors."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from runtime import common, torch, encode_vz, predict, representatives, ppe_prefill
from binding_selectors import select_binding
from model.native_qwen3 import NativeEngine
sys.path.insert(0,str(HERE.parent/"cf_mining"))
from cf_mine import correct, normalize, atomic_json

OUT = Path(common.QWEN_ROOT)/"outputs/binding_oracle"
ARMS = {
    "visionzip": ("baseline_fixed",False),
    "visionzip_ppe": ("baseline_fixed",True),
    "field_anchor_control_ppe": ("field_anchor_control",True),
    "field_oracle": ("field_oracle",False),
    "field_oracle_ppe": ("field_oracle",True),
    "random_oracle_ppe": ("random_oracle",True),
    "spatial_oracle_ppe": ("spatial_oracle",True),
}
AUDIT = {
    "historical": ("historical_exact",False),
    "historical_ds_fixed": ({"name":"baseline_fixed","dominant_num":238},False),
}
SENSITIVITY = {
    "line_anchor_control_ppe": ("line_anchor_control",True),
    "line_oracle_ppe": ("line_oracle",True),
}


@contextmanager
def resolution(eng, pixels):
    import vlmeval.vlm.qwen3_vl.model_fixed_res as fixed
    previous = fixed.QWEN3_FIXED_RESOLUTION, eng.vlm.min_pixels, eng.vlm.max_pixels
    fixed.QWEN3_FIXED_RESOLUTION = pixels
    eng.vlm.min_pixels = eng.vlm.max_pixels = pixels*pixels
    try:
        yield
    finally:
        fixed.QWEN3_FIXED_RESOLUTION, eng.vlm.min_pixels, eng.vlm.max_pixels = previous


def message(path, question):
    return [dict(type="image",value=str(path)),dict(type="text",value=question+"\nAnswer with the value only.")]


def labels_for(item, device):
    # A box can touch the same 32px token as a neighboring value. Resolve this
    # using larger intersection area, independent of every model prediction.
    from qwen_vl_utils.vision_process import smart_resize
    W,H=item["source_image_size"]
    fetched_h,fetched_w=smart_resize(H,W,factor=32,min_pixels=1024**2,max_pixels=1024**2)
    square=max(fetched_w,fetched_h)
    padx=(square-fetched_w)//2
    pady=(square-fetched_h)//2
    native_boxes={}
    raster=torch.arange(1024,device=device)
    left=(raster%32)*32
    top=(raster//32)*32
    def partition(key):
        areas=[]
        for field in item["fields"]:
            raw=field["value_bbox" if key=="value_region_1024" else "context_line_bbox"]
            x,y,w,h=raw
            x0=(x*fetched_w/W+padx)*1024/square
            x1=((x+w)*fetched_w/W+padx)*1024/square
            y0=(y*fetched_h/H+pady)*1024/square
            y1=((y+h)*fetched_h/H+pady)*1024/square
            native_boxes.setdefault(key,[]).append([x0,y0,x1,y1])
            w=(torch.minimum(left+32,left.new_tensor(x1,dtype=torch.float32))-
               torch.maximum(left,left.new_tensor(x0,dtype=torch.float32))).clamp_min(0)
            h=(torch.minimum(top+32,top.new_tensor(y1,dtype=torch.float32))-
               torch.maximum(top,top.new_tensor(y0,dtype=torch.float32))).clamp_min(0)
            area=w*h
            areas.append(area)
        areas=torch.stack(areas)
        labels=areas.argmax(0)+1
        labels[areas.max(0).values==0]=0
        return labels,int(((areas>0).sum(0)>1).sum())
    value,collisions=partition("value_region_1024")
    line,linecollisions=partition("context_line_region_1024")
    if any(not bool((value==i).any()) for i in (1,2)):
        raise ValueError("value region has no distinct token after geometry partition")
    return {"labels":value,"line_labels":line,"value_collisions":collisions,
            "line_collisions":linecollisions,"native_boxes":native_boxes,
            "geometry_version":"native_smart_resize_then_square_v2",
            "fetch_size":[fetched_w,fetched_h],"source_size":[W,H]}


def plain_diag(diag):
    return {k:v for k,v in diag.items() if k != "tensors"}


def save_diag(image,variant,selection,regions,V):
    """Save invariant per-image membership once, outside prediction records."""
    digest=hashlib.sha256((str(image)+json.dumps(variant,sort_keys=True)+regions["geometry_version"]).encode()).hexdigest()
    path=OUT/"diagnostics"/(digest+".json")
    if not path.exists():
        keep,_,_,diag=selection
        reps=representatives(V,diag)
        field_labels=regions["labels"]
        rep_labels=field_labels[reps]
        payload=plain_diag(diag)
        payload.update(image=str(image),variant_config=variant,
                       ppe_representatives=reps.tolist(),
                       ppe_representative_field_labels=rep_labels.tolist(),
                       field_tokens_kept=[int((field_labels[keep]==i).sum()) for i in (1,2)],
                       ppe_field_position_slots=[int((rep_labels==i).sum()) for i in (1,2)],
                       field_token_counts=[int((field_labels==i).sum()) for i in (1,2)],
                       value_token_collisions=regions["value_collisions"],
                       line_token_collisions=regions["line_collisions"],
                       native_boxes=regions["native_boxes"],
                       fetch_size=regions["fetch_size"],geometry_version=regions["geometry_version"])
        atomic_json(path,payload)
    return str(path)


def features(eng, prep, image, pixels=1024):
    # Include file hash and resolution; never reuse one member's encoding for
    # its sibling. Question text does not enter the frozen vision encoder.
    digest = hashlib.sha256(Path(image).read_bytes()).hexdigest()
    path = OUT/"cache"/f"{digest}_{pixels}.pt"
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        data = torch.load(path,map_location="cpu",weights_only=False)
    else:
        if pixels == 1024:
            V,DS,attn,key = encode_vz(eng,prep)
        else:
            V,DS = eng.encode(prep)
            attn = key = None
        data = dict(V=V.cpu(),DS=[x.cpu() for x in DS],
                    attn=None if attn is None else attn.cpu(),
                    key=None if key is None else key.cpu(),
                    gthw=prep["gthw"].cpu(),image_sha256=digest,pixels=pixels)
        torch.save(data,path)
    device = prep["ids"].device
    if not torch.equal(data["gthw"],prep["gthw"].cpu()):
        raise ValueError("feature grid changed")
    return (data["V"].to(device),[x.to(device) for x in data["DS"]],
            None if data["attn"] is None else data["attn"].to(device),
            None if data["key"] is None else data["key"].to(device))


def one(eng,item,side,question,answers,arms,existing,max_tokens,seed):
    image = item[side+"_image"]
    if all(arm in existing for arm in arms):
        return existing
    msg = message(image,question)
    prep = eng.prepare(msg,"DocVQA_VAL")
    V,DS,attn,key = features(eng,prep,image)
    if V.shape[0] != 1024 or prep["n_vis"] != 1024:
        raise ValueError("expected 1024 image tokens")
    regions = (labels_for(item,V.device)
               if any(a not in ("teacher","rres512") and a not in existing for a in arms)
               else None)
    selections = {}
    for arm in arms:
        if arm in existing:
            continue
        if arm == "teacher":
            pred,info = predict(eng,prep,V,DS,max_tokens=max_tokens)
        elif arm == "rres512":
            with resolution(eng,512):
                small = eng.prepare(msg,"DocVQA_VAL")
                Vs,DSs,_,_ = features(eng,small,image,512)
                assert small["n_vis"] == Vs.shape[0] == 256
                pred,info = predict(eng,small,Vs,DSs,max_tokens=max_tokens)
            assert info["n_vis"] == 256
        else:
            variant,ppe = {**ARMS,**AUDIT,**SENSITIVITY}[arm]
            label = json.dumps(variant,sort_keys=True)
            if label not in selections:
                selections[label] = select_binding(V,DS,attn,key,256,None,regions,variant,seed)
            selection = selections[label]
            pred,info = predict(eng,prep,V,DS,selection,ppe=ppe,max_tokens=max_tokens)
            diag=selection[3]
            info["selector"] = {k:diag[k] for k in ("K","dominant_num","contextual_num",
                "replacement_count","unassigned_count","cross_label_assignees")}
            info["selector"]["diagnostic_path"]=save_diag(image,variant,selection,regions,V)
        existing[arm] = dict(prediction=pred,correct=any(correct(pred,a) for a in answers),
                             answers=answers,**info)
    return existing


@torch.no_grad()
def integration_gates(eng,item,seed):
    image=item["original_image"]
    prep=eng.prepare(message(image,item["fields"][0]["question"]),"DocVQA_VAL")
    V,DS,attn,key=features(eng,prep,image)
    regions=labels_for(item,V.device)
    plain=select_binding(V,DS,attn,key,256,None,regions,"baseline_fixed",seed)
    control=select_binding(V,DS,attn,key,256,None,regions,"field_anchor_control",seed)
    oracle=select_binding(V,DS,attn,key,256,None,regions,"field_oracle",seed)
    assert torch.equal(control[0],oracle[0])
    assert torch.equal(control[3]["tensors"]["dominant_idx"],oracle[3]["tensors"]["dominant_idx"])
    # Singleton/native check at the ACTUAL pruned sequence length, not only full N.
    keep=plain[0]
    native=eng.prefill(prep,V,DS,keep)
    with ppe_prefill(eng,prep,keep,keep[:,None].expand(-1,4)):
        singleton=eng.prefill(prep,V,DS,keep)
    assert torch.equal(native.logits,singleton.logits)
    checks={}
    for name,sel in (("visionzip",plain),("field_anchor_control",control),("field_oracle",oracle)):
        reps=representatives(V,sel[3])  # asserts every source belongs to its cluster
        pred,info=predict(eng,prep,V,DS,sel,ppe=True,max_tokens=3)
        assert info["n_vis"] == 256 and info["ds_lengths"] == [256]*3
        checks[name]=dict(passed=True,reps_shape=list(reps.shape),
                          n_vis=info["n_vis"],ds_lengths=info["ds_lengths"],
                          unassigned=sel[3]["unassigned_count"])
    atomic_json(OUT/"integration_gates.json",dict(pass_all=True,
        singleton_ppe_K256_logits_bitwise=True,same_field_anchors=True,
        same_field_dominants=True,checks=checks,
        source_pair=item["pair_id"],prediction_used_for_selection=False))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--limit",type=int,default=60)
    ap.add_argument("--stage",choices=["screen","compare","all"],default="all")
    ap.add_argument("--source-pages",type=int,default=24)
    ap.add_argument("--max-tokens",type=int,default=24)
    ap.add_argument("--seed",type=int,default=20261108)
    args=ap.parse_args()
    manifest=json.loads(Path(args.manifest).read_text())
    items=manifest["records"][:args.limit]
    result_path=OUT/"predictions.json"
    result=json.loads(result_path.read_text()) if result_path.exists() else dict(
        schema_version=1,records=[],sources=[],seed=args.seed,started=time.time(),
        base_commit="a724122d",K=256,input_tokens=1024,
        model=common.BASELINE_MODEL,ppe_slots=4,manifest_history=[])
    # Do not allow silent relabeling of previously predicted pairs.
    fingerprints=result.setdefault("item_fingerprints",{})
    for item in items:
        fingerprint=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
        if item["pair_id"] in fingerprints and fingerprints[item["pair_id"]] != fingerprint:
            raise ValueError("manifest pair changed after predictions: "+item["pair_id"])
        fingerprints[item["pair_id"]]=fingerprint
    result["manifest_history"].append(dict(path=str(Path(args.manifest).resolve()),time=time.time(),n=len(items)))
    result["active_pair_ids"]=[x["pair_id"] for x in items]
    result["stage"]="running_"+args.stage
    atomic_json(result_path,result)
    model=common.load_model(common.BASELINE_MODEL,max_new_tokens=args.max_tokens)
    eng=NativeEngine(model)
    byid={x["pair_id"]:x for x in result["records"]}
    started=time.time()
    if args.stage in ("screen","all"):
        for index,item in enumerate(items):
            row=byid.get(item["pair_id"])
            if row is None:
                row=dict(pair_id=item["pair_id"],source_document_id=item["source_document_id"],
                         split=item["split"],members={})
                byid[item["pair_id"]]=row
                result["records"].append(row)
            for side in ("original","swapped"):
                for fi,field in enumerate(item["fields"]):
                    key=side+"_f"+str(fi)
                    pp=row["members"].setdefault(key,{})
                    one(eng,item,side,field["question"],field["answers_"+side],
                        ["teacher"],pp,args.max_tokens,args.seed)
                    atomic_json(result_path,result)
            row["teacher_valid"]=all(x["teacher"]["correct"] for x in row["members"].values())
            atomic_json(result_path,result)
            print(f"[screen {index+1}/{len(items)}] {item['pair_id']} valid={row['teacher_valid']} "
                  f"accepted={sum(bool(x.get('teacher_valid')) for x in result['records'])} elapsed={time.time()-started:.1f}",flush=True)
    if args.stage in ("compare","all"):
        integration_gates(eng,items[0],args.seed)
        for index,item in enumerate(items):
            row=byid.get(item["pair_id"])
            if row is None or "teacher_valid" not in row:
                raise ValueError("screen must finish before compare")
            if not row["teacher_valid"]:
                continue
            arms=list(ARMS)+["rres512"]+(list(AUDIT) if index<12 else [])
            if item.get("eligible_for_line_oracle"):
                arms+=list(SENSITIVITY)
            for side in ("original","swapped"):
                for fi,field in enumerate(item["fields"]):
                    pp=row["members"][side+"_f"+str(fi)]
                    one(eng,item,side,field["question"],field["answers_"+side],
                        arms,pp,args.max_tokens,args.seed)
                    atomic_json(result_path,result)
            row["compared"]=True
            print(f"[compare {index+1}/{len(items)}] {item['pair_id']} "+
                  " ".join(f"{a}={sum(v[a]['correct'] for v in row['members'].values())}/4" for a in arms)+
                  f" elapsed={time.time()-started:.1f}",flush=True)
            atomic_json(result_path,result)
        sourcebyid={x["pair_id"]:x for x in result["sources"]}
        for item in items[:args.source_pages]:
            row=sourcebyid.get(item["pair_id"])
            if row is None:
                row=dict(pair_id=item["pair_id"],source_document_id=item["source_document_id"],questions={})
                result["sources"].append(row)
            for fi,field in enumerate(item["fields"]):
                pp=row["questions"].setdefault("f"+str(fi),{})
                one(eng,item,"source",field["question"],field["answers_original"],
                    ["teacher"]+list(ARMS)+["rres512"],pp,args.max_tokens,args.seed)
                atomic_json(result_path,result)
            print(f"[source] {item['pair_id']} elapsed={time.time()-started:.1f}",flush=True)
    result["stage"]="complete_"+args.stage
    result["updated"]=time.time()
    atomic_json(result_path,result)


if __name__=="__main__":
    main()
