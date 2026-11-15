"""Small necessary native-interface gates; does not select research arms."""
from pathlib import Path
import json
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runtime import common, torch, encode_vz, ppe_prefill
from model.native_qwen3 import NativeEngine


def main():
    out = Path(common.QWEN_ROOT)/"outputs/binding_oracle"
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((Path(common.QWEN_ROOT)/"outputs/cf_mining/manifest.json").read_text())["records"][0]
    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=24)
    eng = NativeEngine(model)
    msg = [dict(type="image",value=old["source_image"]),dict(type="text",value=old["question"]+"\nAnswer with the value only.")]
    prep = eng.prepare(msg,"DocVQA_VAL")
    with torch.no_grad():
        v0, ds0 = eng.encode(prep)
        v, ds, score, key = encode_vz(eng,prep)
        assert torch.equal(v0,v) and all(torch.equal(a,b) for a,b in zip(ds0,ds))
        keep = torch.arange(v.shape[0],device=v.device)
        native = eng.prefill(prep,v,ds,keep)
        reps = keep[:,None].expand(-1,4)
        with ppe_prefill(eng,prep,keep,reps):
            singleton = eng.prefill(prep,v,ds,keep)
        maxdiff = (native.logits-singleton.logits).abs().max().item()
        assert torch.equal(native.logits,singleton.logits), maxdiff
        # Direct rotary gate also checks that non-singletons affect image only.
        pos = eng.positions_full(prep)[0]
        emb = eng.inner.get_input_embeddings()(prep["ids"])
        stock_cos, stock_sin = eng.text.rotary_emb(emb,pos)
        shifted = reps.clone()
        shifted[:,1:] = ((shifted[:,1:]+1)%keep.numel())
        with ppe_prefill(eng,prep,keep,shifted):
            mod_cos, mod_sin = eng.text.rotary_emb(emb,pos)
        mask = prep["ids"][0] == eng.image_token_id
        assert torch.equal(stock_cos[:,~mask],mod_cos[:,~mask])
        assert torch.equal(stock_sin[:,~mask],mod_sin[:,~mask])
        assert not torch.equal(stock_cos[:,mask],mod_cos[:,mask])
        # Distinct image assignment, but ordinary cache/decode positions.
        with ppe_prefill(eng,prep,keep,shifted):
            changed = eng.prefill(prep,v,ds,keep)
        eng.decode(changed,3)
        assert not changed.decode_positions or changed.decode_positions[0] == changed.prefill_max_pos+1
        result = dict(pass_all=True, stock_hook_features_bitwise=True,
            singleton_ppe_logits_bitwise=True,max_logit_delta=maxdiff,
            text_rotary_unchanged=True,image_rotary_changed=True,
            native_decode_position=True, n_vis=prep["n_vis"],
            deepstack_lengths=[len(x) for x in ds],score_finite=bool(torch.isfinite(score).all()),
            score_shape=list(score.shape),key_shape=list(key.shape),
            torch_version=torch.__version__,time=time.time(),
            source="previous-pilot page, used only for interface smoke")
        assert result["score_finite"]
    (out/"gpu_gates.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
