"""G4 — main-method (RTG+facility+Completion .25) 20-question gate."""
import json, os, sys, torch
LLAVA_ROOT = "/media/disk2/YZX/research/EADP_amp/LLaVA"
sys.path.insert(0, LLAVA_ROOT); sys.path.insert(0, LLAVA_ROOT + "/llava/model")
sys.path.insert(0, "/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/anchor_merge_pilot")
import llava_arch_anchorzip as AZ
AZ.install()   # monkey-patch encode_images with the AnchorZip path
AZ.MODE = "rtg"
import g0_llava_smoke as G0
G0.N = 20
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
disable_torch_init()
questions = [json.loads(q) for q in open(G0.QS)][:20]
tok, model, ip, _ = load_pretrained_model(G0.MODEL_PATH, None,
    get_model_name_from_path(G0.MODEL_PATH), visual_token_num=128,
    beta=2.0, alpha=0.5)
model.eval()
p1 = G0.run_pass(questions, tok, model, ip)
p2 = G0.run_pass(questions, tok, model, ip)
det = all(a["text"] == b["text"] for a, b in zip(p1, p2))
nonempty = all(o["text"] for o in p1)
vt = getattr(model.model, "visual_token_num", None)
res = dict(gate="G4", n=20, deterministic=bool(det), nonempty=bool(nonempty),
           visual_token_num=vt, verdict=bool(det and nonempty),
           outputs=p1)
out = G0.OUT.replace("g0_llava_smoke", "g4_llava_main_gate")
json.dump(res, open(out + ".tmp", "w"), indent=1, ensure_ascii=False)
os.replace(out + ".tmp", out)
print(json.dumps({k: v for k, v in res.items() if k != "outputs"}))
for o in p1[:5]: print(" ", o["question_id"], "->", repr(o["text"][:60]))
