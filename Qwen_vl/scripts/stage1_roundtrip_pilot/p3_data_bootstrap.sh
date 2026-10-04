#!/usr/bin/env bash
# P3 data bootstrap (2026-10-04): LLaVA-1.5/NeXT 10-benchmark eval data.
# Direct HTTP sources first; eval.zip via gdown (Google Drive).
# Downloads to LLaVA/playground/data/eval <downloads> staging, extracted
# in place.  Logs to p3_data_download.log.  Idempotent (skip existing).
set -u
cd /media/disk2/YZX/research/EADP_amp/LLaVA || exit 1
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
EV=playground/data/eval
mkdir -p $EV

dl () {  # dl <url> <dest>
  [ -f "$2" ] && { echo "[skip] $2"; return 0; }
  wget -c -q --show-progress -O "$2.part" "$1" && mv "$2.part" "$2" \
    && echo "[ok] $2" || echo "[FAIL] $1"
}

# ---- eval.zip (LLaVA custom annotations/scripts; Google Drive) ----------
$PY -m pip install -q gdown 2>/dev/null
[ -f $EV/eval.zip ] || gdown --fuzzy -O $EV/eval.zip \
  "https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing" \
  && (cd $EV && unzip -nq eval.zip) && echo "[ok] eval.zip extracted"

# ---- TextVQA (val json + trainval images ~20GB) -------------------------
mkdir -p $EV/textvqa
dl https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json \
   $EV/textvqa/TextVQA_0.5.1_val.json
dl https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip \
   $EV/textvqa/train_val_images.zip \
  && (cd $EV/textvqa && unzip -nq train_val_images.zip) \
  && echo "[ok] textvqa images extracted"

# ---- COCO val2014 (POPE/MME/pope-style) + test2015 (VQAv2) --------------
mkdir -p $EV/pope $EV/vqav2 $EV/mme
dl http://images.cocodataset.org/zips/val2014.zip /tmp/val2014.zip \
  && unzip -nq /tmp/val2014.zip -d $EV/pope/data \
  && mv $EV/pope/data/val2014 $EV/pope/data/val2014 2>/dev/null || true
dl http://images.cocodataset.org/zips/test2015.zip /tmp/test2015.zip \
  && unzip -nq /tmp/test2015.zip -d $EV/vqav2 && echo "[ok] test2015"
dl https://dl.fbaipublicfiles.com/textvqa/data/datasets/TextVQA_0.5.1_val.json \
   $EV/textvqa/TextVQA_0.5.1_val.json

# ---- GQA ---------------------------------------------------------------
mkdir -p $EV/gqa/data
dl https://downloads.cs.stanford.edu/nlp/data/gqa/images1207.zip \
   $EV/gqa/images1207.zip \
  && (cd $EV/gqa && unzip -nq images1207.zip -d data/images) \
  && echo "[ok] gqa images"
dl https://downloads.cs.stanford.edu/nlp/data/gqa/questions.zip \
   $EV/gqa/questions.zip \
  && (cd $EV/gqa && unzip -nq questions.zip -d data) && echo "[ok] gqa questions"

# ---- ScienceQA ----------------------------------------------------------
mkdir -p $EV/scienceqa
dl https://downloads.cs.stanford.edu/nlp/data/scienceqa/data.zip \
   $EV/scienceqa/data.zip \
  && (cd $EV/scienceqa && unzip -nq data.zip) && echo "[ok] scienceqa"

# ---- VizWiz val (local validation per EVAL.md) --------------------------
mkdir -p $EV/vizwiz
dl https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip \
   $EV/vizwiz/Annotations.zip \
  && (cd $EV/vizwiz && unzip -nq Annotations.zip) && echo "[ok] vizwiz ann"
dl https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip \
   $EV/vizwiz/val.zip \
  && (cd $EV/vizwiz && unzip -nq val.zip) && echo "[ok] vizwiz val images"

echo "[done] $(date)"
