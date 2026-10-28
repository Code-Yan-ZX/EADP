#!/usr/bin/env bash
set -u
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
HOLD=/media/disk2/tmp_hold
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
dl () { [ -s "$2" ] && { echo "[skip] $2"; return 0; }
  wget -c -q -O "$2.part" "$1" && mv "$2.part" "$2" && echo "[ok] $2" \
    || { echo "[FAIL] $1"; return 1; }; }
echo "== mmvet =="
mkdir -p $EV/mmvet
dl https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip $EV/mmvet/mm-vet.zip && unzip -nq -o $EV/mmvet/mm-vet.zip -d $EV/mmvet && echo "[ok] mmvet extracted"
echo "== mmbench =="
dl https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv $EV/mmbench/mmbench_dev_20230712.tsv
echo "== scienceqa (gdrive via proxy) =="
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
mkdir -p $EV/scienceqa_img && cd $EV/scienceqa_img
$PY -m gdown 1E9KFdUcY8DOuYLYcA_UiM4X8tnCJ8xGV -O data.zip 2>&1 | tail -1 || echo "[FAIL] scienceqa gdrive"
unset http_proxy https_proxy
[ -s $EV/scienceqa_img/data.zip ] && (cd $EV/scienceqa_img && unzip -nq -o data.zip) && echo "[ok] scienceqa extracted"
cd /media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
echo "== gqa images (direct) =="
dl https://downloads.cs.stanford.edu/nlp/data/gqa/images1207.zip $EV/gqa/images1207.zip && unzip -nq -o $EV/gqa/images1207.zip -d $EV/gqa/data && echo "[ok] gqa extracted"
echo "== test2015 (direct) =="
dl http://images.cocodataset.org/zips/test2015.zip $HOLD/test2015.zip && unzip -nq -o $HOLD/test2015.zip -d $EV/vqav2 && echo "[ok] test2015 extracted"
echo "[done] $(date)"
