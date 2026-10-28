#!/usr/bin/env bash
# P3 data fix round 4 (2026-10-05): ONE serial proxy download chain.
# Order: val2014 -> gqa images -> mme (HF) -> scienceqa (HF) -> test2015
# (test2015 last: VQAv2 testdev needs the eval.ai server anyway)
# Each step extracts after download.  Single wget at a time — the
# multi-wget same-file corruption seen earlier cannot recur.
set -u
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
HOLD=/media/disk2/tmp_hold
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval

dl () { [ -s "$2" ] && { echo "[skip] $2"; return 0; }
  wget -c -q -O "$2.part" "$1" && mv "$2.part" "$2" && echo "[ok] $2" \
    || { echo "[FAIL] $1"; return 1; }; }

echo "== val2014 =="
dl http://images.cocodataset.org/zips/val2014.zip $HOLD/val2014.zip \
  && unzip -nq -o $HOLD/val2014.zip -d $EV/pope/data \
  && echo "[ok] val2014 extracted"

echo "== gqa images =="
dl https://downloads.cs.stanford.edu/nlp/data/gqa/images1207.zip \
   $EV/gqa/images1207.zip \
  && unzip -nq -o $EV/gqa/images1207.zip -d $EV/gqa/data \
  && echo "[ok] gqa images extracted"

echo "== mme (HF) =="
mkdir -p $EV/mme_img
$PY -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('lmms-lab/MME', 'MME_Benchmark.zip', repo_type='dataset',
                    local_dir='$EV/mme_img')
print('[ok]', p)" \
  && unzip -nq -o $EV/mme_img/MME_Benchmark.zip -d $EV/mme_img \
  && echo "[ok] mme extracted"

echo "== scienceqa (HF) =="
mkdir -p $EV/scienceqa_img
$PY -c "
from huggingface_hub import hf_hub_download
for f in ['data.zip', 'pid_splits.json', 'problems.json']:
    try:
        print('[ok]', hf_hub_download('lmms-lab/ScienceQA', f,
              repo_type='dataset', local_dir='$EV/scienceqa_img'))
    except Exception as e:
        print('[FAIL]', f, e)"
[ -s $EV/scienceqa_img/data.zip ] && (cd $EV/scienceqa_img \
  && unzip -nq -o data.zip) && echo "[ok] scienceqa extracted"

echo "[done] $(date)"

echo "== test2015 =="
dl http://images.cocodataset.org/zips/test2015.zip $HOLD/test2015.zip \
  && unzip -nq -o $HOLD/test2015.zip -d $EV/vqav2 \
  && echo "[ok] test2015 extracted"

