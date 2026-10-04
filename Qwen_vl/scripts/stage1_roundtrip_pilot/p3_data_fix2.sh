#!/usr/bin/env bash
# P3 data fix round 2 (2026-10-04 night): repair failed/corrupt downloads.
#  - val2014.zip: earlier copy corrupted (12.6GB vs official ~6.2GB) -> fresh
#  - GQA images1207.zip: 0-byte .part -> fresh
#  - test2015.zip: already downloading (separate wget, untouched)
#  - MME images: HF lmms-lab/MME (MME_Benchmark.zip)
#  - ScienceQA images: HF lmms-lab/ScienceQA (data zip)
# All big files under /media/disk2/tmp_hold or the eval tree (disk2).
set -u
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
HOLD=/media/disk2/tmp_hold
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
mkdir -p $HOLD

dl () { [ -s "$2" ] && { echo "[skip] $2"; return 0; }
  wget -c -q -O "$2.part" "$1" && mv "$2.part" "$2" && echo "[ok] $2" \
    || echo "[FAIL] $1"; }

echo "== val2014 =="
rm -f $HOLD/val2014.zip $HOLD/val2014.zip.part
dl http://images.cocodataset.org/zips/val2014.zip $HOLD/val2014.zip
unzip -nq -o $HOLD/val2014.zip -d $EV/pope/data \
  && echo "[ok] val2014 extracted to pope/data"

echo "== gqa images =="
dl https://downloads.cs.stanford.edu/nlp/data/gqa/images1207.zip \
   $EV/gqa/images1207.zip
[ -s $EV/gqa/images1207.zip ] && unzip -nq -o $EV/gqa/images1207.zip \
  -d $EV/gqa/data && echo "[ok] gqa images extracted"

echo "== mme images (HF lmms-lab/MME) =="
mkdir -p $EV/mme_img
$PY -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('lmms-lab/MME', 'MME_Benchmark.zip', repo_type='dataset',
                    local_dir='$EV/mme_img')
print('[ok]', p)
"
[ -s $EV/mme_img/MME_Benchmark.zip ] && unzip -nq -o \
  $EV/mme_img/MME_Benchmark.zip -d $EV/mme_img && echo "[ok] mme extracted"

echo "== scienceqa (HF lmms-lab/ScienceQA) =="
mkdir -p $EV/scienceqa_img
$PY -c "
from huggingface_hub import hf_hub_download
for f in ['data.zip', 'pid_splits.json', 'problems.json']:
    try:
        p = hf_hub_download('lmms-lab/ScienceQA', f, repo_type='dataset',
                            local_dir='$EV/scienceqa_img')
        print('[ok]', p)
    except Exception as e:
        print('[FAIL]', f, e)
"
[ -s $EV/scienceqa_img/data.zip ] && (cd $EV/scienceqa_img \
  && unzip -nq -o data.zip) && echo "[ok] scienceqa extracted"

echo "[done] $(date)"
