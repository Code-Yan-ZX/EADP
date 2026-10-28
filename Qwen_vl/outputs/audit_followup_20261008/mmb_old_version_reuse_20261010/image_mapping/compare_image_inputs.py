"""CPU-only exact old/V11 MMB image-input compatibility audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import ast, base64, collections, csv, datetime, hashlib, io, json, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
csv.field_size_limit(2**31 - 1)
OUT = Path(__file__).resolve().parent
LIVE = Path('/media/disk2/YZX/research/EADP_amp')
DATA = LIVE / 'LLaVA/playground/data/eval/mmbench'
OLD = DATA / 'mmbench_dev_20230712.tsv'
NEW = DATA / 'MMBench_DEV_EN_V11.tsv'
UTIL = LIVE / 'LLaVA/llava/mm_utils.py'
MODEL_CONFIG = Path('/media/disk2/YZX/doct/FastV/llava-v1.5-7b/config.json')
PROCESSOR_CONFIG = Path('/home/dell/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/preprocessor_config.json')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(array): return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def save(name, obj): (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
def load(path):
    with path.open(newline='') as f: return list(csv.DictReader(f, delimiter='\t'))
def prompt(row):
    options = []
    for key in 'ABCD':
        value = row[key]
        if not value or value.lower() in ('nan', 'none'): break
        options.append(value)
    text = row['question']
    hint = row['hint']
    if hint and hint.lower() not in ('nan', 'none'): text = hint + '\n' + text
    for letter, value in zip('ABCD', options): text += '\n' + letter + '. ' + value
    return text

config = json.loads(MODEL_CONFIG.read_text())
assert config['image_aspect_ratio'] == 'pad'
tree = ast.parse(UTIL.read_text())
helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('expand2square', 'process_images')]
assert {n.name for n in helpers} == {'expand2square', 'process_images'}
ns = {'Image': Image, 'torch': torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=helpers, type_ignores=[])), str(UTIL)+'[exact CPU pad helpers]', 'exec'), ns)
processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336', local_files_only=True)
old_rows, new_rows = load(OLD), load(NEW)
new_by_prompt = collections.defaultdict(list)
new_embedded = {}
for row in new_rows:
    new_by_prompt[prompt(row)].append(row)
    if len(row['image']) > 100: new_embedded[int(row['index']) % 1000000] = row['image']

cache = {}
def image_input(encoded):
    payload = base64.b64decode(encoded)
    payload_sha = hashlib.sha256(payload).hexdigest()
    if payload_sha in cache: return cache[payload_sha]
    image = Image.open(io.BytesIO(payload))
    image.load()
    rgb = np.asarray(image.convert('RGB'))
    final = ns['process_images']([image], processor, SimpleNamespace(**config))[0].unsqueeze(0)
    assert final.device.type == 'cpu' and list(final.shape) == [1, 3, 336, 336]
    final_half = final.half()
    # Same preprocessing before affine rescale/normalization, retained for image diagnostics.
    padded = ns['expand2square'](image, tuple(int(x*255) for x in processor.image_mean))
    pixels = processor.preprocess(padded, return_tensors='np', do_rescale=False, do_normalize=False)['pixel_values'][0]
    assert list(pixels.shape) == [3, 336, 336]
    small = Image.fromarray(pixels.transpose(1, 2, 0)).convert('L').resize((9, 8), Image.Resampling.LANCZOS)
    grey = np.asarray(small)
    dhash_bits = (grey[:, 1:] > grey[:, :-1]).reshape(-1)
    record = dict(payload_sha256=payload_sha, image_size=list(image.size), image_mode=image.mode,
                  decoded_rgb_shape=list(rgb.shape), decoded_rgb_sha256=digest(rgb),
                  processed_float32_sha256=digest(final.numpy()), processed_float16_sha256=digest(final_half.numpy()),
                  processed_rgb8_sha256=digest(pixels), pixels=pixels, dhash_bits=dhash_bits)
    cache[payload_sha] = record
    return record

def priority(row):
    candidates = new_by_prompt.get(prompt(row), [])
    same_ID = any(int(r['index']) == int(row['index']) for r in candidates)
    return (0 if candidates and len(candidates)==1 and not same_ID else 1 if candidates and not same_ID else 2, int(row['index']))

pairs, row_reports = [], []
processing_rows = sorted(old_rows, key=priority)
for position, old_row in enumerate(processing_rows):
    old_index = int(old_row['index'])
    candidates = new_by_prompt.get(prompt(old_row), [])
    pair_records = []
    if candidates:
        old_image = image_input(old_row['image'])
        for new_row in candidates:
            new_index = int(new_row['index'])
            encoded = new_row['image'] if len(new_row['image']) > 100 else new_embedded[new_index % 1000000]
            new_image = image_input(encoded)
            delta = old_image['pixels'].astype(np.float32) - new_image['pixels'].astype(np.float32)
            mse = float(np.mean(delta.astype(np.float64)**2))
            mae = float(np.mean(np.abs(delta)))
            rmse = math.sqrt(mse)
            psnr = None if mse == 0 else 10*math.log10(255**2/mse)
            dhash_distance = int(np.count_nonzero(old_image['dhash_bits'] != new_image['dhash_bits']))
            sizes_equal = old_image['image_size'] == new_image['image_size']
            payload_equal = old_image['payload_sha256'] == new_image['payload_sha256']
            decoded_equal = old_image['decoded_rgb_shape'] == new_image['decoded_rgb_shape'] and old_image['decoded_rgb_sha256'] == new_image['decoded_rgb_sha256']
            fp32_equal = old_image['processed_float32_sha256'] == new_image['processed_float32_sha256']
            fp16_equal = old_image['processed_float16_sha256'] == new_image['processed_float16_sha256']
            near = rmse <= 3.0 and dhash_distance <= 2
            pair = dict(old_index=old_index, old_base_id=old_index%1000000, v11_index=new_index,
                        v11_base_id=new_index%1000000, same_index=old_index==new_index,
                        same_base_id=old_index%1000000==new_index%1000000, native_prompt_exact=True,
                        old_image_size=old_image['image_size'], v11_image_size=new_image['image_size'],
                        image_sizes_exact=sizes_equal, jpeg_payload_exact=payload_equal, decoded_RGB_exact=decoded_equal,
                        processed_float32_exact=fp32_equal, processed_float16_exact=fp16_equal,
                        processed_RGB8_exact=old_image['processed_rgb8_sha256']==new_image['processed_rgb8_sha256'],
                        strict_model_image_input_reusable=fp16_equal and sizes_equal,
                        near_perceptual_diagnostic_only=near, processed_rgb8_MAE=mae, processed_rgb8_RMSE=rmse,
                        processed_rgb8_PSNR_dB=psnr, dHash_hamming_distance_64=dhash_distance,
                        old_image_payload_sha256=old_image['payload_sha256'], v11_image_payload_sha256=new_image['payload_sha256'],
                        old_decoded_RGB_sha256=old_image['decoded_rgb_sha256'], v11_decoded_RGB_sha256=new_image['decoded_rgb_sha256'],
                        old_processed_float32_sha256=old_image['processed_float32_sha256'], v11_processed_float32_sha256=new_image['processed_float32_sha256'],
                        old_processed_float16_sha256=old_image['processed_float16_sha256'], v11_processed_float16_sha256=new_image['processed_float16_sha256'])
            pairs.append(pair); pair_records.append(pair)
    eligible = [p['v11_index'] for p in pair_records if p['strict_model_image_input_reusable']]
    preferred = old_index if old_index in eligible else eligible[0] if len(eligible)==1 else None
    row_reports.append(dict(old_index=old_index, old_base_id=old_index%1000000, native_prompt_sha256=hashlib.sha256(prompt(old_row).encode()).hexdigest(),
                            candidate_v11_indices=[p['v11_index'] for p in pair_records],
                            strict_image_eligible_v11_indices=eligible, image_verified_selected_v11_index_or_null=preferred,
                            same_index_image_eligible=old_index in eligible, cross_id_unique_image_eligible=len(eligible)==1 and eligible[0]!=old_index,
                            ambiguous_after_image=len(eligible)>1 and old_index not in eligible,
                            strict_any_image_eligible=bool(eligible), exact_decoded_RGB_candidate_v11_indices=[p['v11_index'] for p in pair_records if p['decoded_RGB_exact'] and p['image_sizes_exact']],
                            near_perceptual_candidate_v11_indices=[p['v11_index'] for p in pair_records if p['near_perceptual_diagnostic_only'] and p['image_sizes_exact']],
                            note='GT not used in mapping. Near-perceptual is diagnostic only and never authorizes exact answer reuse.'))
    if position%400==0: print(json.dumps({'processed_old_rows':position+1,'candidate_pairs':len(pairs),'cached_images':len(cache)}), flush=True)
    if position==76:
        save('cross_ID_unique_priority_pairs.json', pairs)
        print(json.dumps({'cross_ID_unique_priority_done':True,'n_pairs':len(pairs),'strict_image_eligible':sum(p['strict_model_image_input_reusable'] for p in pairs),'decoded_RGB_exact':sum(p['decoded_RGB_exact'] for p in pairs),'near_perceptual':sum(p['near_perceptual_diagnostic_only'] for p in pairs)}), flush=True)

row_reports.sort(key=lambda row:next(i for i,r in enumerate(old_rows) if int(r['index'])==row['old_index']))

group_rows = collections.defaultdict(list)
for row in row_reports: group_rows[row['old_base_id']].append(row)
group_reports = []
for base, rows in sorted(group_rows.items()):
    group_reports.append(dict(old_base_id=base, n_old_rotation_rows=len(rows), old_indices=[r['old_index'] for r in rows],
                              all_rows_have_exact_prompt_candidate=all(r['candidate_v11_indices'] for r in rows),
                              all_rows_have_strict_image_eligible_candidate=all(r['strict_any_image_eligible'] for r in rows),
                              all_rows_image_verified_unambiguous=all(r['image_verified_selected_v11_index_or_null'] is not None for r in rows),
                              all_rows_same_index_image_eligible=all(r['same_index_image_eligible'] for r in rows),
                              image_verified_selected_v11_indices=[r['image_verified_selected_v11_index_or_null'] for r in rows]))

def stats(sub):
    return dict(n_candidate_pairs=len(sub), image_sizes_exact=sum(r['image_sizes_exact'] for r in sub),
                jpeg_payload_exact=sum(r['jpeg_payload_exact'] for r in sub), decoded_RGB_exact=sum(r['decoded_RGB_exact'] for r in sub),
                processed_float32_exact=sum(r['processed_float32_exact'] for r in sub), processed_float16_exact=sum(r['processed_float16_exact'] for r in sub),
                processed_RGB8_exact=sum(r['processed_RGB8_exact'] for r in sub), strict_image_input_reusable=sum(r['strict_model_image_input_reusable'] for r in sub),
                near_perceptual_diagnostic=sum(r['near_perceptual_diagnostic_only'] for r in sub))
source_paths=[OLD, NEW, UTIL, MODEL_CONFIG, PROCESSOR_CONFIG, Path(__file__)]
summary = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), CPU_only=True,
               CUDA_initialized=torch.cuda.is_initialized(), torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
               old_rows=len(old_rows), old_groups=len(group_rows), v11_rows=len(new_rows), cached_distinct_image_payloads=len(cache),
               processor=processor.to_dict(), model_config=dict(image_aspect_ratio=config['image_aspect_ratio'], vision_tower=config['mm_vision_tower']),
               exact_rule='Native cur_prompt identical; CPU process_images pad/CLIP336 FP16 tensors byte-identical; original image_sizes equal. GT excluded.',
               near_rule='Final processed RGB8 RMSE<=3 and 64-bit dHash distance<=2. Diagnostic only, never exact reuse.',
               pair_statistics={'all':stats(pairs),'same_index':stats([p for p in pairs if p['same_index']]),'cross_index':stats([p for p in pairs if not p['same_index']])},
               row_statistics=dict(exact_prompt_candidates=sum(bool(r['candidate_v11_indices']) for r in row_reports),
                                   strict_image_any_candidate=sum(r['strict_any_image_eligible'] for r in row_reports),
                                   strict_image_unambiguous_or_same_ID=sum(r['image_verified_selected_v11_index_or_null'] is not None for r in row_reports),
                                   strict_image_same_ID=sum(r['same_index_image_eligible'] for r in row_reports),
                                   strict_image_cross_ID_unique=sum(r['cross_id_unique_image_eligible'] for r in row_reports),
                                   ambiguous_after_image=sum(r['ambiguous_after_image'] for r in row_reports)),
               group_statistics={key:sum(r[key] for r in group_reports) for key in ['all_rows_have_exact_prompt_candidate','all_rows_have_strict_image_eligible_candidate','all_rows_image_verified_unambiguous','all_rows_same_index_image_eligible']},
               source_sha256={str(p):sha(p) for p in source_paths},
               caveats=['Image equality alone does not confirm method/generation/checkpoint/execution recipe identity.',
                        'JPEG file equality is not required; actual model FP16 input equality plus image_sizes is required.',
                        'Near-perceptual images can support approximate sensitivity analysis but do not produce exact old-version reproduced scores.'])
assert summary['CUDA_initialized'] is False
save('image_pairs.json', pairs)
save('old_rows_image_mapping.json', row_reports)
save('old_groups_image_mapping.json', group_reports)
with (OUT/'image_pairs.csv').open('w', newline='') as f:
    writer=csv.DictWriter(f, fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
save('summary.json', summary)
print(json.dumps({'finished':True,'summary':summary}, ensure_ascii=False), flush=True)
