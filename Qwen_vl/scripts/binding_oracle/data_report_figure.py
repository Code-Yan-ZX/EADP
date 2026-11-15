"""Data-only table audit figure; no predictions or GPU imports."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import data_build as db


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(db.OUT/'combined_manifest.json'))
    p.add_argument('--pair-id',default='swap_38577_38580');p.add_argument('--out',default=str(db.OUT/'table_swap_audit.png'))
    args=p.parse_args();records=json.loads(Path(args.manifest).read_text())['records'];r=next(x for x in records if x['pair_id']==args.pair_id)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    fonts={size:ImageFont.truetype(font_path,size) for size in [19,21,23,25,30]}
    canvas=Image.new('RGB',(1500,1460),'white');draw=ImageDraw.Draw(canvas)
    draw.text((35,22),'Same amounts, different year binding',font=fonts[30],fill='#16263b')
    draw.text((35,67),'Data audit only: original DocVQA TRAIN annotations; targets shown below are not model predictions.',font=fonts[19],fill='#384858')
    for i,f in enumerate(r['fields']):
        draw.text((35,107+37*i),f'Q{i}: {f["question"]}',font=fonts[23],fill=['#1261a0','#b34620'][i])
    crop=[350,635,1300,825]
    colors=['#1261a0','#b34620']
    names=[('source_image','Untouched real page'),('original_image','Member A: both original amounts rerendered'),('swapped_image','Member B: amounts swapped; year labels unchanged')]
    for panel,(member,title) in enumerate(names):
        top=197+panel*330
        draw.text((35,top),title,font=fonts[25],fill='#16263b')
        im=Image.open(r[member]).convert('RGB').crop(crop)
        painter=ImageDraw.Draw(im)
        for i,f in enumerate(r['fields']):
            x,y,w,h=f['value_bbox'];painter.rectangle((x-crop[0]-3,y-crop[1]-3,x+w-crop[0]+3,y+h-crop[1]+3),outline=colors[i],width=2)
        im=im.resize((1425,285),Image.Resampling.LANCZOS)
        # Retain the exact source crop without stretching; crop aspect is 5:1.
        canvas.paste(im,(35,top+39))
    top=1220
    draw.line((35,top,1460,top),fill='#b8c2cc',width=2)
    draw.text((35,top+12),'Annotated targets',font=fonts[23],fill='#16263b')
    draw.text((410,top+12),'Member A',font=fonts[23],fill='#16263b')
    draw.text((860,top+12),'Member B',font=fonts[23],fill='#16263b')
    for i,f in enumerate(r['fields']):
        yy=top+50+38*i
        draw.text((35,yy),f'Q{i} (year {1998+i})',font=fonts[23],fill=colors[i])
        draw.text((410,yy),f['original_answer'],font=fonts[23],fill='#16263b')
        draw.text((860,yy),f['swapped_answer'],font=fonts[23],fill='#16263b')
    draw.text((35,1410),f'{r["pair_id"]}; source {r["image_id"]}. Same numeric multiset; symmetric font/patch procedure.',font=fonts[19],fill='#384858')
    canvas.save(args.out)
    db.atomic_json(Path(args.out).with_suffix('.json'),dict(pair_id=r['pair_id'],source_document_id=r['source_document_id'],
        source_image=r['source_image'],source_sha256=r['source_sha256'],crop_xyxy=crop,
        selection='Selected by true year/amount table type and readable source layout before inspecting any model predictions.',
        four_targets=[dict(member=member,question=f['question'],answer=f['original_answer' if member=='original' else 'swapped_answer']) for member in ['original','swapped'] for f in r['fields']],
        source_qa_retained=True,predictions_used=False))
    print(args.out)

if __name__=='__main__':
    main()
