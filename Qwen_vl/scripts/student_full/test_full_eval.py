"""CPU-only tests for freeze manifest, verified image materialization and cache."""
import base64
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from PIL import Image
import torch
import full_eval as f

class StubEngine:
    def __init__(self): self.calls=0
    def encode(self,p):
        self.calls+=1
        n=p['n_vis'];v=torch.full((n,2),float(self.calls))
        return v,[v.clone() for _ in range(3)]

class HarnessTests(unittest.TestCase):
    def test_page_cache_checks_actual_pixels_and_grid(self):
        eng=StubEngine(); cache=f.PageCache();cache.clear_to('page-a')
        prep={'has_image':True,'pv':torch.tensor([1.,2.]),'gthw':torch.tensor([[1,64,64]]),'n_vis':1024}
        a,ds,hit=cache.get(eng,prep,1024); self.assertFalse(hit)
        b,ds,hit=cache.get(eng,dict(prep),1024);self.assertTrue(hit);self.assertIs(a,b);self.assertEqual(eng.calls,1)
        with self.assertRaises(ValueError): cache.get(eng,{**prep,'pv':torch.tensor([1.,3.])},1024)
        with self.assertRaises(ValueError): cache.get(eng,{**prep,'gthw':torch.tensor([[1,32,32]])},1024)
        cache.clear_to('page-b');cache.get(eng,prep,1024);self.assertEqual(eng.calls,2)

    def test_materialize_verifies_source_and_matches_official_pil(self):
        with tempfile.TemporaryDirectory() as d:
            old=f.OUT;f.OUT=Path(d)
            try:
                buf=io.BytesIO();Image.new('RGB',(13,9),(127,85,41)).save(buf,format='JPEG',quality=93)
                raw=buf.getvalue(); item={'image_sha256':hashlib.sha256(raw).hexdigest(),'image_path':'page.jpg'}
                row={'image':base64.b64encode(raw).decode()}
                path,digest=f.materialize_image(row,item)
                reference=Path(d)/'expected.jpg';Image.open(io.BytesIO(raw)).save(reference)
                self.assertEqual(path.read_bytes(),reference.read_bytes());self.assertEqual(digest,f.sha256(path))
                self.assertEqual(f.materialize_image(row,item),(path,digest))
                with self.assertRaises(ValueError): f.materialize_image(row,{**item,'image_sha256':'0'*64})
                path.write_bytes(b'corrupt')
                with self.assertRaises(ValueError): f.materialize_image(row,item)
            finally: f.OUT=old

    def test_db_persists_errors_and_resumes_by_frozen_identity(self):
        with tempfile.TemporaryDirectory() as d:
            old=(f.OUT,f.DB,f.FREEZE);f.OUT=Path(d);f.DB=f.OUT/'run.sqlite3';f.FREEZE=f.OUT/'freeze.json';f.FREEZE.write_text('{}')
            try:
                db=f.connect_db();db.execute('INSERT INTO results VALUES(?,?,?,?,?,?,?,?,?,?,?)',(0,'b2','error',None,None,None,None,None,None,None,'failed'));db.commit();db.close()
                db=f.connect_db();self.assertEqual(db.execute('SELECT status,error FROM results').fetchone(),('error','failed'))
                db.execute('INSERT OR REPLACE INTO results VALUES(?,?,?,?,?,?,?,?,?,?,?)',(0,'b2','success','text','[1]',0,256,'a','b','{}',None));db.commit();db.close()
                db=f.connect_db();self.assertEqual(db.execute('SELECT status FROM results').fetchone(),('success',));db.close()
                f.FREEZE.write_text('{"changed":1}')
                with self.assertRaises(ValueError): f.connect_db()
            finally: f.OUT,f.DB,f.FREEZE=old

if __name__=='__main__': unittest.main()
