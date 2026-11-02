"""Validate local TextVQA images against the author's official archive.

Reads ZIP central-directory/range bytes, never downloads the entire 7 GB ZIP.
CRC/size checks cover all used images; extracted panel members get SHA checks.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import time
import urllib.request
import zipfile
import zlib

URL = "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
ANNOTATION = "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json"


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position, self.cache, self.requests = url, 0, {}, []
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
            self.size = int(r.headers["Content-Length"])
            self.headers = dict(r.headers)

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.position
        if n > 16 * 1024 * 1024:
            raise ValueError("Refuse a range exceeding 16 MB")
        result = bytearray()
        while n and self.position < self.size:
            chunk = self.position // (1024 * 1024)
            begin = chunk * 1024 * 1024
            if chunk not in self.cache:
                end = min(begin + 1024 * 1024, self.size) - 1
                request = urllib.request.Request(self.url, headers={"Range": f"bytes={begin}-{end}"})
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(request, timeout=60) as response:
                            expected = f"bytes {begin}-{end}/{self.size}"
                            if response.status != 206 or response.headers.get("Content-Range") != expected:
                                raise ValueError(f"Server did not honor exact range: {response.status} / {response.headers.get('Content-Range')}")
                            payload = response.read(end - begin + 2)
                        break
                    except OSError:
                        if attempt == 2:
                            raise
                        time.sleep(1)
                if len(payload) != end - begin + 1:
                    raise ValueError("Range length mismatch")
                self.cache[chunk] = payload
                self.requests.append({"begin": begin, "end": end, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
            payload = self.cache[chunk]
            start = self.position - begin
            take = min(n, len(payload) - start)
            result.extend(payload[start:start + take])
            self.position += take
            n -= take
        return bytes(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-root", type=Path, default=Path("/media/disk2/YZX/research/EADP_amp"))
    p.add_argument("--panel-images", type=int, default=0, choices=(0, 16, 64))
    args = p.parse_args()
    out = Path(__file__).resolve().parent
    data = args.old_root / "LLaVA/playground/data/eval/textvqa"
    target = out / "TextVQA_0.5.1_val.fresh.json"
    if not target.exists():
        with urllib.request.urlopen(ANNOTATION, timeout=60) as response:
            payload = response.read(10 * 1024 * 1024)
        json.loads(payload)
        target.write_bytes(payload)
    annotation_equal = target.read_bytes() == (data / "TextVQA_0.5.1_val.json").read_bytes()
    questions = [json.loads(line) for line in (data / "llava_textvqa_val_v051_ocr.jsonl").read_text().splitlines() if line.strip()]
    images = sorted({q["image"] for q in questions})
    remote = RangeReader(URL)
    rows, sampled = [], []
    with zipfile.ZipFile(remote) as archive:
        by_name = {Path(item.filename).name: item for item in archive.infolist() if not item.is_dir()}
        for name in images:
            raw = (data / "train_images" / name).read_bytes()
            member = by_name[name]
            rows.append({"image": name, "zip_member": member.filename, "local_size": len(raw), "zip_size": member.file_size, "local_crc32": zlib.crc32(raw) & 0xffffffff, "zip_crc32": member.CRC, "matches": len(raw) == member.file_size and (zlib.crc32(raw) & 0xffffffff) == member.CRC})
        (out / 'official_image_crc_identity.json').write_text(json.dumps({'url': URL, 'archive_headers': remote.headers, 'rows': rows, 'mismatches': sum(not r['matches'] for r in rows), 'annotation_equal': annotation_equal}, indent=2) + '\n')
        panel = [questions[int((i + .5) * len(questions) / 64)] for i in range(64)]
        image_out = out / "fresh_panel_images"
        image_out.mkdir(exist_ok=True)
        for q in panel[:args.panel_images]:
            name = q["image"]
            raw = archive.read(by_name[name])
            sha = hashlib.sha256(raw).hexdigest()
            local_sha = hashlib.sha256((data / "train_images" / name).read_bytes()).hexdigest()
            (image_out / name).write_bytes(raw)
            sampled.append({"image": name, "fresh_sha256": sha, "old_sha256": local_sha, "matches": sha == local_sha})
    report = {"annotation": {"url": ANNOTATION, "fresh_path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "byte_identical_to_local": annotation_equal}, "images_url": URL, "archive_size": remote.size, "archive_headers": remote.headers, "used_images": len(rows), "crc_size_mismatches": [r for r in rows if not r["matches"]], "image_crc_size_checks": rows, "fresh_panel_image_sha_checks": sampled, "range_requests": remote.requests, "downloaded_bytes": sum(r["bytes"] for r in remote.requests), "scope": "CRC32+size identity against fresh official ZIP central directory for all used images; cryptographic payload identity checked only for downloaded panel members. No GPU."}
    destination = out / "official_dataset_identity.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(destination), "images": len(rows), "crc_size_mismatches": len(report["crc_size_mismatches"]), "fresh_sha_samples": len(sampled), "sample_sha_mismatches": sum(not r["matches"] for r in sampled), "annotation_equal": annotation_equal, "downloaded_bytes": report["downloaded_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
