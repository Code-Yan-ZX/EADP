#!/usr/bin/env python3
"""Archive existing research evidence without changing or rerunning inputs.

Build: python -B Qwen_vl/scripts/archive_research.py build
Verify: python -B Qwen_vl/scripts/archive_research.py verify
Only the standard library is used. SQLite databases are archived verbatim.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import tarfile
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "Qwen_vl/outputs"
DEFAULT_ARCHIVE = OUTPUTS / "research_archive/20261115"
EXCLUDED_ROOTS = {"discovery", "e0", "m12", "native_detail_sources", "research_archive"}
TEXT = {".json", ".jsonl", ".md", ".log", ".txt", ".csv", ".tsv", ".py", ".sh"}
DATABASE = {".sqlite", ".sqlite3", ".db"}
OPAQUE = {".xlsx", ".xls", ".pkl"}
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf"}
PART_BYTES = 32 * 1024 * 1024
CHUNK = 1024 * 1024


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def classify(path):
    suffix, name = path.suffix.lower(), path.name.lower()
    if name.endswith(("-wal", "-shm", ".lock")):
        return False, "transient_sqlite_or_lock"
    if suffix in TEXT:
        return True, "source_snapshot" if suffix in {".py", ".sh"} else "text_evidence"
    if suffix in DATABASE:
        return True, "sqlite_original"
    if suffix in OPAQUE:
        return True, "opaque_evaluator_result_never_executed"
    if suffix in IMAGES and (any(word in name for word in
            ("result", "overview", "pareto", "montage", "audit", "contact"))
            or any(part.lower() in {"report", "figures"} for part in path.parts)):
        return True, "result_figure"
    if suffix in {".pt", ".pth", ".bin", ".safetensors", ".npy", ".npz"}:
        return False, "weights_or_feature_cache"
    if suffix in IMAGES or suffix == ".img":
        return False, "source_image"
    if suffix in {".parquet", ".arrow", ".feather"}:
        return False, "source_dataset"
    if suffix == ".pyc":
        return False, "python_bytecode"
    return False, "other_binary_or_unsupported"


def sqlite_audit(path):
    wal = Path(str(path) + "-wal")
    wal_size = wal.stat().st_size if wal.exists() else 0
    if wal_size:
        raise RuntimeError(f"Nonempty SQLite WAL; refusing an incomplete main-file archive: {path}")
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    base = "file:" + quote(str(path.resolve())) + "?mode=ro"
    with sqlite3.connect(base) as source:
        schema = [dict(zip(("type", "name", "table", "sql"), row)) for row in source.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name")]
        tables = [row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        counts = {table: source.execute('SELECT count(*) FROM "' + table.replace('"', '""') + '"').fetchone()[0]
                  for table in tables}
    with sqlite3.connect(base + "&immutable=1") as source:
        check = [row[0] for row in source.execute("PRAGMA quick_check")]
        main_counts = {table: source.execute('SELECT count(*) FROM "' + table.replace('"', '""') + '"').fetchone()[0]
                       for table in tables}
    after = (path.stat().st_size, path.stat().st_mtime_ns)
    if before != after or check != ["ok"] or main_counts != counts or (wal.exists() and wal.stat().st_size):
        raise RuntimeError(f"SQLite changed or failed its read-only/main-file checks: {path}")
    return {"path": str(path.relative_to(ROOT)), "quick_check": "ok", "schema": schema,
            "table_counts": counts, "wal_bytes": wal_size,
            "readonly_equals_immutable_main_file_counts": True,
            "note": "Original database bytes included; WAL/SHM are not needed and are excluded. No checkpoint or write performed."}


class SplitWriter:
    def __init__(self, directory, family):
        self.directory, self.family = directory, family
        self.stream = None
        self.parts = []
        self.total_sha = hashlib.sha256()
        self.total_size = 0

    def write(self, data):
        original = len(data)
        self.total_sha.update(data)
        self.total_size += original
        while data:
            if self.stream is None:
                self.path = self.directory / f"{self.family}.tar.gz.part{len(self.parts):04d}"
                self.stream = self.path.open("xb")
                self.part_sha, self.part_size = hashlib.sha256(), 0
            count = min(len(data), PART_BYTES-self.part_size)
            block, data = data[:count], data[count:]
            self.stream.write(block)
            self.part_sha.update(block)
            self.part_size += count
            if self.part_size == PART_BYTES:
                self._close_part()
        return original

    def _close_part(self):
        if self.stream is not None:
            self.stream.close()
            self.parts.append({"path": str(self.path.relative_to(self.directory.parent)),
                               "size": self.part_size, "sha256": self.part_sha.hexdigest()})
            self.stream = None

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def close(self):
        self._close_part()


class HashReader:
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, count=-1):
        data = self.source.read(count)
        self.digest.update(data)
        self.size += len(data)
        return data


class PartsReader:
    """Concatenate byte ranges, not independently compressed gzip members."""
    def __init__(self, paths):
        self.paths = iter(paths)
        self.stream = None

    def read(self, count=-1):
        pieces, remaining = [], count
        while remaining != 0:
            if self.stream is None:
                try:
                    self.stream = next(self.paths).open("rb")
                except StopIteration:
                    break
            block = self.stream.read(remaining)
            if not block:
                self.stream.close()
                self.stream = None
                continue
            pieces.append(block)
            if remaining > 0:
                remaining -= len(block)
        return b"".join(pieces)

    def close(self):
        if self.stream is not None:
            self.stream.close()


def status_source(path):
    name = path.name.lower()
    eligible = name in {"status.json", "gpu.json", "gates.json", "train_complete.json", "comparison.json"}
    eligible = eligible or ("gate" in name and "failed" in name and path.suffix == ".json")
    if not eligible or path.stat().st_size > 8*1024*1024:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        return {"path": str(path.relative_to(ROOT)), "parseable_json": False}
    keys = ("status", "decision", "current_stage", "phase", "completed_utc", "started_utc", "expected_outputs", "outputs", "timings")
    return {"path": str(path.relative_to(ROOT)), "fields": {key: value[key] for key in keys if isinstance(value, dict) and key in value}}


def build(destination):
    if (destination/"index.json").exists():
        raise RuntimeError("Archive already exists; use verify instead of overwriting it")
    destination.mkdir(parents=True, exist_ok=True)
    parts_dir = destination/"parts"
    parts_dir.mkdir(exist_ok=True)
    families = {path.name: sorted(path.rglob("*")) for path in sorted(OUTPUTS.iterdir())
                if path.is_dir() and path.name not in EXCLUDED_ROOTS}
    root_files = [path for path in sorted(OUTPUTS.iterdir()) if path.is_file()]
    if root_files:
        families["_root_files"] = root_files
    index = {"schema": 1, "archive_date": "2026-11-15", "repository_root": ".",
        "format": "Deterministic PAX tar (regular members, uid/gid/mtime zero), gzip mtime=0, byte-range split; member bytes unchanged.",
        "part_limit_bytes": PART_BYTES, "excluded_roots": sorted(EXCLUDED_ROOTS),
        "scope": "Existing evidence only; no model rerun, changed conclusion, weights/features, source images or raw dataset shards.",
        "opaque_evidence": "Result XLSX/XLS and historical evaluator PKL are preserved as uninterpreted bytes, never loaded/executed.",
        "interpretation_source": "docs/project_handoff.md", "families": {}}
    for family, candidates in sorted(families.items()):
        selected, excluded, statuses = [], [], []
        for path in candidates:
            if not path.is_file():
                continue
            keep, kind = classify(path)
            if keep:
                selected.append((path, kind))
                state = status_source(path)
                if state is not None:
                    statuses.append(state)
            else:
                excluded.append({"path": str(path.relative_to(ROOT)), "size": path.stat().st_size, "type": kind})
        excluded_types = defaultdict(lambda: {"count": 0, "bytes": 0})
        for entry in excluded:
            excluded_types[entry["type"]]["count"] += 1
            excluded_types[entry["type"]]["bytes"] += entry["size"]
        databases = [sqlite_audit(path) for path, kind in selected if kind == "sqlite_original"]
        writer, members, reports = SplitWriter(parts_dir, family), [], []
        with gzip.GzipFile(filename="", fileobj=writer, mode="wb", compresslevel=6, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for path, kind in selected:
                    relative = str(path.relative_to(ROOT))
                    before = path.stat()
                    info = tarfile.TarInfo(relative)
                    info.size, info.mode, info.mtime = before.st_size, 0o644, 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as source:
                        reader = HashReader(source)
                        archive.addfile(info, reader)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or reader.size != info.size:
                        raise RuntimeError(f"Input changed during archive: {path}")
                    member = {"path": relative, "size": info.size, "sha256": reader.digest.hexdigest(), "type": kind}
                    members.append(member)
                    if path.suffix.lower() == ".md":
                        copy_path = destination/"reports"/relative
                        copy_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, copy_path)
                        if sha(copy_path) != member["sha256"]:
                            raise RuntimeError("Browsable report copy changed")
                        reports.append({"path": str(copy_path.relative_to(destination)), "sha256": member["sha256"]})
        writer.close()
        result = {"members": members, "member_count": len(members), "source_bytes": sum(m["size"] for m in members),
            "gzip_bytes": writer.total_size, "gzip_sha256": writer.total_sha.hexdigest(), "parts": writer.parts,
            "excluded_types": dict(excluded_types), "excluded_files": excluded,
            "sqlite": databases, "status_sources": statuses, "browsable_reports": reports,
            "historical_boundaries": [m["path"] for m in members if any(
                marker in m["path"].lower() for marker in ("failed", "archive_geometry", "probe", "preflight", "pre_pandas", "panel96"))]}
        index["families"][family] = result
        print(json.dumps({"family": family, "files": len(members), "source_mib": round(result["source_bytes"]/2**20, 3),
                          "gzip_mib": round(writer.total_size/2**20, 3), "parts": len(writer.parts)}, ensure_ascii=False), flush=True)
    index["totals"] = {"families": len(index["families"]),
        "members": sum(f["member_count"] for f in index["families"].values()),
        "source_bytes": sum(f["source_bytes"] for f in index["families"].values()),
        "gzip_bytes": sum(f["gzip_bytes"] for f in index["families"].values()),
        "parts": sum(len(f["parts"]) for f in index["families"].values()),
        "excluded_files": sum(len(f["excluded_files"]) for f in index["families"].values()),
        "excluded_bytes": sum(sum(v["bytes"] for v in f["excluded_types"].values()) for f in index["families"].values())}
    write_json(destination/"index.json", index)
    readme(destination, index)
    return index


def verify(destination):
    index = json.loads((destination/"index.json").read_text(encoding="utf-8"))
    verified_members = 0
    for family, description in sorted(index["families"].items()):
        combined = hashlib.sha256()
        for part in description["parts"]:
            path = destination/part["path"]
            if not path.is_file() or path.stat().st_size != part["size"] or part["size"] > index["part_limit_bytes"]:
                raise RuntimeError(f"Part size/missing error: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(CHUNK), b""):
                    digest.update(block)
                    combined.update(block)
            if digest.hexdigest() != part["sha256"]:
                raise RuntimeError(f"Part hash changed: {path}")
        if combined.hexdigest() != description["gzip_sha256"]:
            raise RuntimeError(f"Concatenated gzip hash changed: {family}")
        expected = {member["path"]: member for member in description["members"]}
        if len(expected) != description["member_count"]:
            raise RuntimeError("Duplicate member identity")
        seen = set()
        source = PartsReader([destination/part["path"] for part in description["parts"]])
        try:
            with gzip.GzipFile(fileobj=source, mode="rb") as compressed:
                with tarfile.open(fileobj=compressed, mode="r|") as archive:
                    for member in archive:
                        name = member.name
                        if name not in expected or name in seen or not member.isfile():
                            raise RuntimeError(f"Unexpected/duplicate/nonregular member: {name}")
                        if Path(name).is_absolute() or ".." in Path(name).parts:
                            raise RuntimeError("Unsafe archive member path")
                        digest, size = hashlib.sha256(), 0
                        content = archive.extractfile(member)
                        for block in iter(lambda: content.read(CHUNK), b""):
                            digest.update(block)
                            size += len(block)
                        if size != expected[name]["size"] or digest.hexdigest() != expected[name]["sha256"]:
                            raise RuntimeError(f"Member bytes changed: {name}")
                        seen.add(name)
                # Read through the gzip trailer/CRC without extracting files.
                while compressed.read(CHUNK):
                    pass
        finally:
            source.close()
        if seen != set(expected):
            raise RuntimeError(f"Missing archive members: {family}")
        for report in description["browsable_reports"]:
            if sha(destination/report["path"]) != report["sha256"]:
                raise RuntimeError(f"Browsable report changed: {report['path']}")
        verified_members += len(seen)
        print(f"verified {family}: {len(seen)} original members", flush=True)
    if verified_members != index["totals"]["members"]:
        raise RuntimeError("Archive member total differs")
    result = {"status": "PASS", "index_sha256": sha(destination/"index.json"),
              "families": len(index["families"]), "members": verified_members,
              "parts": index["totals"]["parts"], "gzip_bytes": index["totals"]["gzip_bytes"],
              "verification": "Every part and concatenated gzip SHA256, gzip CRC/trailer, every tar member SHA256/size, and browsable report copy checked by streaming; no extraction or source modification."}
    write_json(destination/"verification.json", result)
    return result


def readme(destination, index):
    reports = [entry for family in index["families"].values() for entry in family["browsable_reports"]]
    lines = ["# Research evidence archive — 2026-11-15", "",
        "This preserves existing experiment evidence. It does not rerun experiments, change their conclusions, establish a new method, or make oracle scores deployable results.", "",
        "Code and project reports remain at their normal repository paths. The archives preserve original JSON/JSONL, reports, logs, CSV/TSV, SQLite, historical source snapshots, selected result figures, and opaque result XLSX/PKL bytes. Pickles are never loaded or executed.", "",
        "Weights, feature caches, source images, raw dataset shards, bytecode and transient SQLite WAL/SHM/locks are excluded. index.json records every included original relative path, size and SHA256, all excluded paths/types/sizes, original database schemas/counts/read-only checks, recorded statuses, and every compressed part hash. This is evidence preservation, not a self-contained model/data backup.", "",
        "discovery, e0 and m12 are already tracked historical output families; native_detail_sources contains external raw data; research_archive is never recursively archived. These roots are intentionally excluded.", "",
        "## Latest browsable reports", ""]
    priority = ("detail_allocation", "student_broad", "student_transfer", "paired_pruning", "student_full_aux")
    for label in priority:
        for entry in reports:
            if f"/{label}/" in entry["path"] and entry["path"].endswith("report.md"):
                lines.append(f"- [{label}]({entry['path']})")
    lines += ["", "Other preserved Markdown:", ""]
    for entry in reports:
        lines.append(f"- [{entry['path'].removeprefix('reports/Qwen_vl/outputs/')}]({entry['path']})")
    lines += ["", "## Verification and recovery", "",
        "Run from the repository root using Python 3 (standard library only):", "", "```bash",
        "python3 -B Qwen_vl/scripts/archive_research.py verify", "```", "",
        "verification.json records the completed full streaming verification. Parts are 32 MiB maximum byte ranges of ONE gzip stream per family; an individual part is not a standalone gzip archive.", "",
        "To recover a family into a new empty directory, concatenate its parts in the exact index.json order, then unpack. For example:", "", "```bash",
        "mkdir -p /tmp/eadp-evidence-restore",
        "cat Qwen_vl/outputs/research_archive/20261115/parts/detail_allocation.tar.gz.part* | tar -xz -C /tmp/eadp-evidence-restore",
        "```", "",
        "Archive members use original repository-relative paths. Recover into a separate directory; do not overwrite live experiment inputs/results. Verify first. Original database bytes are preserved exactly; all five database WALs were empty, quick_check passed, and immutable main-file counts matched ordinary read-only counts. Recovered JSON paths still refer to original external datasets/models and need those dependencies for fresh computation.", "",
        "## Completed, failed, and superseded evidence", "",
        "Read the original reports and docs/project_handoff.md for scientific interpretation. The index status_sources retain the recorded state locations; historical_boundaries identify failed attempts, superseded geometry, probes and preflight versions. Presence in this archive does not make a failed gate, old geometry or preflight panel an accepted experiment.", "",
        "In particular: student_transfer/run_20261112 and student_broad/run_20261113 precede their completed v2 runs; their failed gates/snapshots remain historical evidence. binding_oracle/archive_geometry_v1 predates corrected geometry. local_reward_diagnostic/serial_throughput_probe was discarded. detail_allocation/manifest_panel96_preflight.json is the earlier panel; the final 192-per-dataset run concluded STOP_THIS_GEOMETRY. CF carrier failures, the old student's full-DocVQA failure, and negative paired-pruning targets remain negative results. No teacher/oracle result is relabeled as learned or deployable performance.", "",
        "| Family | Files | Original MiB | gzip MiB | Parts |", "|---|---:|---:|---:|---:|"]
    for name, family in sorted(index["families"].items()):
        lines.append(f"| {name} | {family['member_count']} | {family['source_bytes']/2**20:.3f} | {family['gzip_bytes']/2**20:.3f} | {len(family['parts'])} |")
    (destination/"README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    destination = args.archive_dir.resolve()
    if args.mode == "build":
        result = build(destination)
        print(json.dumps(result["totals"], ensure_ascii=False), flush=True)
    else:
        print(json.dumps(verify(destination), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
