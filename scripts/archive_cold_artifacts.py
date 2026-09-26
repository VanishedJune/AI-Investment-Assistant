"""Create verified cold archives for the explicitly approved legacy folders."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
from typing import Iterable
from uuid import uuid4
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "archive" / "cold"
LOCK_FILE = ROOT / "archive" / ".cold-archive.lock"
APPROVED_SOURCES = (
    "backup/etf_159622_alt_CH_S050_3_20260808_223220",
    "backup/etf_512690_legacy_zero_axis_20260808_215150",
    "backup/etf_512800_CHS064_1_to_CHS023_2_20260808_221526",
    "backup/etf_512800_legacy_zero_axis_CH_S050_3_20260808_201709",
    "backup/etf_516150_ch_s050_3_20260808_223538",
    "backup/etf_516150_ch_s064_1_20260808_221609",
    "model_iteration/etf_512010/archive",
)
STALE_LOCK_SECONDS = 24 * 60 * 60


def _file_meta(path: Path) -> dict:
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _is_reparse_point(path: Path) -> bool:
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _files(source: Path) -> list[Path]:
    if source.is_symlink() or _is_reparse_point(source):
        raise RuntimeError(f"reparse points are forbidden: {source}")
    result: list[Path] = []
    for path in source.rglob("*"):
        if path.is_symlink() or _is_reparse_point(path):
            raise RuntimeError(f"reparse points are forbidden: {path}")
        if path.is_file():
            result.append(path)
    return sorted(result, key=lambda item: item.relative_to(source).as_posix())


def inventory(source: Path, source_relative: str) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    total = 0
    for path in _files(source):
        size = path.stat().st_size
        total += size
        meta = _file_meta(path)
        entries.append(
            {
                "path": path.relative_to(source).as_posix(),
                "size_bytes": size,
                "modified_utc": meta["modified_utc"],
            }
        )
    return {
        "schema_version": "cold-archive-v1",
        "source_relative": source_relative,
        "file_count": len(entries),
        "total_bytes": total,
        "entries": entries,
    }


def archive_stem(source_relative: str) -> str:
    return source_relative.replace("/", "__").replace("\\", "__")


def archive_paths(root: Path, source_relative: str) -> tuple[Path, Path]:
    stem = archive_stem(source_relative)
    cold = root / "archive" / "cold"
    return cold / f"{stem}.zip", cold / f"{stem}.meta.json"


def resolve_manifest_path(zip_path: Path, manifest_path: Path) -> Path:
    """优先使用新版清单，旧版 sha256 清单仅用于兼容读取（不校验其摘要值）。"""
    if manifest_path.is_file():
        return manifest_path
    legacy = zip_path.with_name(zip_path.stem + ".sha256.json")
    if legacy.is_file():
        return legacy
    return manifest_path


def _safe_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    return bool(normalized) and not pure.is_absolute() and ".." not in pure.parts


def verify_archive(zip_path: Path, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "cold-archive-v1":
        raise RuntimeError(f"unsupported manifest: {manifest_path}")
    expected_zip_size = manifest.get("archive_size_bytes")
    if expected_zip_size is not None and int(expected_zip_size) != zip_path.stat().st_size:
        raise RuntimeError(f"archive size mismatch: {zip_path}")
    expected = {
        str(entry["path"]): (
            int(entry.get("size_bytes") or entry.get("size")),
            str(entry.get("modified_utc") or ""),
        )
        for entry in manifest.get("entries", [])
    }
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]
        if any(not _safe_member(name) for name in names):
            raise RuntimeError(f"unsafe member in archive: {zip_path}")
        if set(names) != set(expected):
            raise RuntimeError(f"archive member set mismatch: {zip_path}")
        for item in archive.infolist():
            if item.is_dir():
                continue
            size, _modified = expected[item.filename]
            if item.file_size != size:
                raise RuntimeError(f"archive size mismatch: {item.filename}")
            # 读取全部内容以触发zipfile内置的CRC一致性校验；不计算任何哈希/摘要值。
            with archive.open(item, "r") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    pass
    return manifest


def _write_archive(source: Path, source_relative: str, zip_path: Path, manifest_path: Path) -> None:
    manifest = inventory(source, source_relative)
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for entry in manifest["entries"]:
            relative = str(entry["path"])
            archive.write(source / Path(relative), arcname=relative)
    zip_meta = _file_meta(zip_path)
    manifest["archive_size_bytes"] = zip_meta["size_bytes"]
    manifest["archive_modified_utc"] = zip_meta["modified_utc"]
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    verify_archive(zip_path, manifest_path)


class ArchiveLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "ArchiveLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                age = time.time() - self.path.stat().st_mtime
            except OSError as error:
                raise RuntimeError(f"cannot inspect archive lock: {self.path}") from error
            if age < 0 or age < STALE_LOCK_SECONDS:
                raise RuntimeError(f"archive operation already running or lock not stale: {self.path}")
            self.path.unlink(missing_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"archive operation already running: {self.path}") from error
        payload = {
            "pid": os.getpid(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        os.write(self.fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        return self

    def __exit__(self, *_args: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def check(root: Path = ROOT, sources: Iterable[str] = APPROVED_SOURCES) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for relative in sources:
        source = root / Path(relative)
        zip_path, manifest_path = archive_paths(root, relative)
        if source.is_dir():
            files = _files(source)
            size = sum(path.stat().st_size for path in files)
            status = "source_ready"
            count = len(files)
        elif zip_path.is_file():
            existing_manifest = resolve_manifest_path(zip_path, manifest_path)
            if not existing_manifest.is_file():
                raise FileNotFoundError(f"archive manifest missing: {zip_path}")
            manifest = verify_archive(zip_path, existing_manifest)
            size = int(manifest["total_bytes"])
            count = int(manifest["file_count"])
            status = "already_archived"
        else:
            raise FileNotFoundError(f"approved source is missing without valid archive: {source}")
        candidates.append(
            {
                "source_relative": relative,
                "status": status,
                "file_count": count,
                "original_bytes": size,
                "estimated_compressed_bytes": int(size * 0.05),
            }
        )
    usage = shutil.disk_usage(root)
    return {
        "mode": "check",
        "root": str(root),
        "available_bytes": usage.free,
        "estimate_method": "5_percent_conservative_json_ledger_ratio",
        "candidates": candidates,
    }


def commit(root: Path = ROOT, sources: Iterable[str] = APPROVED_SOURCES) -> dict[str, object]:
    sources = tuple(sources)
    before = check(root, sources)
    source_bytes = sum(
        int(item["original_bytes"])
        for item in before["candidates"]
        if item["status"] == "source_ready"
    )
    if int(before["available_bytes"]) < max(256 * 1024 * 1024, int(source_bytes * 0.15)):
        raise RuntimeError("insufficient free space for verified archive staging")
    root_resolved = root.resolve()
    with ArchiveLock(root / "archive" / ".cold-archive.lock"):
        staging = Path(tempfile.mkdtemp(prefix=".cold-staging-", dir=root / "archive"))
        prepared: list[tuple[str, Path, Path]] = []
        try:
            for relative in sources:
                source = (root / relative).resolve()
                if not source.is_relative_to(root_resolved):
                    raise RuntimeError(f"source escapes project root: {source}")
                final_zip, final_manifest = archive_paths(root, relative)
                existing_manifest = resolve_manifest_path(final_zip, final_manifest)
                if final_zip.is_file() and existing_manifest.is_file():
                    verify_archive(final_zip, existing_manifest)
                    prepared.append((relative, final_zip, existing_manifest))
                    continue
                if final_zip.exists() or final_manifest.exists():
                    raise RuntimeError(f"partial final archive exists for {relative}")
                if not source.is_dir():
                    raise FileNotFoundError(source)
                staged_zip = staging / final_zip.name
                staged_manifest = staging / final_manifest.name
                _write_archive(source, relative, staged_zip, staged_manifest)
                prepared.append((relative, staged_zip, staged_manifest))

            (root / "archive" / "cold").mkdir(parents=True, exist_ok=True)
            finals: list[tuple[str, Path, Path]] = []
            for relative, zip_path, manifest_path in prepared:
                final_zip, final_manifest = archive_paths(root, relative)
                if zip_path.parent == staging:
                    os.replace(zip_path, final_zip)
                    os.replace(manifest_path, final_manifest)
                verify_archive(final_zip, final_manifest)
                finals.append((relative, final_zip, final_manifest))

            removed: list[str] = []
            for relative, final_zip, final_manifest in finals:
                verify_archive(final_zip, final_manifest)
                source = (root / relative).resolve()
                if source.is_dir():
                    if not source.is_relative_to(root_resolved):
                        raise RuntimeError(f"source escapes project root: {source}")
                    shutil.rmtree(source)
                    removed.append(relative)

            after_free = shutil.disk_usage(root).free
            report = {
                "schema_version": "cold-archive-run-v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "before": before,
                "removed_sources": removed,
                "available_bytes_after": after_free,
                "bytes_released": after_free - int(before["available_bytes"]),
                "restore_command": "python scripts/archive_cold_artifacts.py --restore <source_relative>",
            }
            report_path = root / "archive" / "cold" / (
                "archive-run-" + datetime.now().strftime("%Y%m%dT%H%M%S") + ".json"
            )
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def restore(source_relative: str, root: Path = ROOT) -> dict[str, object]:
    if source_relative not in APPROVED_SOURCES:
        raise RuntimeError("restore source is not in the approved allowlist")
    destination = (root / source_relative).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise RuntimeError("restore destination escapes project root")
    if destination.exists():
        raise FileExistsError(f"restore destination already exists: {destination}")
    zip_path, manifest_path = archive_paths(root, source_relative)
    manifest_path = resolve_manifest_path(zip_path, manifest_path)
    manifest = verify_archive(zip_path, manifest_path)
    temporary = destination.parent / f".{destination.name}.restore-{uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            if any(not _safe_member(item.filename) for item in archive.infolist()):
                raise RuntimeError("unsafe archive member")
            archive.extractall(temporary)
        expected = {
            str(entry["path"]): (
                int(entry.get("size_bytes") or entry.get("size")),
                str(entry.get("modified_utc") or ""),
            )
            for entry in manifest["entries"]
        }
        restored_files = _files(temporary)
        actual_names = {path.relative_to(temporary).as_posix() for path in restored_files}
        if actual_names != set(expected):
            raise RuntimeError("restored member set mismatch")
        for path in restored_files:
            relative = path.relative_to(temporary).as_posix()
            size, _modified = expected[relative]
            if path.stat().st_size != size:
                raise RuntimeError(f"restored size mismatch: {relative}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return {"restored": source_relative, "file_count": int(manifest["file_count"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--commit", action="store_true")
    modes.add_argument("--restore", metavar="SOURCE_RELATIVE")
    args = parser.parse_args()
    try:
        if args.check:
            report = check()
        elif args.commit:
            report = commit()
        else:
            report = restore(args.restore)
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
