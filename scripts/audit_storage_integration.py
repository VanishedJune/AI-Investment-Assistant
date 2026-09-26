"""Read-only inventory for the InvestmentLab/monthly-report integration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


MONTHLY_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = MONTHLY_ROOT.parent
LAB_ROOT = ASSETS_ROOT / "Asset" / "ETF-Investment-Lab"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path, *, exclude: set[Path]) -> dict[str, object]:
    files = []
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(parent in exclude for parent in (path, *path.parents)):
            continue
        stat = path.stat()
        total += stat.st_size
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _hash(path),
            }
        )
    return {"root": str(root), "file_count": len(files), "total_bytes": total, "files": files}


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _database_status(database: Path) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        requested = (
            "instruments",
            "market_prices",
            "indicator_records",
            "investment_calendar_entries",
            "real_accounts",
            "real_account_positions",
            "v35_model_versions",
            "v37_model_versions",
            "v37_forecasts",
        )
        counts = {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in requested
            if table in tables
        }
        return {
            "path": str(database),
            "size": database.stat().st_size,
            "sha256": _hash(database),
            "quick_check": connection.execute("PRAGMA quick_check").fetchone()[0],
            "user_version": int(connection.execute("PRAGMA user_version").fetchone()[0]),
            "key_table_counts": counts,
        }
    finally:
        connection.close()


def create_report(phase: str) -> dict[str, object]:
    audit_root = MONTHLY_ROOT / "archive" / "audits"
    excludes = {audit_root.resolve()}
    protected = [
        MONTHLY_ROOT / "data_manifest.json",
        MONTHLY_ROOT / "portfolio_state.json",
        MONTHLY_ROOT / "app" / "features" / "latest.json",
        MONTHLY_ROOT / "app" / "reports" / "monthly" / "latest.json",
        *sorted((MONTHLY_ROOT / "scripts").glob("投资决策月报*.html")),
        *sorted(MONTHLY_ROOT.glob("投资决策月报*.lnk")),
    ]
    webviews = sorted((LAB_ROOT / "data").glob("webview2-[0-9]*"))
    archive_candidates = [MONTHLY_ROOT / relative for relative in (
        "backup/etf_159622_alt_CH_S050_3_20260808_223220",
        "backup/etf_512690_legacy_zero_axis_20260808_215150",
        "backup/etf_512800_CHS064_1_to_CHS023_2_20260808_221526",
        "backup/etf_512800_legacy_zero_axis_CH_S050_3_20260808_201709",
        "backup/etf_516150_ch_s050_3_20260808_223538",
        "backup/etf_516150_ch_s064_1_20260808_221609",
        "model_iteration/etf_512010/archive",
    )]
    return {
        "schema_version": "storage-integration-audit-v1",
        "phase": phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "investment_lab": _inventory(LAB_ROOT, exclude=excludes),
        "monthly_project": _inventory(MONTHLY_ROOT, exclude=excludes),
        "database": _database_status(LAB_ROOT / "data" / "investment_lab.db"),
        "protected_monthly_artifacts": {
            str(path.relative_to(MONTHLY_ROOT)): {
                "size": path.stat().st_size,
                "sha256": _hash(path),
            }
            for path in protected
            if path.is_file()
        },
        "webview_profiles": {path.name: _tree_size(path) for path in webviews if path.is_dir()},
        "cold_archive_candidates": {
            str(path.relative_to(MONTHLY_ROOT)): _tree_size(path)
            for path in archive_candidates
            if path.is_dir()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "after"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = create_report(args.phase)
    output = args.output or (
        MONTHLY_ROOT
        / "archive"
        / "audits"
        / f"integration-{args.phase}-{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": "ok", "phase": args.phase, "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
