#!/usr/bin/env python3
"""Dry-run or delete rejected Phase 9 PM AI v3 artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("reports/ml/phase10a_phase9_cleanup_manifest.json")


PATTERNS = [
    "data/ml/portfolio_manager_v3",
    "models/ml/portfolio_manager_v3",
    "logs/backtests/rookie_dealer_02_v2_93*",
    "logs/backtests/rookie_dealer_02_v2_94*",
    "config/profiles/rookie_dealer_02_v2_93*",
    "config/profiles/rookie_dealer_02_v2_94*",
]


def _size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file() or child.is_symlink())


def _paths(root: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in PATTERNS:
        matches = sorted(root.glob(pattern))
        found.extend(matches)
    out: list[Path] = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def build_manifest(root: Path, *, delete: bool) -> dict:
    paths = _paths(root)
    before = sum(_size(path) for path in paths)
    deleted_paths: list[str] = []
    if delete:
        for path in paths:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            deleted_paths.append(str(path.relative_to(root)))
    after = sum(_size(path) for path in paths)
    return {
        "dry_run_reported": True,
        "deleted": bool(delete),
        "deleted_path_count": len(deleted_paths if delete else paths),
        "bytes_before": int(before),
        "bytes_after": int(after),
        "bytes_saved": int(before - after),
        "patterns": PATTERNS,
        "paths": deleted_paths if delete else [str(path.relative_to(root)) for path in paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up rejected Phase 9 PM AI v3 artifacts")
    parser.add_argument("--root", default=str(ROOT), help="Repository root")
    parser.add_argument("--delete", action="store_true", help="Delete matched paths after writing the planned list")
    args = parser.parse_args()
    root = Path(args.root)
    manifest = build_manifest(root, delete=bool(args.delete))
    path = root / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
