#!/usr/bin/env python3
"""Review or install this checkout's versioned CuRobo robot corrections.

The default is read-only. Only --apply writes files, and replacements are
backed up under outputs/curobo_override_backups before any target is changed.
This script uses the Python standard library and does not load CUDA/Isaac Sim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
OVERLAYS = ROOT / "configs" / "curobo_overrides"
CUROBO = ROOT / "curobo"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def confined(path: Path, base: Path) -> Path:
    """Reject paths or symlinks that leave the designated checkout subtree."""
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Path leaves {base}: {path}")
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink target: {path}")
    return path


def validate_checkout() -> None:
    for package in ("isaaclab", "isaaclab_tasks"):
        if not (ROOT / "source" / package).is_dir():
            raise ValueError(f"Not an AffordSim checkout: {ROOT}")
    if CUROBO.is_symlink() or CUROBO.resolve() != ROOT / "curobo":
        raise ValueError("curobo must be a real directory in this checkout, not a symlink")
    if not (CUROBO / "src" / "curobo" / "__init__.py").is_file():
        raise ValueError(f"Extract this checkout's curobo.zip first: missing package at {CUROBO}")
    confined(OVERLAYS, ROOT)
    confined(CUROBO / "src" / "curobo" / "__init__.py", CUROBO)


def make_plan() -> list[dict]:
    manifest_path = confined(OVERLAYS / "manifest.json", OVERLAYS)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("Unsupported or empty override manifest")
    plan = []
    seen = set()
    for entry in manifest["files"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Invalid relative override path: {relative}")
        if relative.parts[:3] != ("src", "curobo", "content"):
            raise ValueError(f"Only CuRobo content overrides are supported: {relative}")
        if str(relative) in seen:
            raise ValueError(f"Duplicate override path: {relative}")
        seen.add(str(relative))
        source = confined(OVERLAYS / relative, OVERLAYS)
        target = confined(CUROBO / relative, CUROBO)
        if not source.is_file():
            raise ValueError(f"Override source is missing: {source}")
        if target.exists() and not target.is_file():
            raise ValueError(f"Target is not a file: {target}")
        new_data = source.read_bytes()
        if digest(new_data) != entry["sha256"]:
            raise ValueError(f"Override checksum mismatch: {source}")
        old_data = target.read_bytes() if target.exists() else None
        status = "unchanged" if old_data == new_data else ("replace" if old_data is not None else "create")
        plan.append({
            "relative": str(relative), "source": source, "target": target,
            "old_data": old_data, "new_data": new_data, "status": status,
            "old_sha256": digest(old_data) if old_data is not None else None,
            "new_sha256": entry["sha256"],
            "archive_sha256": entry.get("archive_sha256"),
        })
    return plan


def install(plan: list[dict]) -> Path | None:
    changes = [item for item in plan if item["status"] != "unchanged"]
    if not changes:
        return None
    # Finish validating every target before creating backups or changing files.
    for item in changes:
        current = item["target"].read_bytes() if item["target"].exists() else None
        if current != item["old_data"]:
            raise ValueError(f"Target changed after review: {item['target']}")
    backup_parent = confined(ROOT / "outputs" / "curobo_override_backups", ROOT)
    backup_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}"
    backup_root = confined(backup_parent / backup_id, ROOT)
    backup_root.mkdir(parents=True, exist_ok=False)
    records = []
    # All existing files are backed up before the first replacement.
    for item in changes:
        record = {key: item[key] for key in ("relative", "status", "old_sha256", "new_sha256")}
        if item["old_data"] is not None:
            backup_file = confined(backup_root / item["relative"], backup_root)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["target"], backup_file)
            if backup_file.read_bytes() != item["old_data"]:
                raise ValueError(f"Backup verification failed: {backup_file}")
            record["backup"] = str(backup_file.relative_to(ROOT))
        records.append(record)
    (backup_root / "manifest.json").write_text(json.dumps({
        "repository": str(ROOT), "target_root": str(CUROBO), "files": records,
    }, indent=2) + "\n")
    for item in changes:
        target = confined(item["target"], CUROBO)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(item["new_data"])
                stream.flush()
                os.fsync(stream.fileno())
            shutil.copymode(item["source"], temporary)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return backup_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Back up differing old files and install the overrides")
    mode.add_argument("--dry-run", action="store_true", help="Review changes without writing anything (default)")
    args = parser.parse_args()
    try:
        validate_checkout()
        plan = make_plan()
        print(f"Repository: {ROOT}")
        for item in plan:
            print(f"{item['status']:9} {item['relative']}")
            if item["status"] == "replace":
                print(f"          SHA-256 {item['old_sha256']} -> {item['new_sha256']}")
                if item["archive_sha256"] and item["old_sha256"] != item["archive_sha256"]:
                    print("          Existing file differs from the bundled archive; --apply will preserve a backup.")
        if not args.apply:
            print("Dry run: no files written. Use --apply to install the listed changes.")
            return 0
        backup_root = install(plan)
        if backup_root is None:
            print("Already current: no files written.")
        else:
            print(f"Installed overrides. Backup and change record: {backup_root}")
            print("Start a fresh collection process to load the corrected CuRobo robot configuration.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
