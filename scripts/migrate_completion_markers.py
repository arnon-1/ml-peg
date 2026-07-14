"""
Migrate completion markers to content-based fingerprints.

`calc_fingerprint` used to hash local files referenced by a model's
configuration (e.g. checkpoints) by size and modification time; it now hashes
their contents. Every marker written before the change therefore holds an
outdated fingerprint, and all calculations would rerun.

Run this script once per machine after pulling the change and before the next
pytest run. Every marker entry whose fingerprint is valid under the old scheme
is rewritten to the new scheme; entries that match neither scheme are
genuinely stale and are left untouched so they rerun. Before a marker is
modified, its original content is backed up to `.completed_old.json` alongside
it (an existing backup is never overwritten). Safe to run repeatedly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml_peg.calcs.utils.completion import (
    MARKER_FILENAME,
    _local_files,
    _model_config,
    calc_fingerprint,
)

CALCS_DIR = Path(__file__).resolve().parent.parent / "ml_peg" / "calcs"
BACKUP_FILENAME = ".completed_old.json"


def legacy_fingerprint(calc_dir: Path, model_name: str) -> str:
    """
    Fingerprint a calculation's inputs as `calc_fingerprint` did before.

    Identical to the current `calc_fingerprint` except that local files are
    hashed by size and modification time instead of by content.

    Parameters
    ----------
    calc_dir
        Directory containing the benchmark's calculation script(s).
    model_name
        Name of the model the calculation runs with.

    Returns
    -------
    str
        Hex digest identifying the calculation's inputs.
    """
    config = _model_config(model_name)

    sha = hashlib.sha256()
    for source in sorted(Path(calc_dir).glob("*.py")):
        sha.update(source.name.encode())
        sha.update(source.read_bytes())
    sha.update(json.dumps(config, sort_keys=True, default=str).encode())
    for path in _local_files(config):
        stat = path.stat()
        sha.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return sha.hexdigest()


def migrate(calcs_dir: Path = CALCS_DIR) -> dict[str, int]:
    """
    Migrate all completion markers under a calculations directory.

    Parameters
    ----------
    calcs_dir
        Directory searched recursively for `outputs/<model>/.completed.json`.

    Returns
    -------
    dict[str, int]
        Marker files scanned and unreadable, and entries migrated, already
        current, and left stale.
    """
    counts = dict.fromkeys(("markers", "migrated", "current", "stale", "unreadable"), 0)
    # Fingerprints per (calc_dir, model): markers hold one entry per test
    fingerprints: dict[tuple[Path, str], tuple[str, str]] = {}

    for marker in sorted(calcs_dir.glob(f"**/outputs/*/{MARKER_FILENAME}")):
        counts["markers"] += 1
        try:
            original = marker.read_text(encoding="utf8")
            content = json.loads(original)
        except (OSError, json.JSONDecodeError) as err:
            counts["unreadable"] += 1
            print(f"[unreadable] {marker}: {err}")
            continue
        if not isinstance(content, dict):
            counts["unreadable"] += 1
            print(f"[unreadable] {marker}: not a JSON object")
            continue

        model_name = marker.parent.name
        calc_dir = marker.parent.parent.parent
        key = (calc_dir, model_name)
        if key not in fingerprints:
            fingerprints[key] = (
                calc_fingerprint(calc_dir, model_name),
                legacy_fingerprint(calc_dir, model_name),
            )
        new, legacy = fingerprints[key]

        changed = False
        for entry in content.values():
            stored = entry.get("fingerprint") if isinstance(entry, dict) else None
            if stored == new:
                counts["current"] += 1
            elif stored == legacy:
                entry["fingerprint"] = new
                counts["migrated"] += 1
                changed = True
            else:
                counts["stale"] += 1

        if changed:
            backup = marker.parent / BACKUP_FILENAME
            if not backup.exists():
                backup.write_text(original, encoding="utf8")
            marker.write_text(
                json.dumps(content, indent=2, sort_keys=True), encoding="utf8"
            )

    return counts


def main() -> None:
    """Run the migration and print a summary."""
    counts = migrate()
    print(
        f"Scanned {counts['markers']} marker file(s): "
        f"{counts['migrated']} entr(y/ies) migrated, "
        f"{counts['current']} already current, "
        f"{counts['stale']} stale (left to rerun), "
        f"{counts['unreadable']} unreadable."
    )


if __name__ == "__main__":
    main()
