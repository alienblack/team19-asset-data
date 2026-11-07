"""Persistent configuration approvals.

An approval is a record, kept in ``config/approvals.json`` and committed with
the configuration, that names the SHA-256 of a configuration file's content
at the moment a person accepted it. Authority never lives in a notebook
kernel: whether a file is approved is decided by comparing its current hash
with the record, so editing the file invalidates the approval by itself and a
fresh kernel can verify everything without anyone present.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECORD_VERSION = 1
APPROVED, CHANGED, MISSING, UNAPPROVED = "approved", "changed", "missing", "unapproved"


class ApprovalError(ValueError):
    """Raised when an approval cannot be recorded or read."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ApprovalRecord:
    sha256: str
    approved_by: str
    approved_at: str
    note: str = ""


@dataclass(frozen=True)
class ApprovalState:
    """State of one configuration file against the record."""

    name: str
    path: Path
    status: str  # approved | changed | missing | unapproved
    sha256: str | None
    record: ApprovalRecord | None

    @property
    def is_approved(self) -> bool:
        return self.status == APPROVED


class ApprovalStore:
    """Reads and writes ``approvals.json``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> dict[str, ApprovalRecord]:
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ApprovalError(f"{self.path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("version") != RECORD_VERSION or not isinstance(raw.get("files"), dict):
            raise ApprovalError(f"{self.path} is not an approvals record (version {RECORD_VERSION}).")
        records: dict[str, ApprovalRecord] = {}
        for name, entry in raw["files"].items():
            try:
                records[name] = ApprovalRecord(
                    sha256=str(entry["sha256"]),
                    approved_by=str(entry["approved_by"]),
                    approved_at=str(entry["approved_at"]),
                    note=str(entry.get("note", "")),
                )
            except (KeyError, TypeError) as exc:
                raise ApprovalError(f"{self.path}: malformed entry for '{name}'.") from exc
        return records

    def write(self, records: dict[str, ApprovalRecord]) -> Path:
        payload = {
            "version": RECORD_VERSION,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": {name: asdict(record) for name, record in sorted(records.items())},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return self.path

    def state(self, name: str, path: Path) -> ApprovalState:
        record = self.read().get(name)
        if not path.is_file():
            return ApprovalState(name=name, path=path, status=MISSING, sha256=None, record=record)
        digest = file_sha256(path)
        if record is None:
            return ApprovalState(name=name, path=path, status=UNAPPROVED, sha256=digest, record=None)
        status = APPROVED if record.sha256 == digest else CHANGED
        return ApprovalState(name=name, path=path, status=status, sha256=digest, record=record)

    def approve(self, name: str, path: Path, approved_by: str, note: str = "") -> ApprovalRecord:
        if not isinstance(approved_by, str) or not approved_by.strip():
            raise ApprovalError("An approval must name who approved (approved_by).")
        if not path.is_file():
            raise ApprovalError(f"Cannot approve '{name}': file not found at {path}.")
        record = ApprovalRecord(
            sha256=file_sha256(path),
            approved_by=approved_by.strip(),
            approved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            note=note.strip(),
        )
        records = self.read()
        records[name] = record
        self.write(records)
        return record

    def revoke(self, name: str) -> bool:
        records = self.read()
        if name not in records:
            return False
        del records[name]
        self.write(records)
        return True

    def to_jsonable(self) -> dict[str, Any]:
        return {name: asdict(record) for name, record in self.read().items()}
