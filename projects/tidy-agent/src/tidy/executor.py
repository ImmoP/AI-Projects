"""Deterministic and safety-critical plan execution.

This module is intentionally independent from smolagents. An LLM may propose a
plan, but only this code validates paths, asks its caller to opt into applying a
plan, moves files, journals successful changes, and performs rollback.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

LOGGER = logging.getLogger(__name__)
DEFAULT_JOURNAL_DIR = Path.home() / ".tidy-agent" / "journal"
MIN_GROUP_SIZE = 3
MAX_GROUP_FOLDER_LENGTH = 40
RESERVED_FOLDER_NAMES = frozenset(
    {
        ".",
        "..",
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)


class UnsafeMoveError(ValueError):
    """Raised when a plan entry violates an executor safety boundary."""


class PlanChangedError(RuntimeError):
    """Raised when an approved plan no longer matches the filesystem."""


class JournalError(RuntimeError):
    """Raised when durable recovery state cannot be persisted."""

    def __init__(self, message: str, *, result: Any | None = None) -> None:
        super().__init__(message)
        self.result = result


class PartialExecutionError(RuntimeError):
    """Raised with structured recovery context after at least one mutation."""

    def __init__(
        self,
        message: str,
        *,
        result: ExecutionResult,
        failed_move: EntryResult | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result
        self.failed_move = failed_move


class RunLockError(RuntimeError):
    """Raised when the target root's mutation lock cannot be acquired."""


class UnverifiedPlatformError(RuntimeError):
    """Raised when this platform's pre-mutation identity check can't be trusted."""


# SourceMetadata's full-field equality check in _move_no_clobber (device,
# inode, mode, size, mtime_ns, ctime_ns) is this executor's core TOCTOU /
# tamper-detection guarantee: the identity captured when a plan was approved
# must still hold immediately before the rename, or the mutation is refused.
#
# On Windows/NTFS this has been observed to spuriously fire on files nobody
# touched: a stat() taken immediately after a write-and-close can report an
# st_ctime_ns that has not yet "settled" -- a stat() microseconds later on the
# *same, untouched* file returns a different value before settling and then
# stays stable. Two stat() calls on an untouched file disagreeing is exactly
# the failure mode this check exists to catch, so on this platform the check
# cannot currently distinguish a real change from filesystem-timing noise.
# Mutating anyway would mean silently running with what looks like an active
# safety guarantee but functions as a coin flip. Until resolved, mutation is
# refused outright rather than proceeding unverified. See README Limitations
# ("The pre-move identity re-check is currently unverified on Windows") and
# the tracked follow-up issue for reproduction steps and status.
_METADATA_IDENTITY_VERIFICATION_RELIABLE = os.name != "nt"


@dataclass(frozen=True)
class EntryResult:
    source: str
    destination: str
    reason: str
    status: str
    message: str = ""


@dataclass(frozen=True)
class GroupEntryResult:
    folder_name: str
    files: tuple[str, ...]
    reason: str
    status: str
    message: str = ""
    invalid_folder_name: bool = False


@dataclass(frozen=True)
class GroupingResult:
    """Executor decision for untrusted semantic group proposals."""

    entries: tuple[GroupEntryResult, ...] = field(default_factory=tuple)

    @property
    def grouped_files(self) -> frozenset[str]:
        return frozenset(
            filename
            for entry in self.entries
            if entry.status == "accepted"
            for filename in entry.files
        )

    @property
    def invalid_folder_names(self) -> int:
        return sum(entry.invalid_folder_name for entry in self.entries)

    @property
    def moves(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "source": filename,
                "destination": f"{entry.folder_name}/{filename}",
                "reason": entry.reason,
            }
            for entry in self.entries
            if entry.status == "accepted"
            for filename in entry.files
        )


@dataclass(frozen=True)
class _PathMove:
    source: Path
    destination: Path
    reason: str


@dataclass(frozen=True)
class SourceMetadata:
    """Identity and change indicators captured for an approved source."""

    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def capture(cls, path: Path) -> SourceMetadata:
        return cls.from_stat(path.stat(follow_symlinks=False))

    @classmethod
    def from_stat(cls, stat_result: os.stat_result) -> SourceMetadata:
        return cls(
            device=stat_result.st_dev,
            inode=stat_result.st_ino,
            mode=stat_result.st_mode,
            size=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
            ctime_ns=stat_result.st_ctime_ns,
        )

    def same_moved_object(self, other: SourceMetadata) -> bool:
        """Compare fields stable across a same-filesystem rename.

        ``ctime`` commonly changes when a file is renamed, so recovery of the
        pre-move ``moving`` state cannot use it. Normal pre-move validation uses
        full dataclass equality and therefore does include ``ctime``.
        """
        return (
            self.device,
            self.inode,
            self.mode,
            self.size,
            self.mtime_ns,
        ) == (
            other.device,
            other.inode,
            other.mode,
            other.size,
            other.mtime_ns,
        )


@dataclass(frozen=True)
class DirectoryIdentity:
    """Stable identity fields for an approved directory object."""

    device: int
    inode: int
    mode: int

    @classmethod
    def capture(cls, path: Path) -> DirectoryIdentity:
        stat_result = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(stat_result.st_mode):
            raise NotADirectoryError(path)
        return cls(stat_result.st_dev, stat_result.st_ino, stat_result.st_mode)

    @classmethod
    def from_stat(cls, stat_result: os.stat_result) -> DirectoryIdentity:
        if not stat.S_ISDIR(stat_result.st_mode):
            raise NotADirectoryError("bound path is not a directory")
        return cls(stat_result.st_dev, stat_result.st_ino, stat_result.st_mode)


@dataclass(frozen=True)
class ParentBinding:
    """The closest approved directory anchoring a move parent."""

    relative_path: str
    identity: DirectoryIdentity


@dataclass(frozen=True)
class ValidatedMove:
    """One exact move approved by the user."""

    source: str
    destination: str
    reason: str
    source_metadata: SourceMetadata
    source_parent: ParentBinding
    destination_anchor: ParentBinding
    destination_missing_parents: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedPlan:
    """Concrete collision-resolved plan produced without filesystem writes."""

    root: str
    root_identity: DirectoryIdentity
    moves: tuple[ValidatedMove, ...]
    entries: tuple[EntryResult, ...]


@dataclass(frozen=True)
class ExecutionResult:
    root: str
    applied: bool
    entries: tuple[EntryResult, ...] = field(default_factory=tuple)
    run_id: str | None = None
    journal_path: str | None = None
    journal_state: str | None = None
    validated_plan: ValidatedPlan | None = field(default=None, repr=False, compare=False)

    @property
    def moved_count(self) -> int:
        return sum(entry.status in {"moved", "restored"} for entry in self.entries)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id(now: datetime) -> str:
    return f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"


def _candidate(root: Path, value: str) -> Path:
    """Resolve an untrusted plan value only as a path relative to *root*.

    In particular, do not call ``expanduser()`` here: a legitimate filename such
    as ``~$report.docx`` is not home-directory syntax, and plan paths must never
    opt into home-directory expansion anyway.
    """
    path = Path(value)
    if path.is_absolute():
        raise UnsafeMoveError("plan paths must be relative to the target directory")
    if path.parts and path.parts[0].startswith("~") and len(path.parts) > 1:
        raise UnsafeMoveError("plan paths must not use home-directory shorthand")
    return root / path


def _inside(root: Path, path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise UnsafeMoveError(f"{label} escapes the target directory") from exc
    if relative == Path("."):
        raise UnsafeMoveError(f"{label} must name a file below the target directory")
    if any(part.startswith(".") for part in relative.parts):
        raise UnsafeMoveError(f"{label} contains a hidden path component")
    return resolved


def _has_symlink_component(root: Path, path: Path) -> bool:
    """Return whether *path* or an existing component below *root* is a symlink."""
    current = path
    while current != root and current != current.parent:
        if current.is_symlink():
            return True
        current = current.parent
    return False


def _validate_text(entry: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = entry.get(key, "" if allow_empty else None)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise UnsafeMoveError(f"{key} must be a non-empty string")
    return value.strip()


def _validate_move(root: Path, entry: Mapping[str, Any]) -> _PathMove:
    if not isinstance(entry, Mapping):
        raise UnsafeMoveError("plan entry must be an object")

    source_text = _validate_text(entry, "source")
    destination_text = _validate_text(entry, "destination")
    reason = _validate_text(entry, "reason", allow_empty=True)
    raw_source = _candidate(root, source_text)
    raw_destination = _candidate(root, destination_text)

    # Check before resolve(), which follows symlinks. Parent symlinks are rejected
    # as well, closing an easy route around the resolved-path boundary.
    if _has_symlink_component(root, raw_source):
        raise UnsafeMoveError("symlink sources and source paths are never moved")
    if _has_symlink_component(root, raw_destination):
        raise UnsafeMoveError("symlink destination paths are never used")

    source = _inside(root, raw_source, label="source")
    destination = _inside(root, raw_destination, label="destination")
    if source == destination:
        raise UnsafeMoveError("source and destination are identical")
    if source.exists() and not source.is_file():
        raise UnsafeMoveError("source is not a regular file")
    return _PathMove(source=source, destination=destination, reason=reason)


def _validate_group_folder_name(name: Any, reserved: set[str]) -> str:
    """Validate an LLM-proposed folder name at the executor boundary."""
    if not isinstance(name, str) or not name:
        raise UnsafeMoveError("folder_name must be a non-empty string")
    if len(name) > MAX_GROUP_FOLDER_LENGTH:
        raise UnsafeMoveError(
            f"folder_name must be at most {MAX_GROUP_FOLDER_LENGTH} characters"
        )
    if not all(character.isalnum() or character in {"-", "_"} for character in name):
        raise UnsafeMoveError(
            "folder_name may contain only letters, digits, '-' and '_'"
        )
    folded = unicodedata.normalize("NFC", name).casefold()
    if folded in RESERVED_FOLDER_NAMES:
        raise UnsafeMoveError("folder_name is reserved")
    if folded in reserved:
        raise UnsafeMoveError("folder_name collides with an existing category or group")
    return name


def _filesystem_case_sensitive(directory: Path) -> bool:
    """Detect case semantics read-only when possible, else choose safely.

    An existing alphabetic directory entry is enough to probe the mounted
    filesystem without creating a file. On Darwin the conservative fallback is
    case-insensitive because the common APFS/HFS+ configurations are; this can
    produce an unnecessary suffix on a case-sensitive volume but never a late
    overwrite. Other POSIX platforms default to case-sensitive and Windows to
    case-insensitive.
    """
    try:
        entries = tuple(directory.iterdir())
    except OSError:
        entries = ()
    for entry in entries:
        alternate = "".join(
            character.swapcase() if character.isalpha() else character
            for character in entry.name
        )
        if alternate == entry.name:
            continue
        try:
            alternate_stat = (directory / alternate).stat(follow_symlinks=False)
            entry_stat = entry.stat(follow_symlinks=False)
        except FileNotFoundError:
            return True
        except OSError:
            continue
        return not (
            alternate_stat.st_dev == entry_stat.st_dev
            and alternate_stat.st_ino == entry_stat.st_ino
        )
    if os.name == "nt":
        return False
    return sys.platform != "darwin"


def _reservation_key(path: Path, *, case_sensitive: bool) -> str:
    key = unicodedata.normalize("NFC", path.as_posix())
    return key if case_sensitive else key.casefold()


def _collision_free(
    path: Path,
    reserved: set[str],
    *,
    case_sensitive: bool,
) -> Path:
    occupied: set[str] = set()
    if path.parent.is_dir():
        try:
            occupied = {
                _reservation_key(entry, case_sensitive=case_sensitive)
                for entry in path.parent.iterdir()
            }
        except OSError:
            occupied = set()
    candidate = path
    counter = 1
    while (
        candidate.exists()
        or candidate.is_symlink()
        or _reservation_key(candidate, case_sensitive=case_sensitive) in reserved
        or _reservation_key(candidate, case_sensitive=case_sensitive) in occupied
    ):
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        counter += 1
    return candidate


def _nearest_existing_parent(root: Path, parent: Path) -> ParentBinding:
    current = parent
    while current != root and not current.exists():
        if current.is_symlink():
            raise UnsafeMoveError("destination parent path became a symlink")
        current = current.parent
    if _has_symlink_component(root, current):
        raise UnsafeMoveError("destination parent path contains a symlink")
    relative = _relative(root, current) if current != root else "."
    return ParentBinding(relative, DirectoryIdentity.capture(current))


def _missing_parent_paths(
    root: Path, parent: Path, anchor: ParentBinding
) -> tuple[str, ...]:
    anchor_path = root if anchor.relative_path == "." else root / anchor.relative_path
    missing: list[str] = []
    current = parent
    while current != anchor_path:
        missing.append(_relative(root, current))
        current = current.parent
    missing.reverse()
    return tuple(missing)


def _binding_from_mapping(value: Any, *, label: str) -> ParentBinding:
    if not isinstance(value, Mapping):
        raise ValueError(f"journal has no valid {label} binding")
    identity = value.get("identity")
    relative_path = value.get("relative_path")
    if not isinstance(identity, Mapping) or not isinstance(relative_path, str):
        raise ValueError(f"journal has no valid {label} binding")
    return ParentBinding(relative_path, DirectoryIdentity(**identity))


_SECURE_DIR_FD = os.name == "posix" and all(
    function in os.supports_dir_fd
    for function in (os.open, os.mkdir, os.stat, os.rename)
)
_MUTATION_HOOK: Any | None = None


def _invoke_mutation_hook(kind: str, source: Path, destination: Path) -> None:
    """Test-only seam immediately before mutation-time rebinding."""
    if _MUTATION_HOOK is not None:
        _MUTATION_HOOK(kind, source, destination)


@contextmanager
def _bound_parent(
    root: Path,
    parent_relative: str,
    *,
    root_identity: DirectoryIdentity,
    anchor: ParentBinding,
    create: bool,
    expected_missing: frozenset[str] = frozenset(),
    created_by_run: set[str] | None = None,
) -> Iterator[tuple[int | None, Path, DirectoryIdentity]]:
    """Open a parent below the approved root without following symlinks.

    POSIX callers receive a directory descriptor that remains bound even if a
    pathname is renamed concurrently. Windows lacks the required ``dir_fd``
    APIs in Python's standard library, so its branch performs the strongest
    available lstat identity checks and yields a pathname.
    """
    relative = Path(parent_relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise UnsafeMoveError("move parent is not below the target root")
    parts = () if parent_relative in {"", "."} else relative.parts

    if _SECURE_DIR_FD:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(root, flags)
        current_relative = "."
        anchor_seen = False
        try:
            current_identity = DirectoryIdentity.from_stat(os.fstat(descriptor))
            if current_identity != root_identity:
                raise PlanChangedError("selected root identity changed; build a new plan")
            if anchor.relative_path == ".":
                if anchor.identity != current_identity:
                    raise PlanChangedError("approved directory identity changed")
                anchor_seen = True
            current_path = root
            for part in parts:
                next_path = current_path / part
                next_relative = _relative(root, next_path)
                try:
                    next_descriptor = os.open(part, flags, dir_fd=descriptor)
                    if (
                        next_relative in expected_missing
                        and (
                            created_by_run is None
                            or next_relative not in created_by_run
                        )
                    ):
                        os.close(next_descriptor)
                        raise PlanChangedError(
                            f"destination parent appeared after approval: {next_relative}"
                        )
                except FileNotFoundError:
                    if not create:
                        raise PlanChangedError(
                            f"approved parent disappeared: {parent_relative}"
                        )
                    os.mkdir(part, dir_fd=descriptor)
                    if created_by_run is not None:
                        created_by_run.add(next_relative)
                    next_descriptor = os.open(part, flags, dir_fd=descriptor)
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise PlanChangedError(
                            f"approved parent became a symlink or non-directory: "
                            f"{parent_relative}"
                        ) from exc
                    raise
                os.close(descriptor)
                descriptor = next_descriptor
                current_path = next_path
                current_relative = _relative(root, current_path)
                current_identity = DirectoryIdentity.from_stat(os.fstat(descriptor))
                if current_relative == anchor.relative_path:
                    if current_identity != anchor.identity:
                        raise PlanChangedError(
                            f"approved parent identity changed: {anchor.relative_path}"
                        )
                    anchor_seen = True
            if not anchor_seen:
                raise PlanChangedError(
                    f"approved parent anchor is not on path: {anchor.relative_path}"
                )
            yield descriptor, root.joinpath(*parts), current_identity
        finally:
            os.close(descriptor)
        return

    current = root
    current_identity = DirectoryIdentity.capture(current)
    if current_identity != root_identity:
        raise PlanChangedError("selected root identity changed; build a new plan")
    anchor_seen = False
    if anchor.relative_path == ".":
        if current_identity != anchor.identity:
            raise PlanChangedError("approved directory identity changed")
        anchor_seen = True
    for part in parts:
        current /= part
        try:
            path_stat = current.stat(follow_symlinks=False)
            current_relative = _relative(root, current)
            if (
                current_relative in expected_missing
                and (
                    created_by_run is None
                    or current_relative not in created_by_run
                )
            ):
                raise PlanChangedError(
                    f"destination parent appeared after approval: {current_relative}"
                )
        except FileNotFoundError:
            if not create:
                raise PlanChangedError(f"approved parent disappeared: {parent_relative}")
            current.mkdir()
            if created_by_run is not None:
                created_by_run.add(_relative(root, current))
            path_stat = current.stat(follow_symlinks=False)
        if current.is_symlink() or not stat.S_ISDIR(path_stat.st_mode):
            raise PlanChangedError(
                f"approved parent became a symlink or non-directory: {parent_relative}"
            )
        current_identity = DirectoryIdentity.from_stat(path_stat)
        current_relative = _relative(root, current)
        if current_relative == anchor.relative_path:
            if current_identity != anchor.identity:
                raise PlanChangedError(
                    f"approved parent identity changed: {anchor.relative_path}"
                )
            anchor_seen = True
    if not anchor_seen:
        raise PlanChangedError(
            f"approved parent anchor is not on path: {anchor.relative_path}"
        )
    yield None, current, current_identity


def _atomic_rename_no_replace(
    source_name: str,
    destination_name: str,
    *,
    source_parent_fd: int | None,
    destination_parent_fd: int | None,
    source_path: Path,
    destination_path: Path,
) -> None:
    """Rename without replacement or fail closed on unsupported platforms."""
    if sys.platform.startswith("linux") and source_parent_fd is not None:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOTSUP, "renameat2(RENAME_NOREPLACE) is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            source_parent_fd,
            os.fsencode(source_name),
            destination_parent_fd,
            os.fsencode(destination_name),
            1,  # RENAME_NOREPLACE
        )
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination_path))
        return
    if sys.platform == "darwin" and source_parent_fd is not None:
        libc = ctypes.CDLL(None, use_errno=True)
        renameatx_np = getattr(libc, "renameatx_np", None)
        if renameatx_np is None:
            raise OSError(errno.ENOTSUP, "renameatx_np(RENAME_EXCL) is unavailable")
        renameatx_np.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            source_parent_fd,
            os.fsencode(source_name),
            destination_parent_fd,
            os.fsencode(destination_name),
            0x00000004 | 0x00000010,  # RENAME_EXCL | RENAME_NOFOLLOW_ANY
        )
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination_path))
        return
    if os.name == "nt":  # os.rename on Windows fails if destination exists.
        os.rename(source_path, destination_path)
        return
    raise OSError(errno.ENOTSUP, "atomic no-clobber rename is unavailable")


def _ensure_same_device(
    source: SourceMetadata, destination_parent: DirectoryIdentity, destination: Path
) -> None:
    if source.device != destination_parent.device:
        raise OSError(
            errno.EXDEV,
            "cross-filesystem moves are intentionally unsupported",
            str(destination),
        )


def _move_no_clobber(
    root: Path,
    source_relative: str,
    destination_relative: str,
    *,
    root_identity: DirectoryIdentity,
    source_parent: ParentBinding,
    destination_parent: ParentBinding,
    expected_source: SourceMetadata,
    mutation_kind: str = "move",
) -> SourceMetadata:
    """Move one approved regular-file object without replacing a destination."""
    source = root / source_relative
    destination = root / destination_relative
    _invoke_mutation_hook(mutation_kind, source, destination)
    source_parent_relative = Path(source_relative).parent.as_posix()
    destination_parent_relative = Path(destination_relative).parent.as_posix()
    with _bound_parent(
        root,
        source_parent_relative,
        root_identity=root_identity,
        anchor=source_parent,
        create=False,
    ) as (source_parent_fd, source_parent_path, _), _bound_parent(
        root,
        destination_parent_relative,
        root_identity=root_identity,
        anchor=destination_parent,
        create=False,
    ) as (destination_parent_fd, destination_parent_path, destination_identity):
        source_name = Path(source_relative).name
        destination_name = Path(destination_relative).name
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        if source_parent_fd is not None:
            source_descriptor = os.open(source_name, flags, dir_fd=source_parent_fd)
        else:
            source_descriptor = os.open(source_parent_path / source_name, flags)
        try:
            current_source = SourceMetadata.from_stat(os.fstat(source_descriptor))
            if not stat.S_ISREG(current_source.mode) or current_source != expected_source:
                raise PlanChangedError(
                    f"approved source changed during execution: {source_relative}"
                )
            _ensure_same_device(current_source, destination_identity, destination)
            if os.name == "nt":  # Windows CRT handles commonly block renames.
                os.close(source_descriptor)
                source_descriptor = -1
            _atomic_rename_no_replace(
                source_name,
                destination_name,
                source_parent_fd=source_parent_fd,
                destination_parent_fd=destination_parent_fd,
                source_path=source_parent_path / source_name,
                destination_path=destination_parent_path / destination_name,
            )
            if destination_parent_fd is not None:
                moved_stat = os.stat(
                    destination_name,
                    dir_fd=destination_parent_fd,
                    follow_symlinks=False,
                )
            else:
                moved_stat = (destination_parent_path / destination_name).stat(
                    follow_symlinks=False
                )
            moved_metadata = SourceMetadata.from_stat(moved_stat)
            if not expected_source.same_moved_object(moved_metadata):
                raise PlanChangedError(
                    f"source identity changed at mutation time: {source_relative}"
                )
            return moved_metadata
        finally:
            if source_descriptor >= 0:
                os.close(source_descriptor)


def _prepare_destination_parent(
    root: Path,
    destination_relative: str,
    *,
    root_identity: DirectoryIdentity,
    anchor: ParentBinding,
    expected_missing: Sequence[str] = (),
    created_by_run: set[str] | None = None,
) -> ParentBinding:
    parent_relative = Path(destination_relative).parent.as_posix()
    with _bound_parent(
        root,
        parent_relative,
        root_identity=root_identity,
        anchor=anchor,
        create=True,
        expected_missing=frozenset(expected_missing),
        created_by_run=created_by_run,
    ) as (_, _, identity):
        return ParentBinding(parent_relative, identity)


def _source_metadata_from_mapping(value: Any) -> SourceMetadata:
    if not isinstance(value, Mapping):
        raise ValueError("journal has no valid source metadata")
    metadata = dict(value)
    # Schema v2 journals predate ctime capture. They retain the former, weaker
    # recovery semantics rather than becoming unreadable.
    metadata.setdefault("ctime_ns", 0)
    return SourceMetadata(**metadata)


@contextmanager
def _exclusive_file_guard(path: Path) -> Iterator[bool]:
    """Best-effort OS lock held for the entire move.

    On POSIX this refuses files carrying a conflicting advisory lock. The helper
    is kept separate so platform-specific in-use checks can be added without
    involving agent code.
    """
    lsof = shutil.which("lsof")
    if lsof is not None:
        try:
            inspection = subprocess.run(
                [lsof, "-t", "--", str(path)],
                capture_output=True,
                check=False,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            # If an available handle inspector cannot give a reliable answer,
            # fail closed instead of moving a potentially active file.
            yield False
            return
        if inspection.returncode == 0 and inspection.stdout.strip():
            yield False
            return
        if inspection.returncode not in {0, 1}:
            yield False
            return

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except (FileNotFoundError, PermissionError, OSError):
        yield False
        return

    locked = False
    try:
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except (BlockingIOError, PermissionError):
            pass
        except ImportError:  # pragma: no cover - exercised on Windows CI
            # Python's CRT descriptor cannot be held across a Windows rename
            # reliably. It proves readability only; root-level serialization
            # remains the actual inter-process Tidy lock on this platform.
            os.close(descriptor)
            descriptor = -1
            yield True
            return
        except OSError:
            pass
        yield locked
    finally:
        if locked:
            try:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except (ImportError, OSError):  # pragma: no cover - defensive
                pass
        if descriptor >= 0:
            os.close(descriptor)


@contextmanager
def _root_run_lock(root: Path) -> Iterator[Path]:
    """Hold one OS-backed lock for every mutation associated with *root*.

    Lock files are persistent coordination points; they are not deleted based
    on age. The operating system releases ownership when a process exits.
    """
    user_key = str(getattr(os, "getuid", lambda: "user")())
    lock_directory = Path(tempfile.gettempdir()) / f"tidy-agent-locks-{user_key}"
    lock_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_directory_stat = lock_directory.stat(follow_symlinks=False)
    if not stat.S_ISDIR(lock_directory_stat.st_mode) or lock_directory.is_symlink():
        raise RunLockError(f"unsafe root lock directory: {lock_directory}")
    if hasattr(os, "getuid") and lock_directory_stat.st_uid != os.getuid():
        raise RunLockError(f"root lock directory is owned by another user: {lock_directory}")
    root_identity = DirectoryIdentity.capture(root)
    lock_material = f"{root_identity.device}:{root_identity.inode}".encode("ascii")
    lock_key = hashlib.sha256(lock_material).hexdigest()
    lock_path = lock_directory / f"{lock_key}.lock"
    lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    lock_flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, lock_flags, 0o600)
    acquired = False
    try:
        if os.name == "posix":
            try:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except (BlockingIOError, OSError) as exc:
                raise RunLockError(
                    f"another apply or undo is already running for {root}"
                ) from exc
        elif os.name == "nt":  # pragma: no cover - exercised on Windows CI
            try:
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                acquired = True
            except (ImportError, OSError) as exc:
                raise RunLockError(
                    f"cannot acquire a safe apply/undo lock for {root}"
                ) from exc
        else:  # pragma: no cover - fail closed on unknown platforms
            raise RunLockError(
                f"root-level mutation locking is unsupported on {sys.platform}"
            )
        yield lock_path
    finally:
        if acquired:
            try:
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                elif os.name == "nt":  # pragma: no cover - Windows CI
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            except OSError:
                LOGGER.warning("Failed to release root lock %s", lock_path)
        os.close(descriptor)


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _flush_file(handle: Any) -> None:
    handle.flush()


def _write_file(handle: Any, text: str) -> None:
    handle.write(text)


def _fsync_file(handle: Any) -> None:
    os.fsync(handle.fileno())


def _replace_file(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry update where directory fsync is supported."""
    if os.name == "nt":  # Windows has no stdlib directory-fsync primitive.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                LOGGER.warning("Directory fsync is unsupported for %s", path)
                return
            raise
    finally:
        os.close(descriptor)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            _write_file(handle, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            _flush_file(handle)
            _fsync_file(handle)
        _replace_file(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class PlanExecutor:
    """Validate and optionally apply an untrusted move plan."""

    def __init__(
        self,
        directory: str | Path,
        *,
        journal_dir: str | Path = DEFAULT_JOURNAL_DIR,
    ) -> None:
        root = Path(directory).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(root)
        self.root = root
        self.root_identity = DirectoryIdentity.capture(root)
        self.case_sensitive = _filesystem_case_sensitive(root)
        self.journal_dir = Path(journal_dir).expanduser()

    def validate_groups(
        self,
        groups: Sequence[Mapping[str, Any]],
        *,
        candidate_files: Sequence[str],
        existing_categories: Sequence[str],
    ) -> GroupingResult:
        """Turn safe, sufficiently large groups into moves without writing.

        Group proposals are untrusted model output. Folder-name policy, source
        containment, category collisions, duplicate membership, and the minimum
        cluster size are therefore enforced here rather than in the prompt or
        proposal tool. Rejected/discarded files remain available to the caller's
        deterministic extension fallback.
        """
        if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
            raise TypeError("groups must be a sequence of group entries")

        candidates = set(candidate_files)
        reserved = {
            unicodedata.normalize("NFC", name).casefold()
            for name in existing_categories
        }
        already_grouped: set[str] = set()
        results: list[GroupEntryResult] = []

        for raw_group in groups:
            raw_name = (
                str(raw_group.get("folder_name", ""))
                if isinstance(raw_group, Mapping)
                else ""
            )
            raw_reason = (
                str(raw_group.get("reason", ""))
                if isinstance(raw_group, Mapping)
                else ""
            )
            raw_files = raw_group.get("files", []) if isinstance(raw_group, Mapping) else []
            display_files = (
                tuple(str(filename) for filename in raw_files)
                if isinstance(raw_files, Sequence)
                and not isinstance(raw_files, (str, bytes))
                else ()
            )
            try:
                if not isinstance(raw_group, Mapping):
                    raise UnsafeMoveError("group entry must be an object")
                folder_name = _validate_group_folder_name(
                    raw_group.get("folder_name"), reserved
                )
            except (UnsafeMoveError, OSError, ValueError) as exc:
                results.append(
                    GroupEntryResult(
                        raw_name,
                        display_files,
                        raw_reason,
                        "rejected",
                        str(exc),
                        invalid_folder_name=isinstance(raw_group, Mapping),
                    )
                )
                continue

            reason = raw_group.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                results.append(
                    GroupEntryResult(
                        folder_name,
                        display_files,
                        raw_reason,
                        "rejected",
                        "reason must be a non-empty string",
                    )
                )
                continue
            if isinstance(raw_files, (str, bytes)) or not isinstance(raw_files, Sequence):
                results.append(
                    GroupEntryResult(
                        folder_name,
                        (),
                        reason.strip(),
                        "rejected",
                        "files must be a list of direct-child filenames",
                    )
                )
                continue

            files: list[str] = []
            seen_in_group: set[str] = set()
            invalid_source: str | None = None
            for filename in raw_files:
                if not isinstance(filename, str) or not filename:
                    invalid_source = "group files must be non-empty strings"
                    break
                try:
                    raw_path = _candidate(self.root, filename)
                    resolved = _inside(self.root, raw_path, label="group source")
                except (UnsafeMoveError, OSError, ValueError) as exc:
                    invalid_source = str(exc)
                    break
                if resolved.parent != self.root or Path(filename).name != filename:
                    invalid_source = "group source must be a direct-child filename"
                    break
                if filename not in candidates:
                    invalid_source = "group source is not an eligible file"
                    break
                if filename in seen_in_group:
                    invalid_source = "group contains a duplicate file"
                    break
                if filename in already_grouped:
                    invalid_source = "file appears in more than one accepted group"
                    break
                seen_in_group.add(filename)
                files.append(filename)

            if invalid_source:
                results.append(
                    GroupEntryResult(
                        folder_name,
                        tuple(files),
                        reason.strip(),
                        "rejected",
                        invalid_source,
                    )
                )
                continue
            if len(files) < MIN_GROUP_SIZE:
                results.append(
                    GroupEntryResult(
                        folder_name,
                        tuple(files),
                        reason.strip(),
                        "discarded",
                        f"cluster must contain at least {MIN_GROUP_SIZE} files",
                    )
                )
                continue

            reserved.add(unicodedata.normalize("NFC", folder_name).casefold())
            already_grouped.update(files)
            results.append(
                GroupEntryResult(
                    folder_name,
                    tuple(files),
                    reason.strip(),
                    "accepted",
                )
            )

        return GroupingResult(tuple(results))

    def validate(self, moves: Sequence[Mapping[str, Any]]) -> ValidatedPlan:
        """Resolve an untrusted proposal into the exact plan a user can approve."""
        if isinstance(moves, (str, bytes)) or not isinstance(moves, Sequence):
            raise TypeError("moves must be a sequence of plan entries")

        entries: list[EntryResult] = []
        validated: list[ValidatedMove] = []
        reserved: set[str] = set()
        seen_sources: set[Path] = set()

        for raw_entry in moves:
            raw_source = str(raw_entry.get("source", "")) if isinstance(raw_entry, Mapping) else ""
            raw_destination = (
                str(raw_entry.get("destination", "")) if isinstance(raw_entry, Mapping) else ""
            )
            raw_reason = str(raw_entry.get("reason", "")) if isinstance(raw_entry, Mapping) else ""
            try:
                move = _validate_move(self.root, raw_entry)
                if move.source in seen_sources:
                    raise UnsafeMoveError("source appears more than once in the plan")
                seen_sources.add(move.source)
                destination = _collision_free(
                    move.destination,
                    reserved,
                    case_sensitive=self.case_sensitive,
                )
                reserved.add(
                    _reservation_key(destination, case_sensitive=self.case_sensitive)
                )
                source_rel = _relative(self.root, move.source)
                destination_rel = _relative(self.root, destination)
                if not move.source.exists():
                    entries.append(
                        EntryResult(
                            source_rel,
                            destination_rel,
                            move.reason,
                            "skipped",
                            "source no longer exists; skipped",
                        )
                    )
                    continue
                metadata = SourceMetadata.capture(move.source)
                source_parent = ParentBinding(
                    _relative(self.root, move.source.parent)
                    if move.source.parent != self.root
                    else ".",
                    DirectoryIdentity.capture(move.source.parent),
                )
                destination_anchor = _nearest_existing_parent(
                    self.root, destination.parent
                )
                destination_missing_parents = _missing_parent_paths(
                    self.root, destination.parent, destination_anchor
                )
            except (UnsafeMoveError, OSError, ValueError, RuntimeError) as exc:
                LOGGER.warning("Rejected move %s -> %s: %s", raw_source, raw_destination, exc)
                entries.append(
                    EntryResult(raw_source, raw_destination, raw_reason, "rejected", str(exc))
                )
                continue

            entries.append(EntryResult(source_rel, destination_rel, move.reason, "planned"))
            validated.append(
                ValidatedMove(
                    source_rel,
                    destination_rel,
                    move.reason,
                    metadata,
                    source_parent,
                    destination_anchor,
                    destination_missing_parents,
                )
            )

        return ValidatedPlan(
            str(self.root), self.root_identity, tuple(validated), tuple(entries)
        )

    def _assert_plan_current(self, plan: ValidatedPlan) -> None:
        """Fail the whole apply before mutation when approved assumptions changed."""
        if Path(plan.root) != self.root:
            raise PlanChangedError("validated plan belongs to a different target root")
        if DirectoryIdentity.capture(self.root) != plan.root_identity:
            raise PlanChangedError("selected root identity changed; build a new plan")
        for move in plan.moves:
            path_move = _validate_move(
                self.root,
                {
                    "source": move.source,
                    "destination": move.destination,
                    "reason": move.reason,
                },
            )
            if not path_move.source.exists() or not path_move.source.is_file():
                raise PlanChangedError(
                    f"approved source changed or disappeared: {move.source}; build a new plan"
                )
            try:
                metadata = SourceMetadata.capture(path_move.source)
            except OSError as exc:
                raise PlanChangedError(
                    f"cannot revalidate approved source {move.source}: {exc}"
                ) from exc
            if metadata != move.source_metadata:
                raise PlanChangedError(
                    f"approved source changed: {move.source}; build a new plan"
                )
            source_parent = DirectoryIdentity.capture(path_move.source.parent)
            if source_parent != move.source_parent.identity:
                raise PlanChangedError(
                    f"approved source parent changed: {move.source_parent.relative_path}"
                )
            anchor_path = (
                self.root
                if move.destination_anchor.relative_path == "."
                else self.root / move.destination_anchor.relative_path
            )
            if (
                anchor_path.is_symlink()
                or DirectoryIdentity.capture(anchor_path)
                != move.destination_anchor.identity
            ):
                raise PlanChangedError(
                    "approved destination parent identity changed; build a new plan"
                )
            for missing_parent in move.destination_missing_parents:
                unexpected_parent = self.root / missing_parent
                if unexpected_parent.exists() or unexpected_parent.is_symlink():
                    raise PlanChangedError(
                        f"destination parent appeared after approval: {missing_parent}; "
                        "build a new plan"
                    )
            if path_move.destination.exists() or path_move.destination.is_symlink():
                raise PlanChangedError(
                    f"approved destination is no longer available: {move.destination}; "
                    "build a new plan"
                )

    def run(
        self,
        moves: Sequence[Mapping[str, Any]] | ValidatedPlan,
        *,
        apply: bool = False,
    ) -> ExecutionResult:
        """Preview or apply a concrete plan; dry-run remains write-free."""
        plan = moves if isinstance(moves, ValidatedPlan) else self.validate(moves)
        if not apply:
            return ExecutionResult(
                root=str(self.root),
                applied=False,
                entries=plan.entries,
                validated_plan=plan,
            )
        return self.execute(plan)

    def execute(self, plan: ValidatedPlan) -> ExecutionResult:
        """Apply exactly *plan* with durable state preceding every mutation."""
        if not _METADATA_IDENTITY_VERIFICATION_RELIABLE:
            raise UnverifiedPlatformError(
                "refusing to mutate: the pre-move identity re-check "
                "(SourceMetadata equality) is not currently verified reliable "
                f"on this platform ({sys.platform}); see "
                "_METADATA_IDENTITY_VERIFICATION_RELIABLE in executor.py"
            )
        # Preserve the no-artifact behavior for an already stale approval. The
        # authoritative revalidation still occurs again while holding the lock.
        self._assert_plan_current(plan)
        with _root_run_lock(self.root):
            self._assert_plan_current(plan)
            now = _utc_now()
            run_id = _run_id(now)
            journal_path = self.journal_dir / f"{run_id}.json"
            operations = [
                {
                    "source": move.source,
                    "destination": move.destination,
                    "reason": move.reason,
                    "source_metadata": asdict(move.source_metadata),
                    "source_parent": asdict(move.source_parent),
                    "destination_anchor": asdict(move.destination_anchor),
                    "destination_missing_parents": list(
                        move.destination_missing_parents
                    ),
                    "destination_parent": None,
                    "moved_metadata": None,
                    "status": "pending",
                    "message": "",
                    "undo_status": "pending",
                }
                for move in plan.moves
            ]
            journal: dict[str, Any] = {
                "schema_version": 3,
                "id": run_id,
                "kind": "run",
                "timestamp": now.isoformat(),
                "root": str(self.root),
                "root_identity": asdict(plan.root_identity),
                "state": "in_progress",
                "operations": operations,
                "moves": [],
                "entries": [asdict(entry) for entry in plan.entries],
                "undone_at": None,
                "undo_moves": [],
            }

            def persist(stage: str) -> None:
                try:
                    _write_json(journal_path, journal)
                except OSError as exc:
                    raise JournalError(
                        f"cannot persist {stage} journal {journal_path}: {exc}; "
                        "execution is not reported as committed"
                    ) from exc

            operation_entries: dict[str, EntryResult] = {}

            def result_entries(*, unfinished: str = "not attempted") -> tuple[EntryResult, ...]:
                return tuple(
                    operation_entries.get(
                        entry.source,
                        EntryResult(
                            entry.source,
                            entry.destination,
                            entry.reason,
                            "skipped",
                            unfinished,
                        ),
                    )
                    if entry.status == "planned"
                    else entry
                    for entry in plan.entries
                )

            # This is intentionally before destination mkdir or the first move.
            persist("initial in-progress")
            failed_entry: EntryResult | None = None
            created_parents: set[str] = set()
            try:
                for operation in operations:
                    source = _inside(
                        self.root,
                        _candidate(self.root, operation["source"]),
                        label="source",
                    )
                    destination = _inside(
                        self.root,
                        _candidate(self.root, operation["destination"]),
                        label="destination",
                    )
                    with _exclusive_file_guard(source) as available:
                        if not available:
                            message = "source is in use or cannot be locked; skipped"
                            operation.update(status="skipped", message=message)
                            entry = EntryResult(
                                operation["source"],
                                operation["destination"],
                                operation["reason"],
                                "skipped",
                                message,
                            )
                            operation_entries[operation["source"]] = entry
                            persist("incremental")
                            continue

                        expected_source = _source_metadata_from_mapping(
                            operation["source_metadata"]
                        )
                        if SourceMetadata.capture(source) != expected_source:
                            raise PlanChangedError(
                                f"approved source changed during execution: "
                                f"{operation['source']}"
                            )
                        if destination.exists() or destination.is_symlink():
                            raise PlanChangedError(
                                f"approved destination changed during execution: "
                                f"{operation['destination']}"
                            )

                        prepared_parent = _prepare_destination_parent(
                            self.root,
                            operation["destination"],
                            root_identity=plan.root_identity,
                            anchor=_binding_from_mapping(
                                operation["destination_anchor"],
                                label="destination anchor",
                            ),
                            expected_missing=operation[
                                "destination_missing_parents"
                            ],
                            created_by_run=created_parents,
                        )
                        operation["destination_parent"] = asdict(prepared_parent)
                        # A persisted 'moving' operation plus the bound parent
                        # identities is enough for deterministic same-device
                        # recovery if the success update never reaches disk.
                        operation["status"] = "moving"
                        persist("pre-move")
                        try:
                            moved_metadata = _move_no_clobber(
                                self.root,
                                operation["source"],
                                operation["destination"],
                                root_identity=plan.root_identity,
                                source_parent=_binding_from_mapping(
                                    operation["source_parent"], label="source parent"
                                ),
                                destination_parent=prepared_parent,
                                expected_source=expected_source,
                            )
                        except Exception as exc:
                            message = f"filesystem changed during execution: {exc}"
                            source_exists = source.exists()
                            destination_exists = destination.exists()
                            moved_observed = False
                            if not source_exists and destination_exists:
                                try:
                                    moved_observed = expected_source.same_moved_object(
                                        SourceMetadata.capture(destination)
                                    )
                                except OSError:
                                    pass
                            definitely_not_moved = (
                                isinstance(exc, FileExistsError)
                                or (
                                    isinstance(exc, OSError)
                                    and exc.errno in {errno.EXDEV, errno.ENOTSUP}
                                )
                                or (source_exists and not destination_exists)
                            )
                            operation.update(
                                status="failed" if definitely_not_moved else "moving",
                                message=message,
                            )
                            failed_entry = EntryResult(
                                operation["source"],
                                operation["destination"],
                                operation["reason"],
                                "moved" if moved_observed else "failed",
                                message,
                            )
                            operation_entries[operation["source"]] = failed_entry
                            persist("incremental failure")
                            raise

                    operation.update(
                        status="moved",
                        message="",
                        moved_metadata=asdict(moved_metadata),
                    )
                    journal["moves"].append(
                        {
                            "source": operation["source"],
                            "destination": operation["destination"],
                            "reason": operation["reason"],
                        }
                    )
                    operation_entries[operation["source"]] = EntryResult(
                        operation["source"],
                        operation["destination"],
                        operation["reason"],
                        "moved",
                    )
                    persist("incremental success")

                final_entries = result_entries(unfinished="not attempted")
                journal["entries"] = [asdict(entry) for entry in final_entries]
                journal["state"] = "commit_pending"
                persist("commit-pending")
                journal["state"] = "committed"
                persist("committed")
            except Exception as exc:
                journal["state"] = "partially_failed"
                partial_entries = result_entries()
                journal["entries"] = [asdict(entry) for entry in partial_entries]
                try:
                    persist("partially-failed")
                except JournalError:
                    pass
                partial_result = ExecutionResult(
                    root=str(self.root),
                    applied=True,
                    entries=partial_entries,
                    run_id=run_id,
                    journal_path=str(journal_path),
                    journal_state="partially_failed",
                    validated_plan=plan,
                )
                if isinstance(exc, JournalError):
                    exc.result = partial_result
                    raise
                if partial_result.moved_count or any(
                    operation.get("status") == "moving" for operation in operations
                ):
                    raise PartialExecutionError(
                        str(exc), result=partial_result, failed_move=failed_entry
                    ) from exc
                raise

            return ExecutionResult(
                root=str(self.root),
                applied=True,
                entries=final_entries,
                run_id=run_id,
                journal_path=str(journal_path),
                journal_state="committed",
                validated_plan=plan,
            )


def _load_journal(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read journal {path.name}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("kind") != "run":
        raise ValueError(f"invalid journal {path.name}")
    return payload


def _select_journal(journal_dir: Path, run_id: str | None) -> tuple[Path, dict[str, Any]]:
    def is_undone(payload: Mapping[str, Any]) -> bool:
        return payload.get("state") == "undone" or bool(payload.get("undone_at"))

    if run_id is not None:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("invalid journal id")
        path = journal_dir / f"{run_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"journal not found: {run_id}")
        payload = _load_journal(path)
        if is_undone(payload):
            raise ValueError(f"run {run_id} has already been undone")
        return path, payload

    for path in sorted(journal_dir.glob("*.json"), reverse=True):
        payload = _load_journal(path)
        if not is_undone(payload):
            return path, payload
    raise FileNotFoundError("no run is available to undo")


def undo(
    run_id: str | None = None,
    *,
    journal_dir: str | Path = DEFAULT_JOURNAL_DIR,
) -> ExecutionResult:
    """Retryably roll back successful moves from one journaled run."""
    if not _METADATA_IDENTITY_VERIFICATION_RELIABLE:
        raise UnverifiedPlatformError(
            "refusing to mutate: the pre-move identity re-check "
            "(SourceMetadata equality) is not currently verified reliable "
            f"on this platform ({sys.platform}); see "
            "_METADATA_IDENTITY_VERIFICATION_RELIABLE in executor.py"
        )
    journal_root = Path(journal_dir).expanduser()
    journal_path, preliminary = _select_journal(journal_root, run_id)
    raw_root = Path(str(preliminary.get("root", "")))
    if raw_root.is_symlink():
        raise PlanChangedError("journal root was replaced by a symlink")
    root = raw_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    with _root_run_lock(raw_root):
        journal = _load_journal(journal_path)
        if journal.get("state") == "undone" or journal.get("undone_at"):
            raise ValueError(f"run {journal.get('id')} has already been undone")
        raw_root_identity = journal.get("root_identity")
        root_identity = (
            DirectoryIdentity(**raw_root_identity)
            if isinstance(raw_root_identity, Mapping)
            else DirectoryIdentity.capture(root)
        )
        if DirectoryIdentity.capture(root) != root_identity:
            raise PlanChangedError("journal root identity changed; undo aborted")
        return _undo_locked(journal_path, journal, root, root_identity)


def _undo_locked(
    journal_path: Path,
    journal: dict[str, Any],
    root: Path,
    root_identity: DirectoryIdentity,
) -> ExecutionResult:
    raw_operations = journal.get("operations")
    if raw_operations is None:
        raw_moves = journal.get("moves")
        if not isinstance(raw_moves, list):
            raise ValueError("journal has no valid moves list")
        raw_operations = [
            {**move, "status": "moved", "undo_status": "pending"}
            for move in raw_moves
            if isinstance(move, Mapping)
        ]
        journal["operations"] = raw_operations
        journal["schema_version"] = 2
    if not isinstance(raw_operations, list) or not all(
        isinstance(operation, dict) for operation in raw_operations
    ):
        raise ValueError("journal has no valid operations list")

    operations: list[dict[str, Any]] = raw_operations
    journal.setdefault("undo_moves", [])
    if not isinstance(journal["undo_moves"], list):
        raise ValueError("journal has no valid undo_moves list")

    def persist(stage: str) -> None:
        try:
            _write_json(journal_path, journal)
        except OSError as exc:
            raise JournalError(
                f"cannot persist {stage} journal {journal_path}: {exc}; "
                "undo is not reported as complete"
            ) from exc

    def source_metadata(operation: Mapping[str, Any]) -> SourceMetadata | None:
        try:
            return _source_metadata_from_mapping(operation.get("source_metadata"))
        except (TypeError, ValueError):
            return None

    def current_metadata(operation: Mapping[str, Any]) -> SourceMetadata | None:
        raw = operation.get("moved_metadata")
        if isinstance(raw, Mapping):
            try:
                return _source_metadata_from_mapping(raw)
            except (TypeError, ValueError):
                return None
        return source_metadata(operation)

    def pre_move_matches(path: Path, operation: Mapping[str, Any]) -> bool:
        expected = source_metadata(operation)
        if expected is None:
            return path.is_file()
        try:
            actual = SourceMetadata.capture(path)
        except OSError:
            return False
        return actual == expected if expected.ctime_ns else expected.same_moved_object(actual)

    def moved_object_matches(path: Path, operation: Mapping[str, Any]) -> bool:
        expected = current_metadata(operation)
        if expected is None:
            return path.is_file()
        try:
            actual = SourceMetadata.capture(path)
        except OSError:
            return False
        if (
            isinstance(operation.get("moved_metadata"), Mapping)
            and "no longer exists" not in str(operation.get("undo_message", ""))
        ):
            return actual == expected
        return expected.same_moved_object(actual)

    # Resolve every possible state left after a crash between rename and the
    # incremental success record. Ambiguous states are retained for manual
    # recovery and are never mutated by Undo.
    for operation in operations:
        if operation.get("status") != "moving":
            continue
        source = _inside(
            root,
            _candidate(root, _validate_text(operation, "source")),
            label="source",
        )
        destination = _inside(
            root,
            _candidate(root, _validate_text(operation, "destination")),
            label="destination",
        )
        source_exists = source.exists()
        destination_exists = destination.exists()
        if source_exists and not destination_exists and pre_move_matches(source, operation):
            operation.update(status="not_moved", message="move did not occur")
        elif (
            not source_exists
            and destination_exists
            and source_metadata(operation) is not None
            and source_metadata(operation).same_moved_object(
                SourceMetadata.capture(destination)
            )
        ):
            operation.update(
                status="moved",
                message="recovered move completed before journal update",
                moved_metadata=asdict(SourceMetadata.capture(destination)),
            )
        else:
            operation.update(
                undo_status="failed",
                undo_message=(
                    "manual recovery required: interrupted move state is ambiguous "
                    "or destination identity mismatches"
                ),
            )

    journal["state"] = "undo_in_progress"
    persist("undo-in-progress")
    entries: list[EntryResult] = []
    case_sensitive = _filesystem_case_sensitive(root)

    for operation in reversed(operations):
        if operation.get("status") != "moved" or operation.get("undo_status") == "restored":
            continue
        try:
            current = _inside(
                root,
                _candidate(root, _validate_text(operation, "destination")),
                label="undo source",
            )
            desired = _inside(
                root,
                _candidate(root, _validate_text(operation, "source")),
                label="undo destination",
            )
            if _has_symlink_component(root, current):
                raise UnsafeMoveError("symlink sources and source paths are never moved")
            if _has_symlink_component(root, desired):
                raise UnsafeMoveError("symlink destination paths are never used")
        except (UnsafeMoveError, OSError, ValueError) as exc:
            operation.update(undo_status="failed", undo_message=str(exc))
            entries.append(
                EntryResult(
                    str(operation.get("destination", "")),
                    str(operation.get("source", "")),
                    "undo",
                    "rejected",
                    str(exc),
                )
            )
            persist("undo failure")
            continue

        current_rel = _relative(root, current)
        target_text = operation.get("undo_target")
        if isinstance(target_text, str) and target_text:
            raw_target = _candidate(root, target_text)
            if _has_symlink_component(root, raw_target):
                operation.update(
                    undo_status="failed",
                    undo_message="saved undo destination became a symlink",
                )
                entries.append(
                    EntryResult(
                        current_rel,
                        target_text,
                        "undo",
                        "rejected",
                        "saved undo destination became a symlink; retry remains available",
                    )
                )
                persist("undo failure")
                continue
            target = _inside(root, raw_target, label="undo destination")
        else:
            target = None
        if (
            operation.get("undo_status") == "undoing"
            and not current.exists()
            and target is not None
            and target.exists()
            and current_metadata(operation) is not None
            and current_metadata(operation).same_moved_object(
                SourceMetadata.capture(target)
            )
        ):
            operation.update(undo_status="restored", undo_message="")
            target_rel = _relative(root, target)
            if not any(
                move.get("source") == current_rel and move.get("destination") == target_rel
                for move in journal["undo_moves"]
                if isinstance(move, Mapping)
            ):
                journal["undo_moves"].append(
                    {"source": current_rel, "destination": target_rel}
                )
            entries.append(EntryResult(current_rel, target_rel, "undo", "restored"))
            persist("recovered undo success")
            continue

        if not current.exists():
            message = "moved file no longer exists; retry remains available"
            operation.update(undo_status="failed", undo_message=message)
            entries.append(
                EntryResult(current_rel, _relative(root, desired), "undo", "skipped", message)
            )
            persist("undo failure")
            continue
        if not current.is_file() or not moved_object_matches(current, operation):
            message = "moved file changed or is not a regular file; retry remains available"
            operation.update(undo_status="failed", undo_message=message)
            entries.append(
                EntryResult(current_rel, _relative(root, desired), "undo", "rejected", message)
            )
            persist("undo failure")
            continue

        destination = (
            target
            if target is not None and not target.exists()
            else _collision_free(desired, set(), case_sensitive=case_sensitive)
        )
        destination_rel = _relative(root, destination)
        expected_current = current_metadata(operation)
        if expected_current is None:
            expected_current = SourceMetadata.capture(current)
        elif "no longer exists" in str(operation.get("undo_message", "")):
            # The same inode may have been temporarily renamed away and back,
            # which legitimately changes ctime. The stable identity check above
            # established that it is still the moved object; bind its current
            # complete metadata for the final mutation-time check.
            expected_current = SourceMetadata.capture(current)
        try:
            source_parent = (
                _binding_from_mapping(
                    operation.get("destination_parent"), label="destination parent"
                )
                if isinstance(operation.get("destination_parent"), Mapping)
                else ParentBinding(
                    _relative(root, current.parent) if current.parent != root else ".",
                    DirectoryIdentity.capture(current.parent),
                )
            )
            destination_anchor = (
                _binding_from_mapping(operation.get("source_parent"), label="source parent")
                if isinstance(operation.get("source_parent"), Mapping)
                else ParentBinding(
                    _relative(root, desired.parent) if desired.parent != root else ".",
                    DirectoryIdentity.capture(desired.parent),
                )
            )
            prepared_parent = _prepare_destination_parent(
                root,
                destination_rel,
                root_identity=root_identity,
                anchor=destination_anchor,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            message = f"filesystem changed during undo; retry remains available: {exc}"
            operation.update(undo_status="failed", undo_message=message)
            entries.append(
                EntryResult(current_rel, destination_rel, "undo", "rejected", message)
            )
            persist("undo failure")
            continue

        operation.update(
            undo_status="undoing",
            undo_target=destination_rel,
            undo_destination_parent=asdict(prepared_parent),
            undo_message="",
        )
        persist("pre-undo")
        with _exclusive_file_guard(current) as available:
            if not available:
                message = "source is in use or cannot be locked; retry remains available"
                operation.update(undo_status="failed", undo_message=message)
                entries.append(
                    EntryResult(current_rel, destination_rel, "undo", "skipped", message)
                )
                persist("undo failure")
                continue
            try:
                restored_metadata = _move_no_clobber(
                    root,
                    current_rel,
                    destination_rel,
                    root_identity=root_identity,
                    source_parent=source_parent,
                    destination_parent=prepared_parent,
                    expected_source=expected_current,
                    mutation_kind="undo",
                )
            except (OSError, RuntimeError, ValueError) as exc:
                message = f"filesystem changed during undo; retry remains available: {exc}"
                current_exists = current.exists()
                destination_exists = destination.exists()
                restored_observed = False
                if not current_exists and destination_exists:
                    try:
                        restored_observed = expected_current.same_moved_object(
                            SourceMetadata.capture(destination)
                        )
                    except OSError:
                        pass
                definitely_not_restored = (
                    isinstance(exc, FileExistsError)
                    or (
                        isinstance(exc, OSError)
                        and exc.errno in {errno.EXDEV, errno.ENOTSUP}
                    )
                    or (current_exists and not destination_exists)
                )
                operation.update(
                    undo_status="failed" if definitely_not_restored else "undoing",
                    undo_message=message,
                )
                entries.append(
                    EntryResult(
                        current_rel,
                        destination_rel,
                        "undo",
                        "restored" if restored_observed else "skipped",
                        message,
                    )
                )
                persist("undo failure")
                continue

        operation.update(
            undo_status="restored",
            undo_message="",
            restored_metadata=asdict(restored_metadata),
        )
        journal["undo_moves"].append(
            {"source": current_rel, "destination": destination_rel}
        )
        entries.append(EntryResult(current_rel, destination_rel, "undo", "restored"))
        persist("incremental undo success")

    unresolved = [
        operation
        for operation in operations
        if operation.get("status") in {"moved", "moving"}
        and operation.get("undo_status") != "restored"
    ]
    now = _utc_now()
    journal["state"] = "partially_undone" if unresolved else "undone"
    journal["undone_at"] = None if unresolved else now.isoformat()
    journal["undo_entries"] = [asdict(entry) for entry in entries]
    persist(journal["state"])
    return ExecutionResult(
        root=str(root),
        applied=True,
        entries=tuple(entries),
        run_id=str(journal.get("id")),
        journal_path=str(journal_path),
        journal_state=str(journal["state"]),
    )
