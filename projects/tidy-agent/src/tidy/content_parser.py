"""Resource-bounded extraction for complex, attacker-controlled documents."""

from __future__ import annotations

import math
import multiprocessing
import queue
import time
import zipfile
from io import BytesIO
from typing import Any, Callable

MAX_PDF_BYTES = 8 * 1024 * 1024
MAX_DOCX_BYTES = 8 * 1024 * 1024
PARSER_TIMEOUT_SECONDS = 3.0
PARSER_MEMORY_BYTES = 512 * 1024 * 1024
PARSER_CPU_SECONDS = 3
MAX_DOCX_ENTRIES = 2_000
MAX_DOCX_ENTRY_BYTES = 8 * 1024 * 1024
MAX_DOCX_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
MAX_DOCX_COMPRESSION_RATIO = 100.0


class UnsafeDocumentError(Exception):
    """Raised when a container violates a deterministic resource limit."""


def validate_docx_archive(data: bytes) -> None:
    """Reject obvious ZIP bombs before python-docx sees the container."""
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise UnsafeDocumentError("malformed_docx") from exc

    if len(entries) > MAX_DOCX_ENTRIES:
        raise UnsafeDocumentError("docx_entry_limit")
    total = 0
    for entry in entries:
        if entry.flag_bits & 0x1:
            raise UnsafeDocumentError("encrypted_docx_entry")
        if entry.file_size > MAX_DOCX_ENTRY_BYTES:
            raise UnsafeDocumentError("docx_entry_size_limit")
        total += entry.file_size
        if total > MAX_DOCX_UNCOMPRESSED_BYTES:
            raise UnsafeDocumentError("docx_uncompressed_size_limit")
        if entry.file_size and entry.file_size / max(1, entry.compress_size) > (
            MAX_DOCX_COMPRESSION_RATIO
        ):
            raise UnsafeDocumentError("docx_compression_ratio_limit")


def _apply_posix_resource_limits(timeout: float) -> None:
    """Apply best-effort POSIX limits; process termination remains portable."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return
    try:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (PARSER_MEMORY_BYTES, PARSER_MEMORY_BYTES),
        )
    except (OSError, ValueError):
        # Some macOS configurations do not support a useful RLIMIT_AS. The hard
        # input/output limits and parent-enforced timeout still apply.
        pass
    try:
        cpu_seconds = max(PARSER_CPU_SECONDS, math.ceil(timeout))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    except (OSError, ValueError):
        pass


def _document_parser_worker(
    result_queue: multiprocessing.Queue,
    extension: str,
    data: bytes,
    max_chars: int,
    timeout: float,
) -> None:
    """Parse one already-size-bounded byte string in an isolated process."""
    _apply_posix_resource_limits(timeout)
    try:
        if extension == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(data))
            chunks: list[str] = []
            pages_read = 0
            for page in reader.pages[:2]:
                chunks.append(page.extract_text() or "")
                pages_read += 1
                if sum(map(len, chunks)) + max(0, len(chunks) - 1) > max_chars:
                    break
            text = "\n".join(chunks)
            result_queue.put(
                {
                    "status": "ok",
                    "text": text[:max_chars],
                    "truncated": len(text) > max_chars or len(reader.pages) > pages_read,
                    "pages_read": pages_read,
                }
            )
            return

        if extension == ".docx":
            try:
                validate_docx_archive(data)
            except UnsafeDocumentError as exc:
                result_queue.put({"status": "skipped", "reason": str(exc)})
                return
            from docx import Document

            document = Document(BytesIO(data))
            chunks: list[str] = []
            length = 0
            truncated = False
            for paragraph in document.paragraphs:
                separator = 1 if chunks else 0
                chunks.append(paragraph.text)
                length += separator + len(paragraph.text)
                if length > max_chars:
                    truncated = True
                    break
            text = "\n".join(chunks)
            result_queue.put(
                {
                    "status": "ok",
                    "text": text[:max_chars],
                    "truncated": truncated or len(text) > max_chars,
                    "pages_read": None,
                }
            )
            return
        result_queue.put({"status": "error", "error_type": "unsupported_format"})
    except BaseException as exc:
        # Only the exception class crosses the process boundary. Parser messages
        # and tracebacks can contain snippets from attacker-controlled input.
        result_queue.put(
            {"status": "error", "error_type": type(exc).__name__}
        )


def run_parser_subprocess(
    worker: Callable[..., None],
    args: tuple[Any, ...],
    *,
    timeout: float = PARSER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run and, on timeout, actually terminate a parser process."""
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=worker, args=(result_queue, *args))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(1)
        if process.is_alive():  # pragma: no cover - defensive OS fallback
            process.kill()
            process.join(1)
        result_queue.close()
        return {"status": "timeout"}
    try:
        result = result_queue.get(timeout=1)
    except queue.Empty:
        result = {"status": "error", "error_type": "parser_process_failed"}
    finally:
        result_queue.close()
    return result


def parse_document_bytes(
    extension: str,
    data: bytes,
    max_chars: int,
    *,
    timeout: float = PARSER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Validate a container and return a small structured extraction result."""
    return run_parser_subprocess(
        _document_parser_worker,
        (extension, data, max_chars, timeout),
        timeout=timeout,
    )


def delayed_test_worker(result_queue: multiprocessing.Queue, delay: float) -> None:
    """Deterministic public worker used by the timeout regression test."""
    time.sleep(delay)
    result_queue.put({"status": "ok"})
