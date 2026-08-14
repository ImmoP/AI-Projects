"""Registry of dev-fixture cases that cannot be committed to git.

``report:final`` (a real, empty ground-truth case -- see
``evals/expected.yaml``) contains a colon, which is illegal in NTFS
filenames and is read as an Alternate Data Stream separator on Windows. The
path cannot be checked out on ``windows-latest`` runners and is therefore
not tracked in git.

Its content never varies (it is always empty), so ``evals/freeze_datasets.py``
can compute a manifest entry for it without reading it from disk. Callers
that need a complete, on-disk fixture directory (see
``tests/test_abcd.py::test_committed_dev_manifest_matches_every_canonical_file``)
materialize it themselves, skipped on Windows where that is not possible.
"""
from __future__ import annotations

import hashlib

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

# name -> size in bytes. Every entry here is currently empty; if a future
# entry needs non-empty content, extend this to store the content itself
# instead of just a size.
RUNTIME_ONLY_FILES: dict[str, int] = {
    "report:final": 0,
}
