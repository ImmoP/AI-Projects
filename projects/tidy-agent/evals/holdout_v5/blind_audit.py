"""Blind historical-overlap and lexical-similarity audit for Holdout v5.

Mirrors the frozen ``evals/holdout_v4/blind_audit.py`` (which is pinned and
must not be modified) but targets ``evals/holdout_v5/fixture``. Runs only after
Holdout v5 authoring freeze (``AUTHORING_FROZEN.json`` exists).

Loads filename strings only from the historical corpus (Development and prior
Holdout ``fixture/`` directories) -- never labels, rationales, predictions,
per-file metrics, or reports. Returns and prints aggregate counts only; no
filename, matched pair, or nearest-neighbor information ever leaves this
module, by construction: the public functions return integers, never the
underlying string sets or match lists.

Two independent measures, per ``evals/holdout_v5_protocol.md``:

* Exact historical overlap (Unicode NFC + casefold equality) -- a hard
  exclusion criterion. ``exact_historical_overlap_count`` must equal 0.
* High lexical similarity (token Jaccard >= 0.70, or
  ``difflib.SequenceMatcher`` ratio >= 0.82, or >=80% shorter-filename token
  containment for filename pairs with >=5 normalized tokens on the shorter
  side) -- diagnostic only, reported but never a rejection criterion.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

HISTORICAL_FIXTURE_DIRS = (
    "evals/fixture",
    "evals/holdout/fixture",
    "evals/holdout_v2/fixture",
    "evals/holdout_v3/fixture",
    "evals/holdout_v4/fixture",
    "evals/calibration/fixture",
    "evals/boundary_calibration/fixture",
    "evals/e3_error_calibration/fixture",
    "evals/veto_precision_calibration/fixture",
)

_TOKEN_SPLIT = re.compile(r"[^\w]+", re.UNICODE)


def _normalize(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def load_historical_filenames(repo_root: Path = REPO_ROOT) -> tuple[str, ...]:
    names: list[str] = []
    for rel in HISTORICAL_FIXTURE_DIRS:
        d = repo_root / rel
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.is_file():
                names.append(p.name)
    return tuple(names)


def load_holdout_v5_filenames(repo_root: Path = REPO_ROOT) -> tuple[str, ...]:
    fixture_dir = repo_root / "evals" / "holdout_v5" / "fixture"
    if not fixture_dir.is_dir():
        return ()
    return tuple(p.name for p in fixture_dir.iterdir() if p.is_file())


def exact_overlap_count(historical: tuple[str, ...], holdout: tuple[str, ...]) -> int:
    holdout_normalized = {_normalize(n) for n in holdout}
    return sum(1 for n in historical if _normalize(n) in holdout_normalized)


def _tokens(name: str) -> tuple[str, ...]:
    normalized = _normalize(name)
    return tuple(t for t in _TOKEN_SPLIT.split(normalized) if t)


def _is_near_similar(a: str, b: str) -> bool:
    tokens_a = _tokens(a)
    tokens_b = _tokens(b)
    set_a, set_b = set(tokens_a), set(tokens_b)
    if set_a and set_b:
        union = set_a | set_b
        jaccard = len(set_a & set_b) / len(union) if union else 0.0
        if jaccard >= 0.70:
            return True
    ratio = SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()
    if ratio >= 0.82:
        return True
    shorter, longer = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    if len(shorter) >= 5:
        longer_set = set(longer)
        contained = sum(1 for t in shorter if t in longer_set)
        if contained / len(shorter) >= 0.80:
            return True
    return False


def high_lexical_similarity_count(historical: tuple[str, ...], holdout: tuple[str, ...]) -> int:
    count = 0
    for h in historical:
        for v5 in holdout:
            if _is_near_similar(h, v5):
                count += 1
                break
    return count


def run_blind_audit(repo_root: Path = REPO_ROOT) -> dict[str, int]:
    historical = load_historical_filenames(repo_root)
    holdout = load_holdout_v5_filenames(repo_root)
    return {
        "historical_files_compared": len(historical),
        "holdout_v5_files": len(holdout),
        "exact_historical_overlap_count": exact_overlap_count(historical, holdout),
        "high_lexical_similarity_count": high_lexical_similarity_count(historical, holdout),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_blind_audit(), indent=2))
