"""Generate deterministic cryptographic manifests for dev and holdout fixtures."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.runtime_only_fixture_files import EMPTY_SHA256, RUNTIME_ONLY_FILES  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def freeze(
    *,
    designation: str,
    fixture: Path,
    expected: Path,
    output: Path,
    extra_manifest_fields: dict[str, Any] | None = None,
) -> None:
    ground_truth = yaml.safe_load(expected.read_text(encoding="utf-8")) or {}
    names = ground_truth.get("files")
    if not isinstance(names, dict) or not names:
        raise ValueError(f"{expected} has no files mapping")
    records: list[dict[str, Any]] = []
    for name in sorted(names, key=str.casefold):
        if name in RUNTIME_ONLY_FILES:
            records.append(
                {"path": name, "size": RUNTIME_ONLY_FILES[name], "sha256": EMPTY_SHA256}
            )
            continue
        path = fixture / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {"path": name, "size": path.stat().st_size, "sha256": _sha256(path)}
        )
    canonical_names = set(names)
    extras = sorted(
        path.name for path in fixture.iterdir() if path.is_file() and path.name not in canonical_names
    )
    expected_record = {
        "path": str(expected.relative_to(ROOT)),
        "size": expected.stat().st_size,
        "sha256": _sha256(expected),
    }
    manifest = {
        "designation": designation,
        "files": records,
        "ground_truth": expected_record,
        "source_fixture": str(fixture.relative_to(ROOT)),
        "ambient_files_excluded_from_canonical_fixture": extras,
    }
    # dataset_sha256 is computed over files + ground_truth only, so appending
    # extra_manifest_fields afterward never changes it and stays compatible
    # with every existing caller of ``_verify_dataset_manifest``.
    manifest["dataset_sha256"] = _canonical_digest(
        {"files": records, "ground_truth": expected_record}
    )
    if extra_manifest_fields:
        manifest.update(extra_manifest_fields)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    freeze(
        designation="development benchmark",
        fixture=ROOT / "fixture",
        expected=ROOT / "expected.yaml",
        output=ROOT / "dev/fixture_manifest.json",
    )
    freeze(
        designation="locked holdout — not yet evaluated",
        fixture=ROOT / "holdout/fixture",
        expected=ROOT / "holdout/expected.yaml",
        output=ROOT / "holdout/fixture_manifest.json",
    )
    freeze(
        designation="development-calibration",
        fixture=ROOT / "calibration/fixture",
        expected=ROOT / "calibration/expected.yaml",
        output=ROOT / "calibration/fixture_manifest.json",
    )
    freeze(
        designation="locked holdout v2 — not yet evaluated",
        fixture=ROOT / "holdout_v2/fixture",
        expected=ROOT / "holdout_v2/expected.yaml",
        output=ROOT / "holdout_v2/fixture_manifest.json",
        extra_manifest_fields={
            "created_date": "2026-08-12",
            "production_candidate_commit": "948fc6c85b5e8f1c58598d9ffaa6c59a33a8a8a1",
            "production_candidate_commit_message": (
                "tidy: integrate explicit abstention agreement gate"
            ),
            "model_inference_performed": False,
            "construction_note": (
                "Cases were independently authored without opening or "
                "inspecting evals/holdout/ or evals/calibration/ case-level "
                "content. Ground truth was assigned before any model "
                "inference; none has occurred as of this freeze."
            ),
        },
    )
    freeze(
        designation="post-holdout-v2 development boundary-calibration",
        fixture=ROOT / "boundary_calibration/fixture",
        expected=ROOT / "boundary_calibration/expected.yaml",
        output=ROOT / "boundary_calibration/fixture_manifest.json",
        extra_manifest_fields={
            "created_date": "2026-08-12",
            "model_inference_performed": False,
            "construction_note": (
                "Cases were freshly authored for the post-Holdout-v2 E4/E5 "
                "Development cycle without inspecting evals/holdout/ or "
                "evals/holdout_v2/ case-level content; no filename or label "
                "here duplicates one from either. This is a development "
                "fixture, not a Holdout -- it may be inspected, rerun, and "
                "tuned against repeatedly."
            ),
        },
    )
    freeze(
        designation="veto-precision development stress fixture",
        fixture=ROOT / "veto_precision_calibration/fixture",
        expected=ROOT / "veto_precision_calibration/expected.yaml",
        output=ROOT / "veto_precision_calibration/fixture_manifest.json",
        extra_manifest_fields={
            "created_date": "2026-08-12",
            "model_inference_performed": False,
            "construction_note": (
                "Cases were freshly authored after the previous live "
                "E3/E4/E5 Development result, using only its aggregate "
                "findings (never Holdout case-level data), specifically to "
                "stress-test and tune E4-refined veto precision. Not "
                "independent validation -- a post-result Development stress "
                "fixture, reusable for tuning."
            ),
        },
    )
    freeze(
        designation="E3 automatic-error development stress fixture",
        fixture=ROOT / "e3_error_calibration/fixture",
        expected=ROOT / "e3_error_calibration/expected.yaml",
        output=ROOT / "e3_error_calibration/fixture_manifest.json",
        extra_manifest_fields={
            "created_date": "2026-08-12",
            "model_inference_performed": False,
            "construction_note": (
                "Cases were freshly authored after the corrected 2026-08-12 "
                "E4-precision Development cycle, using only its aggregate "
                "finding (1 unique E3 automatic error across 185 files, "
                "below the frozen minimum of 3 -- never Holdout case-level "
                "data), specifically to increase the chance of observing "
                "enough genuinely difficult E3 automatic decisions to "
                "evaluate that frozen threshold. Not independent validation "
                "-- a post-result Development stress fixture, reusable for "
                "tuning. No filename, wording, or label here duplicates one "
                "from evals/calibration, evals/boundary_calibration, or "
                "evals/veto_precision_calibration."
            ),
        },
    )


if __name__ == "__main__":
    main()
