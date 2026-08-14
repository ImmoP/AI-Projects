"""Build the synthetic, rerunnable infrastructure-smoke fixture.

This is NOT evaluation evidence. It exists only to give the one-time-eval
runtime (``evals/one_time_eval_runtime.py``) and its smoke runner
(``evals/run_one_time_smoke.py``) a realistic-sized (120-file) batch to
exercise the real request/response/persistence lifecycle against, before a
future independent Holdout is constructed. Filenames are deliberately
generated from an obvious ``synthetic_*`` template -- never adversarial,
never copied from any historical Development or Holdout fixture -- and the
fixture may be regenerated and rerun freely.

Run directly to (re)generate the fixture, manifest, and intended-label file:

    python3 evals/synthetic_one_time_smoke/build_fixture.py
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixture"
MANIFEST_PATH = HERE / "fixture_manifest.json"

REAL_CATEGORIES = ("Documents", "Code", "Images", "Archives", "Installers")
REVIEW_LABEL = "_ToReview"

# Deterministic descriptor words per category -- purely for readable,
# non-adversarial synthetic names. Cycled, not randomized, so the fixture
# is exactly reproducible.
DESCRIPTORS: dict[str, list[str]] = {
    "document": [
        "project_notes", "meeting_summary", "status_memo", "budget_outline",
        "policy_draft", "client_letter", "release_notes", "audit_summary",
        "quarterly_review", "onboarding_guide", "expense_summary", "faq_draft",
        "contract_outline", "training_notes", "survey_summary", "agenda_notes",
        "handbook_draft", "invoice_summary", "proposal_outline", "briefing_notes",
    ],
    "code": [
        "parser_module", "utils_helper", "api_client", "test_harness",
        "config_loader", "build_script", "data_pipeline", "cache_layer",
        "auth_middleware", "queue_worker", "schema_migration", "cli_entrypoint",
        "logging_setup", "retry_policy", "metrics_exporter", "task_scheduler",
        "template_engine", "session_manager", "router_module", "validation_layer",
    ],
    "image": [
        "event_snapshot", "team_photo", "product_shot", "banner_graphic",
        "landscape_capture", "portrait_session", "screenshot_capture", "logo_variant",
        "poster_render", "icon_set", "thumbnail_batch", "wallpaper_variant",
        "diagram_render", "cover_art", "avatar_capture", "scene_render",
        "gallery_item", "print_layout", "sticker_art", "banner_variant",
    ],
    "archive": [
        "backup_bundle", "project_snapshot", "release_bundle", "cold_storage",
        "export_bundle", "migration_snapshot", "weekly_backup", "config_bundle",
        "asset_bundle", "log_bundle", "media_bundle", "dataset_snapshot",
        "workspace_backup", "source_bundle", "state_snapshot", "archive_bundle",
        "monthly_backup", "session_bundle", "checkpoint_bundle", "vault_bundle",
    ],
    "installer": [
        "device_setup", "driver_package", "runtime_installer", "update_package",
        "firmware_setup", "toolchain_installer", "client_setup", "agent_installer",
        "plugin_package", "printer_setup", "codec_package", "sdk_installer",
        "launcher_setup", "service_installer", "utility_package", "bundle_installer",
        "patch_package", "component_setup", "extension_installer", "suite_setup",
    ],
    "review": [
        "unresolved_material", "unlabeled_item", "unspecified_content", "misc_bundle",
        "mixed_content", "unsorted_item", "pending_material", "uncategorized_item",
        "generic_placeholder", "loose_material", "unassigned_item", "leftover_content",
        "unnamed_bundle", "undetermined_item", "assorted_material", "unfiled_item",
        "spare_content", "orphaned_item", "unrouted_material", "catchall_item",
    ],
}

CATEGORY_LABELS = {
    "document": "Documents",
    "code": "Code",
    "image": "Images",
    "archive": "Archives",
    "installer": "Installers",
    "review": REVIEW_LABEL,
}


def build_cases() -> list[dict]:
    cases = []
    for family in ("document", "code", "image", "archive", "installer", "review"):
        descriptors = DESCRIPTORS[family]
        assert len(descriptors) == 20, family
        for i, descriptor in enumerate(descriptors, start=1):
            name = f"synthetic_{family}_case_{i:03d}_{descriptor}"
            name = unicodedata.normalize("NFC", name)
            cases.append({"filename": name, "intended_category": CATEGORY_LABELS[family]})
    return cases


def main() -> None:
    cases = build_cases()
    assert len(cases) == 120
    names = [c["filename"] for c in cases]
    assert len(set(names)) == 120, "duplicate synthetic filenames"

    for name in names:
        assert name == unicodedata.normalize("NFC", name)
        assert "." not in name
        for ch in '<>:"/\\|?*':
            assert ch not in name
        assert not name.endswith(" ") and not name.endswith(".")
        assert 0 < len(name) <= 200

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for existing in FIXTURE_DIR.iterdir():
        if existing.is_file():
            existing.unlink()
    for name in names:
        path = FIXTURE_DIR / name
        path.write_bytes(b"")
        assert path.stat().st_size == 0

    dataset_listing = "\n".join(sorted(names)).encode("utf-8")
    dataset_sha256 = hashlib.sha256(dataset_listing).hexdigest()

    real_count = sum(1 for c in cases if c["intended_category"] != REVIEW_LABEL)
    review_count = sum(1 for c in cases if c["intended_category"] == REVIEW_LABEL)
    per_category = {cat: 0 for cat in REAL_CATEGORIES}
    for c in cases:
        if c["intended_category"] in per_category:
            per_category[c["intended_category"]] += 1

    manifest = {
        "purpose": "infrastructure_smoke_only",
        "evaluation_evidence": False,
        "rerunnable": True,
        "total_files": 120,
        "real_category_count": real_count,
        "review_count": review_count,
        "real_category_counts": per_category,
        "assertions": {
            "zero_byte": True,
            "extensionless": True,
            "nfc_normalized": True,
            "cross_platform_safe": True,
            "collision_free": True,
            "unique_count": 120,
        },
        "dataset_sha256": dataset_sha256,
        "cases": sorted(cases, key=lambda c: c["filename"]),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(names)} synthetic fixture files to {FIXTURE_DIR}")
    print(f"Wrote manifest to {MANIFEST_PATH}")
    print(f"dataset_sha256={dataset_sha256}")


if __name__ == "__main__":
    main()
