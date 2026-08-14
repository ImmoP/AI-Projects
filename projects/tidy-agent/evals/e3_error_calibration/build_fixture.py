"""Materialize the E3 automatic-error Development stress fixture
(metadata-only, empty files).

This is a **post-result Development stress fixture**, not independent
validation and not a Holdout: it was authored after the corrected
2026-08-12 E4-precision Development cycle (aggregate finding only -- that
cycle observed just 1 unique E3 automatic error across 185 Development
files, well below the frozen minimum of 3 required to interpret veto
precision/recall robustly -- never any Holdout case-level data) to increase
the chance of observing enough genuinely difficult E3 automatic decisions
for that frozen threshold to eventually be evaluated. It may be inspected,
rerun, and reused repeatedly, exactly like ``evals/calibration``,
``evals/boundary_calibration``, and ``evals/veto_precision_calibration``.
Constructing this fixture does not change, tune, or re-evaluate E3,
E4-current, or E4-refined in any way -- no model inference of any kind has
occurred as part of building it.

Unlike ``evals/veto_precision_calibration`` (which asked "can E4-refined
avoid false vetoes on correct E3 decisions?"), this fixture asks a
different, complementary question: "when metadata is semantically
deceptive but still plausible enough for E3 to commit to an automatic
decision, does E3 make correlated wrong automatic decisions -- and can the
already-frozen veto strategies catch them?" It deliberately avoids explicit
uncertainty markers ("or"/"oder"/"unclear"/"unklar"/"unknown"/"maybe"/
"unbekannt"/"unbestimmt" and equivalents), which encourage abstention
rather than confident misclassification and so do not exercise this
failure mode.

No filename, wording choice, or label here was copied or reconstructed
from ``evals/holdout``, ``evals/holdout_v2``, ``evals/calibration``,
``evals/boundary_calibration``, or ``evals/veto_precision_calibration``;
every case was freshly authored for this fixture, targeting GENERIC
semantic failure families identified from aggregate prior Development
evidence rather than any specific previously observed filename.

Every file is extensionless, so none can be resolved by the deterministic
extension rules in ``config/rules.yaml`` and none can be excluded by its
glob patterns -- all 72 cases are guaranteed, by construction, to reach
unresolved-file classification (E3/E4-current/E4-refined). Files are 0
bytes: this cycle stays metadata-only, so content is never read.

Composition (72 total): 48 real-category cases (66.7%), 24 ``_ToReview``
cases (33.3%).

Real-category distribution: Documents 10, Code 10, Images 10, Archives 9,
Installers 9.

Real-category stress families (a file's PRIMARY family; ``SECONDARY_TAGS``
below records optional secondary tags such as ``multilingual`` and
``compound_morphology`` -- neither the family nor the tags are ever sent
to a model):

* **subject_vs_artifact (16)** -- the filename carries a strong cue for
  category A because category A is the file's SUBJECT, while the actual
  artifact is category B (what the file itself IS, not what it is about).
* **tool_vs_output (10)** -- an action/tool cue points toward the category
  the tool operates ON, but the file is actually the tool's own
  implementation or documentation.
* **container_lexical_trap (8)** -- package/bundle/backup/collection/
  release/distribution vocabulary, deliberately NOT paired with Archives
  ground truth, so the vocabulary itself cannot be trusted as a shortcut.
* **installer_driver_trap (8)** -- driver/setup/installation/deployment/
  firmware vocabulary, deliberately NOT paired with Installers ground
  truth, for the same reason.
* **media_document_trap (6)** -- media vocabulary paired with a
  non-Images artifact, plus (to avoid a one-directional trap) some
  genuine Images cases that carry documentation/workflow vocabulary.

``_ToReview`` stress families (24 total, no explicit ambiguity markers):

* **latent_dual_role (10)** -- two plausible artifact types are present
  but grammar/context never establishes which one the file actually is;
  no "A or B" wording.
* **latent_container_content (6)** -- metadata cannot establish whether
  the file itself is a container/archive or something that merely
  describes/represents the container's contents.
* **dominant_cue_ambiguity (8)** -- one category has a strong lexical cue,
  but the file-role relation needed to commit to it is absent, so a
  different interpretation remains equally legitimate; E3 may confidently
  pick the dominant cue and thereby produce an automatic error.

Ground truth was assigned before any inference (none has occurred: this is
implementation- and fixture-construction-only), using the same principle
as every prior Development/Holdout fixture in this project: if the
permitted metadata is sufficient for a human evaluator to safely choose
one production category, assign that category; otherwise ``_ToReview``.
Cases were never labelled by asking what E3, E4-current, or E4-refined
would predict, and no filename was adjusted after any (nonexistent)
prediction.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parent
FIXTURE_ROOT = ROOT / "fixture"
EXPECTED_PATH = ROOT / "expected.yaml"

REVIEW_DIRECTORY = "_ToReview"

# (filename, ground_truth_category_or_review, evaluator-only rationale,
#  primary_family, secondary_tags) -- rationale/family/tags are NEVER sent
# to a model; only the filename itself is scanned as fixture metadata.
CASES: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
    # === Documents (10): 2 subject_vs_artifact, 2 tool_vs_output, ==========
    # === 2 container_lexical_trap, 3 installer_driver_trap, 1 media_document_trap
    (
        "charity_gala_photo_retrospective_writeup",
        "Documents",
        "A written retrospective piece about a charity gala's photography; "
        "the artifact is the writeup itself, photography is only its topic.",
        "subject_vs_artifact",
        (),
    ),
    (
        "περιγραφή_αρχιτεκτονικής_διαδικασίας_δημιουργίας",
        "Documents",
        "Greek: a description of a build process's architecture; the "
        "artifact is the written description, the build/code is only its "
        "subject.",
        "subject_vs_artifact",
        ("multilingual",),
    ),
    (
        "duplicate_finder_utility_user_guide",
        "Documents",
        "A user guide explaining how to use a duplicate-file finder; the "
        "guide is documentation, not the utility's source or the utility "
        "itself.",
        "tool_vs_output",
        (),
    ),
    (
        "release_notities_maphelper_toepassing",
        "Documents",
        "Dutch: release notes for a folder-helper application; the notes "
        "are a Documents artifact describing the tool's changes, not code.",
        "tool_vs_output",
        ("multilingual",),
    ),
    (
        "sicherungsrotation_richtlinie_memo",
        "Documents",
        "German: an internal policy memo explaining a backup-rotation "
        "schedule; the artifact is the memo, 'sicherung' (backup) names "
        "the memo's subject, not the file's own type.",
        "container_lexical_trap",
        ("multilingual", "compound_morphology"),
    ),
    (
        "acuerdo_derechos_publicacion_coleccion_fotografica",
        "Documents",
        "Spanish: a publishing-rights agreement covering a photo "
        "collection; the artifact is the agreement text, 'coleccion' "
        "(collection) names what the agreement is about.",
        "container_lexical_trap",
        ("multilingual",),
    ),
    (
        "deployment_rollback_procedure_writeup",
        "Documents",
        "A written procedure explaining how to roll back a deployment; "
        "the artifact is the procedure document, not the deployed "
        "software itself.",
        "installer_driver_trap",
        (),
    ),
    (
        "arbeitsplatz_einrichtung_checkliste_aushang",
        "Documents",
        "German: a posted checklist for setting up a new workstation; "
        "'einrichtung' (setup) describes the checklist's topic, the "
        "artifact is the printed checklist itself.",
        "installer_driver_trap",
        ("multilingual", "compound_morphology"),
    ),
    (
        "plan_pruebas_certificacion_controlador_cumplimiento",
        "Documents",
        "Spanish: a compliance test plan for driver certification; the "
        "artifact is the written test plan, 'controlador' (driver) is "
        "only its subject matter.",
        "installer_driver_trap",
        ("multilingual",),
    ),
    (
        "familientreffen_diashow_sprechertext",
        "Documents",
        "German: the speaker's script written for a family-reunion "
        "slideshow narration; the artifact is the written script, not "
        "the slideshow images themselves.",
        "media_document_trap",
        ("multilingual", "compound_morphology"),
    ),
    # === Code (10): 1 subject_vs_artifact, 5 tool_vs_output, ================
    # === 2 container_lexical_trap, 2 installer_driver_trap, 0 media_document_trap
    (
        "employee_onboarding_checklist_app_sourcecode",
        "Code",
        "The source code implementing an onboarding-checklist app; "
        "'checklist' evokes a Documents topic, but the file is the app's "
        "own implementation.",
        "subject_vs_artifact",
        (),
    ),
    (
        "batch_photo_resize_tool_sourcecode",
        "Code",
        "Source code implementing a batch photo-resizing tool; the tool "
        "operates on images, but the file is the tool's own "
        "implementation, not an image.",
        "tool_vs_output",
        (),
    ),
    (
        "folder_deduplication_scanner_implementation",
        "Code",
        "Implementation source for a folder de-duplication scanner; a "
        "straightforward tool implementation.",
        "tool_vs_output",
        (),
    ),
    (
        "nachtsicherung_planungsdienst_quellcode",
        "Code",
        "German: source code for a nightly backup-scheduling service; "
        "'sicherung' (backup) points toward Archives, but the file is the "
        "scheduler's own source code.",
        "tool_vs_output",
        ("multilingual", "compound_morphology"),
    ),
    (
        "installationsassistent_schrittsequenzer_quellcode",
        "Code",
        "German: source code for an installation-wizard step sequencer; "
        "'installation' points toward Installers, but the file is the "
        "wizard-building source itself.",
        "tool_vs_output",
        ("multilingual", "compound_morphology"),
    ),
    (
        "codigo_fonte_pipeline_exportacao_documentos",
        "Code",
        "Portuguese: source code for a document-export pipeline; "
        "'documentos' points toward Documents, but the file is the "
        "pipeline's own source.",
        "tool_vs_output",
        ("multilingual",),
    ),
    (
        "release_tagging_automation_script",
        "Code",
        "An automation script that tags releases in a CI pipeline; "
        "'release' evokes a packaged distribution, but the file is a "
        "script (Code), not the release artifact.",
        "container_lexical_trap",
        (),
    ),
    (
        "utility_unione_raccolte_playlist_sorgente",
        "Code",
        "Italian: source for a utility that merges playlist collections; "
        "'raccolte' (collections) points toward Archives, but the file is "
        "the merging utility's own source.",
        "container_lexical_trap",
        ("multilingual",),
    ),
    (
        "driver_compatibility_regression_test_harness",
        "Code",
        "A regression test harness for driver compatibility; 'driver' "
        "points toward Installers, but the harness itself is test code.",
        "installer_driver_trap",
        (),
    ),
    (
        "installatieflow_integratietest_suite_afrekenmodule",
        "Code",
        "Dutch: an integration-test suite covering a checkout module's "
        "setup flow; 'installatie' (setup) points toward Installers, but "
        "the suite itself is test code.",
        "installer_driver_trap",
        ("multilingual", "compound_morphology"),
    ),
    # === Images (10): 4 subject_vs_artifact, 0 tool_vs_output, ==============
    # === 2 container_lexical_trap, 1 installer_driver_trap, 3 media_document_trap
    (
        "workbench_printed_installer_manual_photo",
        "Images",
        "A photograph of a printed installer manual lying on a workbench; "
        "the artifact is the photo, the manual is only the photographed "
        "subject.",
        "subject_vs_artifact",
        (),
    ),
    (
        "whiteboard_code_architecture_sketch_snapshot",
        "Images",
        "A snapshot of a whiteboard sketch showing code architecture; the "
        "artifact is the snapshot photo, not the code it depicts.",
        "subject_vs_artifact",
        (),
    ),
    (
        "archivspende_vereinbarung_unterschrift_aufnahme",
        "Images",
        "German: a photo of the signature page on an archive-donation "
        "agreement; the artifact is the photo, the agreement is only its "
        "subject.",
        "subject_vs_artifact",
        ("multilingual", "compound_morphology"),
    ),
    (
        "foto_carpeta_copia_seguridad_descomprimida_monitor",
        "Images",
        "Spanish: a photo of a decompressed backup folder shown on a "
        "monitor; the artifact is the photo, the backup folder is only "
        "what's pictured.",
        "subject_vs_artifact",
        ("multilingual",),
    ),
    (
        "verkaufsstart_veranstaltungsfotografie_rohmaterial",
        "Images",
        "German: raw, unedited event photography from a product launch "
        "day; 'verkaufsstart' (launch/release) evokes a packaged "
        "distribution, but the files are the raw photographs themselves.",
        "container_lexical_trap",
        ("multilingual", "compound_morphology"),
    ),
    (
        "portrait_groupe_journee_collecte_fondation_annuelle",
        "Images",
        "French: a group portrait from an annual fundraiser's collection "
        "day; 'collecte' (fundraising collection) evokes Archives, but "
        "the artifact is a portrait photograph.",
        "container_lexical_trap",
        ("multilingual",),
    ),
    (
        "firmware_flashing_progress_bar_screenshot",
        "Images",
        "A screenshot capturing a firmware-flashing progress bar; "
        "'firmware flashing' evokes Installers, but the artifact is a "
        "screenshot image.",
        "installer_driver_trap",
        (),
    ),
    (
        "werkstroom_documentatie_stilstaand_beeld_export",
        "Images",
        "Dutch: a still image exported from workflow-documentation "
        "training footage; 'documentatie' evokes Documents, but the "
        "exported artifact is a still image.",
        "media_document_trap",
        ("multilingual",),
    ),
    (
        "protokollstil_bildunterschriften_infografik",
        "Images",
        "German: an infographic laid out like meeting-minutes captions; "
        "'protokoll' (minutes/log) evokes Documents, but the artifact is "
        "a rendered infographic image.",
        "media_document_trap",
        ("multilingual", "compound_morphology"),
    ),
    (
        "arkusz_specyfikacji_wizualny_wykres_porownawczy",
        "Images",
        "Polish: a specification sheet rendered visually as a comparison "
        "chart; 'specyfikacji' (specification) evokes Documents, but the "
        "artifact is the rendered chart image.",
        "media_document_trap",
        ("multilingual",),
    ),
    # === Archives (9): 5 subject_vs_artifact, 1 tool_vs_output, =============
    # === 0 container_lexical_trap, 2 installer_driver_trap, 1 media_document_trap
    (
        "ten_year_photo_portfolio_compressed_package",
        "Archives",
        "A compressed package bundling a decade of photography; the "
        "artifact is the compressed package (Archives), photography is "
        "only its subject.",
        "subject_vs_artifact",
        (),
    ),
    (
        "quellcode_schnappschuesse_vor_refaktorierung_gepackt",
        "Archives",
        "German: packed source-code snapshots taken before a refactor; "
        "the artifact is the packed bundle, the code is only its "
        "contents' subject.",
        "subject_vs_artifact",
        ("multilingual",),
    ),
    (
        "coleccion_comprimida_contratos_firmados_ejercicio_fiscal",
        "Archives",
        "Spanish: a compressed collection of signed contracts for a "
        "fiscal year; the artifact is the compressed collection, "
        "contracts are only its subject.",
        "subject_vs_artifact",
        ("multilingual",),
    ),
    (
        "veraltete_installationsbinaerdateien_tarball",
        "Archives",
        "German: a tarball containing outdated installer binaries; the "
        "artifact is the tarball (Archives), the installer binaries are "
        "only its contents' subject.",
        "subject_vs_artifact",
        ("multilingual", "compound_morphology"),
    ),
    (
        "esportazione_compressa_galleria_immagini_catalogo",
        "Archives",
        "Italian: a compressed export of a catalog's image gallery; the "
        "artifact is the compressed export, the images are only its "
        "subject.",
        "subject_vs_artifact",
        ("multilingual",),
    ),
    (
        "internal_build_automation_toolkit_source_package",
        "Archives",
        "A packed source distribution of an internal build-automation "
        "toolkit; the toolkit is code, but the file itself is a packed "
        "distribution (Archives), not raw source.",
        "tool_vs_output",
        (),
    ),
    (
        "gepackte_sammlung_veralteter_druckertreiber_referenz",
        "Archives",
        "German: a packed reference collection of outdated printer "
        "drivers; 'treiber' (driver) points toward Installers, but the "
        "artifact genuinely is a packed archive of driver files.",
        "installer_driver_trap",
        ("multilingual", "compound_morphology"),
    ),
    (
        "instantaneo_comprimido_ambiente_implantacao_desativado",
        "Archives",
        "Portuguese: a compressed snapshot of a decommissioned deployment "
        "environment; 'implantacao' (deployment) points toward "
        "Installers, but the artifact is a compressed environment "
        "snapshot.",
        "installer_driver_trap",
        ("multilingual",),
    ),
    (
        "verpacktes_filmmaterial_export_kaltlagerung",
        "Archives",
        "German: a packed footage export bundled for cold storage; "
        "'filmmaterial' (footage) evokes Images, but the artifact is the "
        "packed bundle itself.",
        "media_document_trap",
        ("multilingual", "compound_morphology"),
    ),
    # === Installers (9): 4 subject_vs_artifact, 2 tool_vs_output, ===========
    # === 2 container_lexical_trap, 0 installer_driver_trap, 1 media_document_trap
    (
        "photo_tagging_desktop_app_setup_package",
        "Installers",
        "A setup package for a photo-tagging desktop application; the "
        "artifact is the installer (Installers), photo-tagging is only "
        "the app's subject.",
        "subject_vs_artifact",
        (),
    ),
    (
        "leichtgewichtiger_quellcode_verwaltung_client_installer",
        "Installers",
        "German: an installer for a lightweight source-control client; "
        "'quellcode' (source code) points toward Code, but the artifact "
        "is the client's own installer.",
        "subject_vs_artifact",
        ("multilingual", "compound_morphology"),
    ),
    (
        "paquet_deploiement_utilitaire_signature_documents",
        "Installers",
        "French: a deployment package for a document-signing utility; "
        "'documents' points toward Documents, but the artifact is the "
        "utility's own deployment package.",
        "subject_vs_artifact",
        ("multilingual",),
    ),
    (
        "installatiebestand_archiefverkenner_companion_app",
        "Installers",
        "Dutch: an installer file for an archive-browsing companion app; "
        "'archief' (archive) points toward Archives, but the artifact is "
        "the companion app's installer.",
        "subject_vs_artifact",
        ("multilingual",),
    ),
    (
        "sicherungsplanung_menueleisten_dienstprogramm_setup",
        "Installers",
        "German: the packaged setup for a backup-scheduling menu-bar "
        "utility; the utility performs backups, but the file is the "
        "utility's own setup package.",
        "tool_vs_output",
        ("multilingual", "compound_morphology"),
    ),
    (
        "instalator_dodatku_automatycznego_generowania_raportow",
        "Installers",
        "Polish: an installer build for an automatic report-generation "
        "add-in; 'raportow' (reports) points toward Documents, but the "
        "artifact is the add-in's own installer.",
        "tool_vs_output",
        ("multilingual",),
    ),
    (
        "verteilungspaket_schriftartenverwaltung_dienstprogramm",
        "Installers",
        "German: a distribution package for a font-management utility; "
        "'verteilungspaket' (distribution package) evokes Archives, but "
        "a ready-to-run setup package is conventionally Installers.",
        "container_lexical_trap",
        ("multilingual", "compound_morphology"),
    ),
    (
        "instalador_paquete_lanzamiento_historial_portapapeles",
        "Installers",
        "Spanish: a release-package installer for a clipboard-history "
        "helper; 'paquete'/'lanzamiento' (package/release) evoke "
        "Archives, but the ready-to-run setup is Installers.",
        "container_lexical_trap",
        ("multilingual",),
    ),
    (
        "einrichtungsassistent_bildschirmfoto_anmerkungswerkzeug",
        "Installers",
        "German: a setup wizard for a screenshot-annotation toolkit; "
        "'bildschirmfoto' (screenshot) evokes Images, but the artifact is "
        "the toolkit's own setup wizard.",
        "media_document_trap",
        ("multilingual", "compound_morphology"),
    ),
    # === _ToReview: latent_dual_role (10) ===================================
    # Two plausible artifact types, never announced with "or"/explicit
    # uncertainty wording -- grammar/context alone doesn't establish which.
    (
        "quarterly_review_slide_deck_export",
        REVIEW_DIRECTORY,
        "Documents vs Images: could be an exported deck treated as a "
        "document, or a rendered image export of the slides; the export "
        "format is never established.",
        "latent_dual_role",
        (),
    ),
    (
        "system_design_notes_repository",
        REVIEW_DIRECTORY,
        "Documents vs Code: could be written design notes, or a "
        "specification-as-code repository referred to informally as "
        "'notes'; genuinely unresolved.",
        "latent_dual_role",
        (),
    ),
    (
        "project_wrapup_package_summary",
        REVIEW_DIRECTORY,
        "Documents vs Archives: could be a packaged/archived project "
        "bundle, or a written summary document about the wrap-up "
        "package; unresolved.",
        "latent_dual_role",
        (),
    ),
    (
        "onboarding_kit_reference",
        REVIEW_DIRECTORY,
        "Documents vs Installers: could be a written reference guide, or "
        "the onboarding kit itself as an installable software bundle; "
        "unresolved.",
        "latent_dual_role",
        (),
    ),
    (
        "projektquellen_sammelablage",
        REVIEW_DIRECTORY,
        "German: Code vs Archives -- could be a reference to a project "
        "source-code directory, or a packed collective storage bundle of "
        "the sources; unresolved.",
        "latent_dual_role",
        ("multilingual", "compound_morphology"),
    ),
    (
        "toepassing_buildresultaat",
        REVIEW_DIRECTORY,
        "Dutch: Code vs Installers -- could refer to the application's "
        "source/build output as code, or to the build result being the "
        "installer artifact itself; unresolved.",
        "latent_dual_role",
        ("multilingual",),
    ),
    (
        "recuerdos_viaje_respaldo",
        REVIEW_DIRECTORY,
        "Spanish: Images vs Archives -- could be the travel-memory photos "
        "themselves, or a backup bundle of them; unresolved.",
        "latent_dual_role",
        ("multilingual",),
    ),
    (
        "interfaccia_anteprima_pacchetto",
        REVIEW_DIRECTORY,
        "Italian: Images vs Installers -- could be a preview screenshot "
        "of an interface, or the packaged application named after its "
        "own interface preview; unresolved.",
        "latent_dual_role",
        ("multilingual",),
    ),
    (
        "pacote_distribuicao_legado",
        REVIEW_DIRECTORY,
        "Portuguese: Archives vs Installers -- 'legacy distribution "
        "package' is equally plausible as a compressed legacy archive or "
        "a legacy installer package; unresolved.",
        "latent_dual_role",
        ("multilingual",),
    ),
    (
        "prototyp_visualisierung_quellmaterial",
        REVIEW_DIRECTORY,
        "German: Code vs Images -- could be the prototype's source "
        "material (Code), or its rendered visualizations (Images); "
        "unresolved.",
        "latent_dual_role",
        ("multilingual", "compound_morphology"),
    ),
    # === _ToReview: latent_container_content (6) ============================
    # Cannot establish whether the file itself IS a container/archive, or
    # is something that merely represents/describes the contained material.
    (
        "media_holdings_overview",
        REVIEW_DIRECTORY,
        "Could itself be an archive/container of media holdings, or an "
        "overview document describing them; the file's own type is never "
        "established.",
        "latent_container_content",
        (),
    ),
    (
        "projektbestand_uebertragung",
        REVIEW_DIRECTORY,
        "German: could be the transferred container itself (an archived "
        "project holding), or a transfer record/note about it; "
        "unresolved.",
        "latent_container_content",
        ("multilingual", "compound_morphology"),
    ),
    (
        "contenido_empaquetado_referencia",
        REVIEW_DIRECTORY,
        "Spanish: could be the packaged content container itself, or a "
        "reference document pointing at packaged content; unresolved.",
        "latent_container_content",
        ("multilingual",),
    ),
    (
        "technische_onderdelen_overzicht",
        REVIEW_DIRECTORY,
        "Dutch: could be a packed collection of technical components "
        "(the container itself), or an overview document listing them; "
        "unresolved.",
        "latent_container_content",
        ("multilingual",),
    ),
    (
        "raccolta_risorse_progetto",
        REVIEW_DIRECTORY,
        "Italian: could be a packed resource collection (the container "
        "itself), or a manifest document listing project resources; "
        "unresolved.",
        "latent_container_content",
        ("multilingual",),
    ),
    (
        "arquivo_historico_departamental",
        REVIEW_DIRECTORY,
        "Portuguese: 'arquivo' means both 'file' and 'archive' -- "
        "genuinely ambiguous whether this is a single historical document "
        "or a compressed historical archive; unresolved.",
        "latent_container_content",
        ("multilingual",),
    ),
    # === _ToReview: dominant_cue_ambiguity (8) ===============================
    # A strong lexical cue for one category, but the file-role relation
    # needed to commit to it is absent; a model may confidently pick the
    # dominant cue and thereby produce an automatic error.
    (
        "installer_feedback_summary",
        REVIEW_DIRECTORY,
        "'installer' is a strong Installers cue, but 'feedback summary' "
        "suggests a Documents write-up about installer feedback; the "
        "file-role relation is never established.",
        "dominant_cue_ambiguity",
        (),
    ),
    (
        "archivkonzept_entwurf",
        REVIEW_DIRECTORY,
        "German: 'archiv' is a dominant Archives cue, but 'entwurf' "
        "(draft) suggests a Documents draft ABOUT an archive concept; "
        "unresolved.",
        "dominant_cue_ambiguity",
        ("multilingual", "compound_morphology"),
    ),
    (
        "codigo_revision_bitacora",
        REVIEW_DIRECTORY,
        "Spanish: 'codigo' is a dominant Code cue, but 'bitacora' "
        "(log/journal) suggests a Documents journal entry about a code "
        "review; unresolved.",
        "dominant_cue_ambiguity",
        ("multilingual",),
    ),
    (
        "fotoproject_conceptnota",
        REVIEW_DIRECTORY,
        "Dutch: 'foto' is a dominant Images cue, but 'conceptnota' "
        "(concept note) suggests a Documents note about a photo project; "
        "unresolved.",
        "dominant_cue_ambiguity",
        ("multilingual",),
    ),
    (
        "instalacja_notatka_wewnetrzna",
        REVIEW_DIRECTORY,
        "Polish: 'instalacja' is a dominant Installers cue, but "
        "'notatka wewnetrzna' (internal note) suggests a Documents note; "
        "unresolved.",
        "dominant_cue_ambiguity",
        ("multilingual",),
    ),
    (
        "archive_note_reflexion",
        REVIEW_DIRECTORY,
        "French: 'archive' is a dominant Archives cue, but 'note de "
        "reflexion' (reflection note) suggests a Documents artifact; "
        "unresolved.",
        "dominant_cue_ambiguity",
        ("multilingual",),
    ),
    (
        "image_processing_reflection",
        REVIEW_DIRECTORY,
        "'image processing' is a dominant cue toward Code or Images, but "
        "'reflection' suggests a written Documents piece about the "
        "topic; multi-way, unresolved.",
        "dominant_cue_ambiguity",
        (),
    ),
    (
        "treibernotiz_intern",
        REVIEW_DIRECTORY,
        "German: 'treiber' (driver) is a dominant Installers/Code cue, "
        "but 'notiz' (note) suggests a Documents artifact; the file-role "
        "relation is never established.",
        "dominant_cue_ambiguity",
        ("multilingual", "compound_morphology"),
    ),
)

# Explicit ambiguity-marker vocabulary this fixture deliberately avoids in
# its `_ToReview` filenames (item 11/23): these encourage abstention rather
# than confident misclassification and so would not exercise the target
# failure mode. Whole-token match (split on "_"), case-insensitive.
EXPLICIT_AMBIGUITY_MARKERS = frozenset(
    {
        "or",
        "oder",
        "unclear",
        "unklar",
        "unknown",
        "maybe",
        "unbekannt",
        "unbestimmt",
    }
)


def main() -> None:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    expected_names = {name for name, _category, _rationale, _family, _tags in CASES}
    for existing in FIXTURE_ROOT.iterdir():
        if existing.is_file() and existing.name not in expected_names:
            raise RuntimeError(f"unexpected e3-error-calibration fixture file: {existing.name}")
    for name, _category, _rationale, _family, _tags in CASES:
        (FIXTURE_ROOT / name).write_bytes(b"")

    files = {name: [category] for name, category, _rationale, _family, _tags in CASES}
    EXPECTED_PATH.write_text(
        yaml.safe_dump(
            {"files": files}, allow_unicode=True, sort_keys=True, default_flow_style=False
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
