"""Materialize the veto-precision Development stress fixture (metadata-only,
empty files).

This is a **post-result Development stress fixture**, not independent
validation and not a Holdout: it was authored after the previous live
Development results (aggregate findings only -- no case-level Holdout data)
specifically to stress-test and tune E4-refined's veto precision. It may be
inspected, rerun, and tuned against repeatedly, exactly like
``evals/calibration`` and ``evals/boundary_calibration``.

Purpose: measure two things E4-refined must get right simultaneously --
(1) false-veto burden on filenames that legitimately mix cue families from
more than one category but still have one clearly correct category (the
"hard negative" cases below), and (2) whether E4-refined still catches
genuine category-boundary ambiguity, generic insufficient metadata, and
container/content uncertainty. No filename, wording choice, or label here
was copied or reconstructed from ``evals/holdout``, ``evals/holdout_v2``,
``evals/calibration``, or ``evals/boundary_calibration``; every case was
freshly authored for this fixture.

Every file is extensionless, so none can be resolved by the deterministic
extension rules in ``config/rules.yaml`` and none can be excluded by its
glob patterns -- all 72 cases are guaranteed, by construction, to reach
unresolved-file classification (E3/E4-current/E4-refined). Files are 0
bytes: this cycle stays metadata-only, so content is never read.

Composition (72 total): 48 real-category cases (66.7%), 24 ``_ToReview``
cases (33.3%).

Real-category cases (48):

* **Hard negatives for the veto (24)** -- filenames that legitimately
  contain lexical cues from more than one category's vocabulary but still
  have exactly one correct category once the actual file type is
  considered: a document *about* images, code *for* image processing,
  documentation *of* an installer, source code that *manages* archives, a
  script over photo metadata, code that *builds* packages, and further
  variations distributed across all five categories. These exist
  specifically to measure false-veto burden.
* **Ordinary classifiable cases (24)** -- moderate single-category
  evidence, fresh multilingual examples (Portuguese, Polish, Spanish,
  Italian, Dutch), weak/generic noise words alongside a real cue,
  category-specific compounds, and realistic non-trivial filenames.

``_ToReview`` cases (24):

* **Explicit category-boundary ambiguity (12)** -- one case per ordered
  pair among the five categories (plus two repeated pairs with different
  wording), each using an explicit disjunction/uncertainty marker over
  genuinely unresolved two-category semantics.
* **Generic insufficient metadata (6)** -- including fresh
  Spanish/Russian/Chinese/Japanese examples.
* **Container/content uncertainty (6)** -- metadata that cannot
  distinguish whether the file itself is a container/package or whether
  the filename merely describes contained material.

Ground truth was assigned before any inference (none has occurred: this is
implementation- and fixture-construction-only), using the same principle as
every prior Development/Holdout fixture in this project: if the permitted
metadata is sufficient for a human evaluator to safely choose one
production category, assign that category; otherwise ``_ToReview``. Cases
were never labelled by asking what E3/E4-current/E4-refined would predict.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parent
FIXTURE_ROOT = ROOT / "fixture"
EXPECTED_PATH = ROOT / "expected.yaml"

REVIEW_DIRECTORY = "_ToReview"

# (filename, ground_truth_category_or_review, short human-audit rationale)
CASES: tuple[tuple[str, str, str], ...] = (
    # === Hard negatives for the veto (24) ================================
    # --- ground truth Documents (5) ---
    ("fotokurs_abschluss_bericht_teilnehmer", "Documents",
     "Report about a photo course; 'foto' is a topical distractor, the file "
     "itself is a report"),
    ("treiber_versionshistorie_aenderungsprotokoll_dokument", "Documents",
     "Driver changelog document; 'treiber' is a distractor, 'dokument' "
     "names the actual file type"),
    ("installationsreport_fehleranalyse_dokument", "Documents",
     "Installation error-analysis report; 'installation' is a distractor"),
    ("archivformat_spezifikation_dokument", "Documents",
     "Specification document about an archive format; 'archiv' is a "
     "distractor"),
    ("sicherungsstrategie_konzept_dokument", "Documents",
     "Backup-strategy concept document; 'sicherung' is a distractor"),
    # --- ground truth Code (7) ---
    ("bildkonvertierung_stapelverarbeitung_skript", "Code",
     "Script for batch image conversion; 'bild' is a distractor, 'skript' "
     "names the actual file type"),
    ("archivpflege_backend_verwaltung_quellcode", "Code",
     "Backend source managing archive maintenance; 'archiv' is a distractor"),
    ("fotometadaten_extraktion_automatisierung_skript", "Code",
     "Script extracting photo metadata; 'foto' is a distractor"),
    ("paketerstellung_build_pipeline_quellcode", "Code",
     "Source implementing a package-build pipeline; 'paket' is a distractor"),
    ("installationsroutine_entwicklung_quellcode", "Code",
     "Source implementing an install routine; 'installation' is a distractor"),
    ("bildkompression_algorithmus_quellcode", "Code",
     "Image-compression algorithm source; 'bild' is a distractor"),
    ("treiberkompatibilitaet_testsuite_skript", "Code",
     "Driver-compatibility test-suite script; 'treiber' is a distractor"),
    # --- ground truth Images (4) ---
    ("objektiv_kalibrierung_protokoll_aufnahmen", "Images",
     "Lens-calibration test shots; 'protokoll' is a distractor, 'aufnahmen' "
     "names the actual file type"),
    ("treiber_test_aufnahmen_serie", "Images",
     "Test shots taken during driver testing; 'treiber' is a distractor"),
    ("vertragsunterzeichnung_zeremonie_aufnahmen", "Images",
     "Photos from a contract-signing ceremony; 'vertrag' is a distractor"),
    ("quellcode_review_meeting_schnappschuesse", "Images",
     "Snapshots from a code-review meeting; 'quellcode' is a distractor"),
    # --- ground truth Archives (4) ---
    ("fotosammlung_komplettarchiv_2026", "Archives",
     "Complete archive of a photo collection; 'foto' is a distractor, "
     "'archiv' names the actual file type"),
    ("quellcode_releasepaket_archiv", "Archives",
     "Archived release package of source code; 'quellcode' is a distractor"),
    ("vertragsunterlagen_gesamtarchiv_kunde", "Archives",
     "Complete archive of contract documents; 'vertrag' is a distractor"),
    ("installationsdateien_sicherungsarchiv", "Archives",
     "Backup archive of installation files; 'installation' is a distractor"),
    # --- ground truth Installers (4) ---
    ("bildbearbeitung_software_setup_datei", "Installers",
     "Setup file for image-editing software; 'bild' is a distractor"),
    ("quellcode_editor_installationsprogramm", "Installers",
     "Installer for a source-code editor; 'quellcode' is a distractor"),
    ("dokumentenverwaltung_anwendung_installer", "Installers",
     "Installer for a document-management application; 'dokument' is a "
     "distractor"),
    ("archivierungssoftware_setup_ausfuehrbar", "Installers",
     "Setup for archiving software; 'archiv' is a distractor"),
    # === Ordinary classifiable cases (24) =================================
    # --- Images (5) ---
    ("herbstspaziergang_schnappschuss_familie", "Images",
     "Family snapshot from an autumn walk"),
    ("produktfotografie_katalog_serie_v3", "Images",
     "Product-photography catalogue series; weak version noise"),
    ("fotos_de_la_boda_completo", "Images",
     "Spanish: complete wedding photos"),
    ("zdjecia_z_podrozy_zagranicznej", "Images",
     "Polish: photos from a trip abroad"),
    ("galerie_ausstellung_werke_2026", "Images",
     "Exhibition gallery works"),
    # --- Documents (5) ---
    ("mitarbeitergespraech_protokoll_entwurf_v2", "Documents",
     "Employee-conversation minutes draft; weak version/draft noise"),
    ("relatorio_reuniao_diretoria_anual", "Documents",
     "Portuguese: annual board meeting report"),
    ("lettera_raccomandata_risposta_cliente", "Documents",
     "Italian: registered letter, client reply"),
    ("haushaltsplan_uebersicht_naechstes_jahr", "Documents",
     "Household budget overview for next year"),
    ("kuendigungsschreiben_muster_vorlage", "Documents",
     "Termination-letter template"),
    # --- Code (5) ---
    ("datenmigrationsskript_v4_final", "Code",
     "Data-migration script; weak version/final noise"),
    ("webanwendung_frontend_komponenten_bibliothek", "Code",
     "Frontend component library for a web application"),
    ("algoritmo_de_ordenamiento_implementacion", "Code",
     "Spanish: sorting-algorithm implementation"),
    ("infrastructure_as_code_terraform_module", "Code",
     "Infrastructure-as-code Terraform module"),
    ("eenheidstest_suite_nieuwe_versie", "Code",
     "Dutch: unit-test suite, new version"),
    # --- Archives (5) ---
    ("jahresabschluss_unterlagen_komprimiert", "Archives",
     "Compressed year-end financial documents"),
    ("serverkonfiguration_vollsicherung_woche12", "Archives",
     "Full server configuration backup, week 12"),
    ("archivo_comprimido_documentos_antiguos", "Archives",
     "Spanish: compressed archive of old documents"),
    ("kompletny_backup_systemu_operacyjnego", "Archives",
     "Polish: complete operating-system backup"),
    ("musiksammlung_gepackt_uebertragung", "Archives",
     "Packed music collection for transfer"),
    # --- Installers (4) ---
    ("audiotreiber_setup_neueversion", "Installers",
     "Audio-driver setup, new version"),
    ("programma_di_installazione_completo", "Installers",
     "Italian: complete installation program"),
    ("spelclient_installatiebestand_versie2", "Installers",
     "Dutch: game-client installation file"),
    ("netzwerkadapter_firmware_flashtool", "Installers",
     "Network-adapter firmware flash tool"),
    # === _ToReview: explicit category-boundary ambiguity (12) =============
    ("bild_oder_layoutdokument_vorlage_unklar", REVIEW_DIRECTORY,
     "Images vs Documents: explicit 'or', genuinely unresolved"),
    ("aufnahme_oder_renderingskript_ungewiss", REVIEW_DIRECTORY,
     "Images vs Code: explicit uncertainty, genuinely unresolved"),
    ("fotosammlung_oder_komprimiertes_paket_unklar", REVIEW_DIRECTORY,
     "Images vs Archives: explicit 'or', genuinely unresolved"),
    ("bildschirmfoto_oder_installationsscreen_unklar", REVIEW_DIRECTORY,
     "Images vs Installers: explicit 'or', genuinely unresolved"),
    ("dokumentation_oder_quellcode_referenz_unklar", REVIEW_DIRECTORY,
     "Documents vs Code: explicit 'or', genuinely unresolved"),
    ("vertrag_oder_archivierte_unterlagen_ungewiss", REVIEW_DIRECTORY,
     "Documents vs Archives: explicit uncertainty, genuinely unresolved"),
    ("anleitung_oder_installationsdatei_unklar", REVIEW_DIRECTORY,
     "Documents vs Installers: explicit 'or', genuinely unresolved"),
    ("skript_oder_komprimiertes_projekt_ungewiss", REVIEW_DIRECTORY,
     "Code vs Archives: explicit uncertainty, genuinely unresolved"),
    ("quellcode_oder_ausfuehrbare_installationsdatei", REVIEW_DIRECTORY,
     "Code vs Installers: explicit 'or', genuinely unresolved"),
    ("sicherungspaket_oder_installer_unklar", REVIEW_DIRECTORY,
     "Archives vs Installers: explicit 'or', genuinely unresolved"),
    ("bericht_oder_bildmaterial_ungewiss_typ", REVIEW_DIRECTORY,
     "Documents vs Images (second pairing): explicit uncertainty"),
    ("programmdokumentation_oder_quelltext_unklar", REVIEW_DIRECTORY,
     "Code vs Documents (second pairing): explicit 'or'"),
    # === _ToReview: generic insufficient metadata (6) ======================
    ("namenlose_datei_ohne_zusammenhang", REVIEW_DIRECTORY,
     "Unnamed file, no context of any kind"),
    ("exportiert_am_unbekannten_datum", REVIEW_DIRECTORY,
     "Exported on an unknown date, no topic"),
    ("sin_informacion_adicional_disponible", REVIEW_DIRECTORY,
     "Spanish: no additional information available"),
    ("неназванный_объект_без_контекста", REVIEW_DIRECTORY,
     "Russian: unnamed object without context"),
    ("未命名内容待整理", REVIEW_DIRECTORY,
     "Chinese: unnamed content pending organization"),
    ("内容不明_確認必要", REVIEW_DIRECTORY,
     "Japanese: content unclear, confirmation needed"),
    # === _ToReview: container/content uncertainty (6) ======================
    ("gesamtpaket_inhalt_nicht_spezifiziert", REVIEW_DIRECTORY,
     "Package framing, content type explicitly unspecified"),
    ("sammlung_verschiedener_dateitypen_gemischt", REVIEW_DIRECTORY,
     "Mixed collection of different file types, no dominant type"),
    ("projektbuendel_ungeklaerte_zusammensetzung", REVIEW_DIRECTORY,
     "Project bundle, composition genuinely unclarified"),
    ("datenpaket_herkunft_und_inhalt_unklar", REVIEW_DIRECTORY,
     "Data package, origin and content both explicitly unclear"),
    ("archiv_moeglicherweise_gemischte_inhalte", REVIEW_DIRECTORY,
     "Archive, possibly mixed contents; genuine container/content doubt"),
    ("exportbuendel_ohne_inhaltsangabe", REVIEW_DIRECTORY,
     "Export bundle, no content listing at all"),
)


def main() -> None:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    expected_names = {name for name, _category, _rationale in CASES}
    for existing in FIXTURE_ROOT.iterdir():
        if existing.is_file() and existing.name not in expected_names:
            raise RuntimeError(f"unexpected veto-precision fixture file: {existing.name}")
    for name, _category, _rationale in CASES:
        (FIXTURE_ROOT / name).write_bytes(b"")

    files = {name: [category] for name, category, _rationale in CASES}
    EXPECTED_PATH.write_text(
        yaml.safe_dump(
            {"files": files}, allow_unicode=True, sort_keys=True, default_flow_style=False
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
