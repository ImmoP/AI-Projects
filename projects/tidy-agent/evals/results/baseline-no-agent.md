# tidy-agent evaluation

> **Superseded — do not quote these numbers.** Produced before the metric definitions were corrected (2026-08-10) and before `thesis_draft.docx` joined the `bachelorarbeit` ground-truth group. Its accuracy rows credit abstention like a correct decision, and any clustering row counts files co-located by extension rule. Kept as a record of the run only; the current results are the `-x3` reports.


- Status: **complete** (58/58 cases written)
- Model: `rules-only`
- Think: `model default`
- Per-case timeout: 120.0 s
- Evaluated: 2026-08-10T13:22:39.023375+00:00
- Judge: deterministic expected-category comparison (no LLM judge)

| Metric | Result |
|---|---:|
| Overall assignment accuracy | 96.6% (56/58) |
| Accuracy on unresolved files | 91.3% (21/23) |
| `_ToReview/` rate | 43.4% (23/53) |
| Average agent steps | 0.00 |
| Average agent latency | 0.000 s |
| Average completion tokens | 0.0 |
| Completion tokens total | 0 |
| Timeouts | 0 |
| Invalid plan entries | 0 |
| Correction rounds | 0 |
| Errors | 0 |

## Assignments

| File | Mode | Status | Predicted | Accepted | Correct | Steps | Latency | Tokens |
|---|---|---|---|---|---:|---:|---:|---:|
| `Screenshot 2026-08-10 at 10.15.22.png` | rule | ok | `Images` | `Images` | PASS | 0 | 0.000 s | 0 |
| `IMG_4821.HEIC` | rule | ok | `Images` | `Images` | PASS | 0 | 0.000 s | 0 |
| `urlaub_münchen.jpg` | rule | ok | `Images` | `Images` | PASS | 0 | 0.000 s | 0 |
| `logo-final.svg` | rule | ok | `Images` | `Images` | PASS | 0 | 0.000 s | 0 |
| `scan-receipt.webp` | rule | ok | `Images` | `Images` | PASS | 0 | 0.000 s | 0 |
| `Rechnung_2025-11.pdf` | rule | ok | `Documents` | `Documents` | PASS | 0 | 0.000 s | 0 |
| `Scan_2024-03-11.pdf` | rule | ok | `Documents` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `dokument_final_final.pdf` | rule | ok | `Documents` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `thesis_draft.docx` | rule | ok | `Documents` | `Documents` | PASS | 0 | 0.000 s | 0 |
| `Budget Q4.xlsx` | rule | ok | `Documents` | `Documents` | PASS | 0 | 0.000 s | 0 |
| `präsentation-köln.pptx` | rule | ok | `Documents` | `Documents` | PASS | 0 | 0.000 s | 0 |
| `notes.txt` | rule | ok | `Documents` | `Documents` | PASS | 0 | 0.000 s | 0 |
| `daten.csv` | rule | ok | `Documents` | `Documents` | PASS | 0 | 0.000 s | 0 |
| `project-source.zip` | rule | ok | `Archives` | `Archives` | PASS | 0 | 0.000 s | 0 |
| `photos_backup.tar.gz` | rule | ok | `Archives` | `Archives` | PASS | 0 | 0.000 s | 0 |
| `archive_2024.7z` | rule | ok | `Archives` | `Archives` | PASS | 0 | 0.000 s | 0 |
| `logs.tgz` | rule | ok | `Archives` | `Archives` | PASS | 0 | 0.000 s | 0 |
| `tidy_agent.py` | rule | ok | `Code` | `Code` | PASS | 0 | 0.000 s | 0 |
| `app.tsx` | rule | ok | `Code` | `Code` | PASS | 0 | 0.000 s | 0 |
| `config.yaml` | rule | ok | `Code` | `Code` | PASS | 0 | 0.000 s | 0 |
| `notebook.ipynb` | rule | ok | `Code` | `Code` | PASS | 0 | 0.000 s | 0 |
| `build.sh` | rule | ok | `Code` | `Code` | PASS | 0 | 0.000 s | 0 |
| `package.json` | rule | ok | `Code` | `Code` | PASS | 0 | 0.000 s | 0 |
| `installer.dmg` | rule | ok | `Installers` | `Installers` | PASS | 0 | 0.000 s | 0 |
| `setup_windows.exe` | rule | ok | `Installers` | `Installers` | PASS | 0 | 0.000 s | 0 |
| `app-release.AppImage` | rule | ok | `Installers` | `Installers` | PASS | 0 | 0.000 s | 0 |
| `tool.deb` | rule | ok | `Installers` | `Installers` | PASS | 0 | 0.000 s | 0 |
| `mystery` | no-agent | ok | `_ToReview` | `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `meeting-notes` | no-agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `invoice_march` | no-agent | ok | `_ToReview` | `Documents` | FAIL | 0 | 0.000 s | 0 |
| `vacation-photo` | no-agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `source-code-snippet` | no-agent | ok | `_ToReview` | `Code`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `download` | no-agent | ok | `_ToReview` | `_ToReview`, `Archives`, `Installers` | PASS | 0 | 0.000 s | 0 |
| `data.backup` | no-agent | ok | `_ToReview` | `Archives`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `design.sketch` | no-agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `report:final` | no-agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `Überweisung_2024` | no-agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `README` | no-agent | ok | `_ToReview` | `Documents`, `Code`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `~$Bachelorarbeit_v4.docx` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `.DS_Store` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `partial.crdownload` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `download.part` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `cache.tmp` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `urlaub_2025` | no-agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `rechnung_august` | no-agent | ok | `_ToReview` | `Documents` | FAIL | 0 | 0.000 s | 0 |
| `quellcode_alt` | no-agent | ok | `_ToReview` | `Code`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `backup.gw` | no-agent | ok | `_ToReview` | `Archives`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `kundenliste.bak` | no-agent | ok | `_ToReview` | `Documents`, `Archives`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `messwerte.dat` | no-agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `vertrag.old` | no-agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `final_final_v2` | no-agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `ohne_name` | no-agent | ok | `_ToReview` | `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `setup_latest` | no-agent | ok | `_ToReview` | `Installers`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `foto_export` | no-agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `projektstand` | no-agent | ok | `_ToReview` | `Code`, `Archives`, `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `dokument.pdf` | rule | ok | `Documents` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `unbenannt 3.txt` | rule | ok | `Documents` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `Scan_2025-01-09.pdf` | rule | ok | `Documents` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
