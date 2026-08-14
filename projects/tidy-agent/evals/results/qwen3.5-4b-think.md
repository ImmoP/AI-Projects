# tidy-agent evaluation

> **Superseded — do not quote these numbers.** Produced before the metric definitions were corrected (2026-08-10) and before `thesis_draft.docx` joined the `bachelorarbeit` ground-truth group. Its accuracy rows credit abstention like a correct decision, and any clustering row counts files co-located by extension rule. Kept as a record of the run only; the current results are the `-x3` reports.


- Status: **complete** (58/58 cases written)
- Model: `ollama_chat/qwen3.5:4b`
- Think: `True`
- Per-case timeout: 120.0 s
- Evaluated: 2026-08-10T14:08:58.913424+00:00
- Judge: deterministic expected-category comparison (no LLM judge)
- Warm-up: ok in 9.975 s

| Metric | Result |
|---|---:|
| Overall assignment accuracy | 89.7% (52/58) |
| Accuracy on unresolved files | 73.9% (17/23) |
| `_ToReview/` rate | 35.4% (17/48) |
| Average agent steps | 2.65 |
| Average agent latency | 93.360 s |
| Average completion tokens | 1051.4 |
| Completion tokens total | 24183 |
| Timeouts | 5 |
| Invalid plan entries | 12 |
| Correction rounds | 13 |
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
| `mystery` | agent | ok | `_ToReview` | `_ToReview` | PASS | 3 | 23.240 s | 853 |
| `meeting-notes` | agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 3 | 38.435 s | 1448 |
| `invoice_march` | agent | timeout | `TIMEOUT` | `Documents` | FAIL | 0 | 120.010 s | 0 |
| `vacation-photo` | agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 3 | 40.820 s | 1535 |
| `source-code-snippet` | agent | ok | `Code` | `Code`, `_ToReview` | PASS | 1 | 11.720 s | 461 |
| `download` | agent | ok | `_ToReview` | `_ToReview`, `Archives`, `Installers` | PASS | 1 | 11.152 s | 437 |
| `data.backup` | agent | ok | `_ToReview` | `Archives`, `_ToReview` | PASS | 3 | 43.993 s | 1653 |
| `design.sketch` | agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 2 | 24.691 s | 973 |
| `report:final` | agent | timeout | `TIMEOUT` | `Documents`, `_ToReview` | FAIL | 0 | 120.005 s | 0 |
| `Überweisung_2024` | agent | timeout | `TIMEOUT` | `Documents`, `_ToReview` | FAIL | 0 | 120.005 s | 0 |
| `README` | agent | ok | `_ToReview` | `Documents`, `Code`, `_ToReview` | PASS | 9 | 649.415 s | 2563 |
| `~$Bachelorarbeit_v4.docx` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `.DS_Store` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `partial.crdownload` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `download.part` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `cache.tmp` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `urlaub_2025` | agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 1 | 24.939 s | 697 |
| `rechnung_august` | agent | ok | `_ToReview` | `Documents` | FAIL | 9 | 83.547 s | 3124 |
| `quellcode_alt` | agent | ok | `_ToReview` | `Code`, `_ToReview` | PASS | 1 | 42.089 s | 675 |
| `backup.gw` | agent | ok | `_ToReview` | `Archives`, `_ToReview` | PASS | 1 | 13.060 s | 521 |
| `kundenliste.bak` | agent | ok | `_ToReview` | `Documents`, `Archives`, `_ToReview` | PASS | 9 | 89.012 s | 3124 |
| `messwerte.dat` | agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 1 | 116.823 s | 581 |
| `vertrag.old` | agent | timeout | `TIMEOUT` | `Documents`, `_ToReview` | FAIL | 0 | 120.004 s | 0 |
| `final_final_v2` | agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 9 | 139.080 s | 3201 |
| `ohne_name` | agent | timeout | `TIMEOUT` | `_ToReview` | FAIL | 0 | 120.004 s | 0 |
| `setup_latest` | agent | ok | `_ToReview` | `Installers`, `_ToReview` | PASS | 1 | 150.355 s | 610 |
| `foto_export` | agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 1 | 15.613 s | 628 |
| `projektstand` | agent | ok | `_ToReview` | `Code`, `Archives`, `Documents`, `_ToReview` | PASS | 3 | 29.279 s | 1099 |
| `dokument.pdf` | rule | ok | `Documents` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `unbenannt 3.txt` | rule | ok | `Documents` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `Scan_2025-01-09.pdf` | rule | ok | `Documents` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
