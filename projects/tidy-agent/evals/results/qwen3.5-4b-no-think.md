# tidy-agent evaluation

> **Superseded — do not quote these numbers.** Produced before the metric definitions were corrected (2026-08-10) and before `thesis_draft.docx` joined the `bachelorarbeit` ground-truth group. Its accuracy rows credit abstention like a correct decision, and any clustering row counts files co-located by extension rule. Kept as a record of the run only; the current results are the `-x3` reports.


- Status: **complete** (58/58 cases written)
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Per-case timeout: 120.0 s
- Evaluated: 2026-08-10T14:36:07.632711+00:00
- Judge: deterministic expected-category comparison (no LLM judge)
- Warm-up: ok in 6.380 s

| Metric | Result |
|---|---:|
| Overall assignment accuracy | 98.3% (57/58) |
| Accuracy on unresolved files | 95.7% (22/23) |
| `_ToReview/` rate | 39.6% (21/53) |
| Average agent steps | 2.09 |
| Average agent latency | 14.677 s |
| Average completion tokens | 478.9 |
| Completion tokens total | 11015 |
| Timeouts | 0 |
| Invalid plan entries | 22 |
| Correction rounds | 22 |
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
| `mystery` | agent | ok | `_ToReview` | `_ToReview` | PASS | 1 | 6.011 s | 209 |
| `meeting-notes` | agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 2 | 14.052 s | 510 |
| `invoice_march` | agent | ok | `_ToReview` | `Documents` | FAIL | 1 | 4.901 s | 166 |
| `vacation-photo` | agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 1 | 24.212 s | 301 |
| `source-code-snippet` | agent | ok | `_ToReview` | `Code`, `_ToReview` | PASS | 1 | 10.093 s | 388 |
| `download` | agent | ok | `_ToReview` | `_ToReview`, `Archives`, `Installers` | PASS | 1 | 9.477 s | 363 |
| `data.backup` | agent | ok | `_ToReview` | `Archives`, `_ToReview` | PASS | 1 | 10.287 s | 395 |
| `design.sketch` | agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 4 | 25.199 s | 841 |
| `report:final` | agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 5 | 19.179 s | 537 |
| `Überweisung_2024` | agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 9 | 64.252 s | 2021 |
| `README` | agent | ok | `_ToReview` | `Documents`, `Code`, `_ToReview` | PASS | 1 | 7.243 s | 265 |
| `~$Bachelorarbeit_v4.docx` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `.DS_Store` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `partial.crdownload` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `download.part` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `cache.tmp` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `urlaub_2025` | agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 1 | 9.175 s | 344 |
| `rechnung_august` | agent | ok | `Documents` | `Documents` | PASS | 4 | 28.799 s | 979 |
| `quellcode_alt` | agent | ok | `Code` | `Code`, `_ToReview` | PASS | 4 | 21.859 s | 715 |
| `backup.gw` | agent | ok | `_ToReview` | `Archives`, `_ToReview` | PASS | 1 | 4.270 s | 136 |
| `kundenliste.bak` | agent | ok | `_ToReview` | `Documents`, `Archives`, `_ToReview` | PASS | 1 | 8.954 s | 340 |
| `messwerte.dat` | agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 3 | 15.822 s | 525 |
| `vertrag.old` | agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 1 | 10.930 s | 421 |
| `final_final_v2` | agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 1 | 6.318 s | 223 |
| `ohne_name` | agent | ok | `_ToReview` | `_ToReview` | PASS | 2 | 12.885 s | 447 |
| `setup_latest` | agent | ok | `_ToReview` | `Installers`, `_ToReview` | PASS | 1 | 12.163 s | 484 |
| `foto_export` | agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 1 | 5.902 s | 210 |
| `projektstand` | agent | ok | `_ToReview` | `Code`, `Archives`, `Documents`, `_ToReview` | PASS | 1 | 5.578 s | 195 |
| `dokument.pdf` | rule | ok | `Documents` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `unbenannt 3.txt` | rule | ok | `Documents` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
| `Scan_2025-01-09.pdf` | rule | ok | `Documents` | `Documents`, `_ToReview` | PASS | 0 | 0.000 s | 0 |
