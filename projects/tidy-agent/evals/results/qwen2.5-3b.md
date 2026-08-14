# tidy-agent evaluation

> **Superseded — do not quote these numbers.** Produced before the metric definitions were corrected (2026-08-10) and before `thesis_draft.docx` joined the `bachelorarbeit` ground-truth group. Its accuracy rows credit abstention like a correct decision, and any clustering row counts files co-located by extension rule. Kept as a record of the run only; the current results are the `-x3` reports.


- Status: **partial** (48/58 cases written)
- Model: `ollama_chat/qwen2.5:3b`
- Think: `False`
- Per-case timeout: 120.0 s
- Evaluated: 2026-08-10T14:41:10.681741+00:00
- Judge: deterministic expected-category comparison (no LLM judge)
- Warm-up: ok in 10.692 s

| Metric | Result |
|---|---:|
| Overall assignment accuracy | 79.3% (46/58) |
| Accuracy on unresolved files | 60.9% (14/23) |
| `_ToReview/` rate | 25.6% (11/43) |
| Average agent steps | 3.62 |
| Average agent latency | 7.553 s |
| Average completion tokens | 359.7 |
| Completion tokens total | 5755 |
| Timeouts | 0 |
| Invalid plan entries | 15 |
| Correction rounds | 15 |
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
| `mystery` | agent | ok | `_ToReview` | `_ToReview` | PASS | 3 | 8.526 s | 398 |
| `meeting-notes` | agent | ok | `Documents` | `Documents`, `_ToReview` | PASS | 6 | 13.086 s | 657 |
| `invoice_march` | agent | ok | `_ToReview` | `Documents` | FAIL | 9 | 16.496 s | 744 |
| `vacation-photo` | agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 9 | 18.275 s | 862 |
| `source-code-snippet` | agent | ok | `Code` | `Code`, `_ToReview` | PASS | 2 | 3.122 s | 134 |
| `download` | agent | ok | `_ToReview` | `_ToReview`, `Archives`, `Installers` | PASS | 2 | 3.006 s | 128 |
| `data.backup` | agent | ok | `_ToReview` | `Archives`, `_ToReview` | PASS | 2 | 3.341 s | 153 |
| `design.sketch` | agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 2 | 3.597 s | 164 |
| `report:final` | agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 1 | 5.083 s | 279 |
| `Überweisung_2024` | agent | ok | `_ToReview` | `Documents`, `_ToReview` | PASS | 1 | 3.721 s | 192 |
| `README` | agent | ok | `Documents` | `Documents`, `Code`, `_ToReview` | PASS | 2 | 3.096 s | 137 |
| `~$Bachelorarbeit_v4.docx` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `.DS_Store` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `partial.crdownload` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `download.part` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `cache.tmp` | excluded | excluded | `EXCLUDED` | `EXCLUDED` | PASS | 0 | 0.000 s | 0 |
| `urlaub_2025` | agent | ok | `_ToReview` | `Images`, `_ToReview` | PASS | 9 | 23.225 s | 1157 |
| `rechnung_august` | agent | ok | `_ToReview` | `Documents` | FAIL | 2 | 3.056 s | 134 |
| `quellcode_alt` | agent | ok | `Code` | `Code`, `_ToReview` | PASS | 2 | 3.303 s | 149 |
| `backup.gw` | agent | ok | `_ToReview` | `Archives`, `_ToReview` | PASS | 2 | 3.304 s | 146 |
| `kundenliste.bak` | agent | ok | `Archives` | `Documents`, `Archives`, `_ToReview` | PASS | 4 | 6.608 s | 321 |
