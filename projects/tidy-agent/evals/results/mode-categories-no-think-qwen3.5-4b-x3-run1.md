# tidy-agent evaluation — category mode

- Status: **ok**
- Run mode: **fixed categories only** (`--no-group`, content reading `False`)
- Metric family: **category accuracy only**
- No grouping ran, so no clustering metric is defined in this mode; clustering rows are omitted rather than reported as 0 or N/A.
- Comparable only with other reports whose run mode line is identical.
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Standard run timeout: 300.0 s
- Evaluated: 2026-08-10T22:14:50.882431+00:00
- Warm-up: ok in 11.534 s
- Judge: deterministic expected-category comparison (no LLM judge)

| Metric | Result |
|---|---:|
| Category accuracy, all eligible files | 93.2% (68/73) |
| Category accuracy, all unresolved files (`_ToReview` accepted) | 82.1% (23/28) |
| Decision rate, unresolved files | 78.6% (22/28) |
| Accuracy on decided files only | 77.3% (17/22) |
| Abstentions (`_ToReview`), unresolved files | 6/28 |
| Accuracy, strict subset (exactly one accepted category) | 80.0% (4/5) |
| Files omitted by the model | 1/28 |
| Invalid-assignment fallbacks | 0/28 |
| Files without any usable proposal | 0/28 |
| `_ToReview/` rate, all eligible files | 8.2% (6/73) |
| Invalid plan entries | 1 |
| Correction rounds | 1 |
| Classification steps | 3 |
| Classification latency | 32.247 s |
| Classification input tokens | 11215 |
| Classification completion tokens | 1121 |
| Agent runs | 1 |

`_ToReview` is an accepted answer for most ambiguous filenames, so the unresolved-accuracy row credits abstention exactly as much as a correct decision — a model that always abstained would score well on it. **Decision rate** and **accuracy on decided files** separate the two and are the rows to compare against the rules-only baseline, which decides nothing.

Omitted by the model (deterministic `_ToReview/` fallback): `Überweisung_2024`

## Assignments

| File | Mode | Predicted | Accepted | Correct | Fallback |
|---|---|---|---|---|---|
| `Screenshot 2026-08-10 at 10.15.22.png` | rule | `Images` | `Images` | PASS | — |
| `IMG_4821.HEIC` | rule | `Images` | `Images` | PASS | — |
| `urlaub_münchen.jpg` | rule | `Images` | `Images` | PASS | — |
| `logo-final.svg` | rule | `Images` | `Images` | PASS | — |
| `scan-receipt.webp` | rule | `Images` | `Images` | PASS | — |
| `Fahrradverkauf.jpg` | rule | `Images` | `Images` | PASS | — |
| `Kundenportal_Relaunch_Mockup.png` | rule | `Images` | `Images` | PASS | — |
| `Rechnung_2025-11.pdf` | rule | `Documents` | `Documents` | PASS | — |
| `Scan_2024-03-11.pdf` | rule | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Scan_2025-01-09.pdf` | rule | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `dokument.pdf` | rule | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `dokument_final_final.pdf` | rule | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `thesis_draft.docx` | rule | `Documents` | `Documents` | PASS | — |
| `Budget Q4.xlsx` | rule | `Documents` | `Documents` | PASS | — |
| `präsentation-köln.pptx` | rule | `Documents` | `Documents` | PASS | — |
| `notes.txt` | rule | `Documents` | `Documents` | PASS | — |
| `unbenannt 3.txt` | rule | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `daten.csv` | rule | `Documents` | `Documents` | PASS | — |
| `Bachelorarbeit_v3.docx` | rule | `Documents` | `Documents` | PASS | — |
| `Bachelorarbeit_v4.docx` | rule | `Documents` | `Documents` | PASS | — |
| `Bachelorarbeit_v5.docx` | rule | `Documents` | `Documents` | PASS | — |
| `Bachelorarbeit_v6.docx` | rule | `Documents` | `Documents` | PASS | — |
| `Kundenportal_Relaunch_Vertrag.pdf` | rule | `Documents` | `Documents` | PASS | — |
| `Kundenportal_Relaunch_Notizen.txt` | rule | `Documents` | `Documents` | PASS | — |
| `Klimastudie_Daten.csv` | rule | `Documents` | `Documents` | PASS | — |
| `Klimastudie_Entwurf.docx` | rule | `Documents` | `Documents` | PASS | — |
| `Geburtstagskarte_Lena.pdf` | rule | `Documents` | `Documents` | PASS | — |
| `Kochrezept_Risotto.txt` | rule | `Documents` | `Documents` | PASS | — |
| `museum_ticket.pdf` | rule | `Documents` | `Documents` | PASS | — |
| `project-source.zip` | rule | `Archives` | `Archives` | PASS | — |
| `photos_backup.tar.gz` | rule | `Archives` | `Archives` | PASS | — |
| `archive_2024.7z` | rule | `Archives` | `Archives` | PASS | — |
| `logs.tgz` | rule | `Archives` | `Archives` | PASS | — |
| `tidy_agent.py` | rule | `Code` | `Code` | PASS | — |
| `app.tsx` | rule | `Code` | `Code` | PASS | — |
| `config.yaml` | rule | `Code` | `Code` | PASS | — |
| `notebook.ipynb` | rule | `Code` | `Code` | PASS | — |
| `build.sh` | rule | `Code` | `Code` | PASS | — |
| `package.json` | rule | `Code` | `Code` | PASS | — |
| `server_inventory.yaml` | rule | `Code` | `Code` | PASS | — |
| `Klimastudie_Auswertung.ipynb` | rule | `Code` | `Code` | PASS | — |
| `installer.dmg` | rule | `Installers` | `Installers` | PASS | — |
| `setup_windows.exe` | rule | `Installers` | `Installers` | PASS | — |
| `app-release.AppImage` | rule | `Installers` | `Installers` | PASS | — |
| `tool.deb` | rule | `Installers` | `Installers` | PASS | — |
| `mystery` | agent | `_ToReview` | `_ToReview` | PASS | — |
| `ohne_name` | agent | `Documents` | `_ToReview` | FAIL | — |
| `meeting-notes` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `invoice_march` | agent | `Documents` | `Documents` | PASS | — |
| `rechnung_august` | agent | `Documents` | `Documents` | PASS | — |
| `report:final` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Überweisung_2024` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | omitted |
| `vertrag.old` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `final_final_v2` | agent | `Code` | `Documents`, `_ToReview` | FAIL | — |
| `messwerte.dat` | agent | `Code` | `Documents`, `_ToReview` | FAIL | — |
| `vacation-photo` | agent | `Images` | `Images`, `_ToReview` | PASS | — |
| `urlaub_2025` | agent | `_ToReview` | `Images`, `_ToReview` | PASS | — |
| `foto_export` | agent | `Images` | `Images`, `_ToReview` | PASS | — |
| `design.sketch` | agent | `Documents` | `Images`, `_ToReview` | FAIL | — |
| `source-code-snippet` | agent | `Code` | `Code`, `_ToReview` | PASS | — |
| `quellcode_alt` | agent | `Code` | `Code`, `_ToReview` | PASS | — |
| `data.backup` | agent | `Archives` | `Archives`, `_ToReview` | PASS | — |
| `backup.gw` | agent | `Images` | `Archives`, `_ToReview` | FAIL | — |
| `setup_latest` | agent | `_ToReview` | `Installers`, `_ToReview` | PASS | — |
| `download` | agent | `_ToReview` | `_ToReview`, `Archives`, `Installers` | PASS | — |
| `kundenliste.bak` | agent | `Archives` | `Documents`, `Archives`, `_ToReview` | PASS | — |
| `README` | agent | `Code` | `Documents`, `Code`, `_ToReview` | PASS | — |
| `projektstand` | agent | `Documents` | `Code`, `Archives`, `Documents`, `_ToReview` | PASS | — |
| `Kritik_Bachelorarbeit_GPT.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Pruefbericht_Bachelorarbeit_v4.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Kundenportal_Relaunch_Briefing.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Review_Klimastudie.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `music_mix.m3u` | agent | `_ToReview` | `_ToReview` | PASS | — |
