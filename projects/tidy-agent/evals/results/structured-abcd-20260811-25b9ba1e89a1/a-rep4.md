# tidy-agent evaluation — category mode

- Status: **ok**
- Run mode: **fixed categories only** (`--no-group`, content reading `False`)
- Metric family: **category accuracy only**
- No grouping ran, so no clustering metric is defined in this mode; clustering rows are omitted rather than reported as 0 or N/A.
- Comparable only with other reports whose run mode line is identical.
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Standard run timeout: 240.0 s
- Ground truth: `expected.yaml sha256:dfe0b35b34c6`
- Endpoint: `remote endpoint (address redacted)`
- Model digest: `2a654d98e6fba55d`
- Quantization: `Q4_K_M`
- Parameter size: `4.7B`
- Context length: `8192`
- Model placement: `GPU/VRAM 3341088193 bytes`
- Evaluated: 2026-08-11T19:31:34.593890+00:00
- Warm-up: ok in 19.052 s
- Judge: deterministic expected-category comparison (no LLM judge)

| Metric | Result |
|---|---:|
| Category accuracy, all eligible files | 92.1% (70/76) |
| Category accuracy, all unresolved files (`_ToReview` accepted) | 80.6% (25/31) |
| Decision rate, unresolved files | 77.4% (24/31) |
| Accuracy on decided files only | 75.0% (18/24) |
| Abstentions (`_ToReview`), unresolved files | 7/31 |
| Accuracy, strict subset (exactly one accepted category) | 66.7% (4/6) |
| Files omitted by the model | 1/31 |
| Invalid-assignment fallbacks | 0/31 |
| Files without any usable proposal | 0/31 |
| `_ToReview/` rate, all eligible files | 9.2% (7/76) |
| Invalid plan entries | 0 |
| Correction rounds | 0 |
| Classification steps | 1 |
| Classification latency | 19.837 s |
| Classification input tokens | 295 |
| Classification completion tokens | 354 |
| Classification backend | structured_model |
| Structured output mode | json_object |
| Classification model requests | 1 |
| Peek-phase model requests | 0 |
| Final-classification model requests | 1 |
| Strict JSON parse failures | 0 |
| Schema validation failures | 0 |
| Provider failures | 0 |
| Incomplete structured responses | 1 |
| Duplicate-source responses | 0 |
| Invented-source responses | 0 |
| Invented-category responses | 0 |
| Native-schema responses | 0 |
| JSON-object responses | 1 |
| Strict plain-JSON responses | 0 |
| Structured fallbacks to `_ToReview` | 1 |
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
| `ohne_name` | agent | `_ToReview` | `_ToReview` | PASS | — |
| `meeting-notes` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | — |
| `invoice_march` | agent | `Documents` | `Documents` | PASS | — |
| `rechnung_august` | agent | `Documents` | `Documents` | PASS | — |
| `report:final` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Überweisung_2024` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | omitted |
| `vertrag.old` | agent | `Archives` | `Documents`, `_ToReview` | FAIL | — |
| `final_final_v2` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | — |
| `messwerte.dat` | agent | `Code` | `Documents`, `_ToReview` | FAIL | — |
| `vacation-photo` | agent | `Images` | `Images`, `_ToReview` | PASS | — |
| `urlaub_2025` | agent | `Images` | `Images`, `_ToReview` | PASS | — |
| `foto_export` | agent | `Images` | `Images`, `_ToReview` | PASS | — |
| `design.sketch` | agent | `Images` | `Images`, `_ToReview` | PASS | — |
| `source-code-snippet` | agent | `Code` | `Code`, `_ToReview` | PASS | — |
| `quellcode_alt` | agent | `Code` | `Code`, `_ToReview` | PASS | — |
| `data.backup` | agent | `Archives` | `Archives`, `_ToReview` | PASS | — |
| `backup.gw` | agent | `Documents` | `Archives`, `_ToReview` | FAIL | — |
| `setup_latest` | agent | `Installers` | `Installers`, `_ToReview` | PASS | — |
| `download` | agent | `Code` | `_ToReview`, `Archives`, `Installers` | FAIL | — |
| `kundenliste.bak` | agent | `Archives` | `Documents`, `Archives`, `_ToReview` | PASS | — |
| `README` | agent | `Documents` | `Documents`, `Code`, `_ToReview` | PASS | — |
| `projektstand` | agent | `_ToReview` | `Code`, `Archives`, `Documents`, `_ToReview` | PASS | — |
| `Kritik_Bachelorarbeit_GPT.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Pruefbericht_Bachelorarbeit_v4.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Kundenportal_Relaunch_Briefing.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Review_Klimastudie.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `music_mix.m3u` | agent | `Images` | `_ToReview` | FAIL | — |
| `steuerbescheid_2024` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `sitzungsprotokoll_q3` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | — |
| `geraetedump` | agent | `Documents` | `_ToReview` | FAIL | — |
