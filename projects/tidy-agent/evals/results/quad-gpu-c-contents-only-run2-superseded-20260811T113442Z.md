# tidy-agent evaluation — category mode

- Status: **ok**
- Run mode: **fixed categories only** (`--no-group`, content reading `True`)
- Metric family: **category accuracy only**
- No grouping ran, so no clustering metric is defined in this mode; clustering rows are omitted rather than reported as 0 or N/A.
- Comparable only with other reports whose run mode line is identical.
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Standard run timeout: 900.0 s
- Ground truth: `expected.yaml sha256:c97e0d99d26b`
- Endpoint: `remote GPU host (tailnet address redacted)`
- Model digest: `2a654d98e6fba55d`
- Quantization: `Q4_K_M`
- Parameter size: `4.7B`
- Context length: `8192`
- Model placement: `GPU/VRAM 3341088193 bytes`
- Evaluated: 2026-08-11T11:34:42.603123+00:00
- Warm-up: ok in 7.481 s
- Judge: deterministic expected-category comparison (no LLM judge)

| Metric | Result |
|---|---:|
| Category accuracy, all eligible files | 97.3% (71/73) |
| Category accuracy, all unresolved files (`_ToReview` accepted) | 92.9% (26/28) |
| Decision rate, unresolved files | 0.0% (0/28) |
| Accuracy on decided files only | n/a (0/0) |
| Abstentions (`_ToReview`), unresolved files | 28/28 |
| Accuracy, strict subset (exactly one accepted category) | 60.0% (3/5) |
| Files omitted by the model | 0/28 |
| Invalid-assignment fallbacks | 0/28 |
| Files without any usable proposal | 28/28 |
| `_ToReview/` rate, all eligible files | 38.4% (28/73) |
| Invalid plan entries | 0 |
| Correction rounds | 0 |
| Classification steps | 9 |
| Classification latency | 124.588 s |
| Classification input tokens | 52290 |
| Classification completion tokens | 4033 |
| Content peeks attempted | 1 |
| Peeks that returned text | 0 |
| Unresolved files peek_file can read at all | 4/28 |
| Agent runs | 1 |

`_ToReview` is an accepted answer for most ambiguous filenames, so the unresolved-accuracy row credits abstention exactly as much as a correct decision — a model that always abstained would score well on it. **Decision rate** and **accuracy on decided files** separate the two and are the rows to compare against the rules-only baseline, which decides nothing.

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
| `mystery` | agent | `_ToReview` | `_ToReview` | PASS | no-proposal |
| `ohne_name` | agent | `_ToReview` | `_ToReview` | PASS | no-proposal |
| `meeting-notes` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | no-proposal |
| `invoice_march` | agent | `_ToReview` | `Documents` | FAIL | no-proposal |
| `rechnung_august` | agent | `_ToReview` | `Documents` | FAIL | no-proposal |
| `report:final` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | no-proposal |
| `Überweisung_2024` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | no-proposal |
| `vertrag.old` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | no-proposal |
| `final_final_v2` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | no-proposal |
| `messwerte.dat` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | no-proposal |
| `vacation-photo` | agent | `_ToReview` | `Images`, `_ToReview` | PASS | no-proposal |
| `urlaub_2025` | agent | `_ToReview` | `Images`, `_ToReview` | PASS | no-proposal |
| `foto_export` | agent | `_ToReview` | `Images`, `_ToReview` | PASS | no-proposal |
| `design.sketch` | agent | `_ToReview` | `Images`, `_ToReview` | PASS | no-proposal |
| `source-code-snippet` | agent | `_ToReview` | `Code`, `_ToReview` | PASS | no-proposal |
| `quellcode_alt` | agent | `_ToReview` | `Code`, `_ToReview` | PASS | no-proposal |
| `data.backup` | agent | `_ToReview` | `Archives`, `_ToReview` | PASS | no-proposal |
| `backup.gw` | agent | `_ToReview` | `Archives`, `_ToReview` | PASS | no-proposal |
| `setup_latest` | agent | `_ToReview` | `Installers`, `_ToReview` | PASS | no-proposal |
| `download` | agent | `_ToReview` | `_ToReview`, `Archives`, `Installers` | PASS | no-proposal |
| `kundenliste.bak` | agent | `_ToReview` | `Documents`, `Archives`, `_ToReview` | PASS | no-proposal |
| `README` | agent | `_ToReview` | `Documents`, `Code`, `_ToReview` | PASS | no-proposal |
| `projektstand` | agent | `_ToReview` | `Code`, `Archives`, `Documents`, `_ToReview` | PASS | no-proposal |
| `Kritik_Bachelorarbeit_GPT.md` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | no-proposal |
| `Pruefbericht_Bachelorarbeit_v4.md` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | no-proposal |
| `Kundenportal_Relaunch_Briefing.md` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | no-proposal |
| `Review_Klimastudie.md` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | no-proposal |
| `music_mix.m3u` | agent | `_ToReview` | `_ToReview` | PASS | no-proposal |
