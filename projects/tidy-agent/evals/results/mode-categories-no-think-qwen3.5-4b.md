# tidy-agent evaluation — category mode

> **Superseded — do not quote these numbers.** This report was produced before the metric definitions were corrected: clustering rows counted files that the extension rules co-located rather than files that were clustered, the unresolved-accuracy row credits abstention like a correct decision, and the grouped-file count is a different set from the group members in this report's own placement table. Kept as a record of the run only; re-run before citing.


- Status: **ok**
- Run mode: **fixed categories only** (`--no-group`, content reading `False`)
- Metric family: **category accuracy only**
- No grouping ran, so no clustering metric is defined in this mode; clustering rows are omitted rather than reported as 0 or N/A.
- Comparable only with other reports whose run mode line is identical.
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Standard run timeout: 300.0 s
- Evaluated: 2026-08-10T17:44:15.719187+00:00
- Warm-up: ok in 6.294 s
- Judge: deterministic expected-category comparison (no LLM judge)

| Metric | Result |
|---|---:|
| Category accuracy, all eligible files | 90.4% (66/73) |
| Category accuracy, unresolved files only | 75.0% (21/28) |
| Files omitted by the model | 0/28 |
| Invalid-assignment fallbacks | 0/28 |
| Files without any usable proposal | 0/28 |
| `_ToReview/` rate | 5.5% (4/73) |
| Invalid plan entries | 0 |
| Correction rounds | 0 |
| Classification steps | 1 |
| Classification latency | 27.022 s |
| Classification input tokens | 2888 |
| Classification completion tokens | 1121 |
| Agent runs | 1 |

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
| `meeting-notes` | agent | `Code` | `Documents`, `_ToReview` | FAIL | — |
| `invoice_march` | agent | `Documents` | `Documents` | PASS | — |
| `rechnung_august` | agent | `Documents` | `Documents` | PASS | — |
| `report:final` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Überweisung_2024` | agent | `Archives` | `Documents`, `_ToReview` | FAIL | — |
| `vertrag.old` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `final_final_v2` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `messwerte.dat` | agent | `Archives` | `Documents`, `_ToReview` | FAIL | — |
| `vacation-photo` | agent | `Images` | `Images`, `_ToReview` | PASS | — |
| `urlaub_2025` | agent | `Images` | `Images`, `_ToReview` | PASS | — |
| `foto_export` | agent | `Images` | `Images`, `_ToReview` | PASS | — |
| `design.sketch` | agent | `Archives` | `Images`, `_ToReview` | FAIL | — |
| `source-code-snippet` | agent | `Code` | `Code`, `_ToReview` | PASS | — |
| `quellcode_alt` | agent | `Code` | `Code`, `_ToReview` | PASS | — |
| `data.backup` | agent | `Documents` | `Archives`, `_ToReview` | FAIL | — |
| `backup.gw` | agent | `Images` | `Archives`, `_ToReview` | FAIL | — |
| `setup_latest` | agent | `Archives` | `Installers`, `_ToReview` | FAIL | — |
| `download` | agent | `_ToReview` | `_ToReview`, `Archives`, `Installers` | PASS | — |
| `kundenliste.bak` | agent | `Archives` | `Documents`, `Archives`, `_ToReview` | PASS | — |
| `README` | agent | `Documents` | `Documents`, `Code`, `_ToReview` | PASS | — |
| `projektstand` | agent | `Documents` | `Code`, `Archives`, `Documents`, `_ToReview` | PASS | — |
| `Kritik_Bachelorarbeit_GPT.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Pruefbericht_Bachelorarbeit_v4.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Kundenportal_Relaunch_Briefing.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Review_Klimastudie.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `music_mix.m3u` | agent | `_ToReview` | `_ToReview` | PASS | — |
