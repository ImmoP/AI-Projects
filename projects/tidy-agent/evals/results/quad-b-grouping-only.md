# tidy-agent evaluation — grouping mode

- Status: **ok**
- Run mode: **grouping** (`--group`, content reading `False`)
- Metric family: **clustering metrics + category accuracy for ungrouped files**
- Comparable only with other reports whose run mode line is identical.
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Standard run timeout: 1800.0 s
- Clustering run timeout: 2400.0 s
- Ground truth: `expected.yaml sha256:c97e0d99d26b`
- Endpoint: `http://127.0.0.1:11434`
- Model digest: `2a654d98e6fba55d`
- Quantization: `Q4_K_M`
- Parameter size: `4.7B`
- Context length: `8192`
- Model placement: `CPU only (size_vram=0)`
- Evaluated: 2026-08-11T10:37:21.074782+00:00
- Warm-up: ok in 27.602 s
- Judge: deterministic group-membership and expected-category comparison (no LLM judge)

### Clustering metrics

Only an accepted semantic group folder counts as a clustering decision. Files the extension rules placed in a fixed category are treated as ungrouped, not as a cluster, so rows about clustering cannot be moved by rule placements. Purity alone is maximised by grouping nothing and must be read together with group cohesion.

| Metric | Result |
|---|---:|
| Clustering purity, files in group folders | 100.0% (7/7) |
| Ground-truth files placed in a group folder | 7/21 |
| Clustering ground truth | 15 group members + 6 scatter files |
| Files in a group folder without clustering ground truth | 0 |
| Destination purity, all evaluated files (includes fixed category folders) | 61.9% (13/21) |
| Fully co-located expected groups | 0.0% (0/3) |
| Scatter files in an accepted group folder | 0/6 |
| Scatter files in a proposed cluster (before executor filtering) | 0/6 |
| …of those, dropped by the minimum-cluster-size filter | 0/6 |
| Scatter files sharing a fixed category folder (not a clustering error) | 4/6 |
| Invalid proposed folder names | 1 |
| Legacy clustering task tokens | 4056 |
| Compact clustering task tokens | 511 |
| Task token reduction | 87.4% |
| Legacy/compact task characters | 10932/1609 |
| Actual clustering input tokens | 3064 |
| Actual clustering completion tokens | 570 |
| Clustering steps | 1 |
| Clustering latency | 368.845 s |
| All agent input tokens | 5955 |
| Average agent latency | 449.866 s |
| Agent runs | 2 |
| Agent steps | 2 |
| Completion tokens | 1656 |
| Prompt measurement | LiteLLM token_counter (ollama_chat/qwen3.5:4b) |

### Category metrics for ungrouped files

7 scored file(s) were placed in a semantic group folder and are excluded here; they are listed under *Files excluded from category scoring* below. That count covers files with category ground truth, the same set as the 7 file(s) with clustering ground truth counted above.

| Metric | Result |
|---|---:|
| Category accuracy, ungrouped eligible files | 95.5% (63/66) |
| Category accuracy, ungrouped unresolved files (`_ToReview` accepted) | 89.3% (25/28) |
| Decision rate, unresolved files | 71.4% (20/28) |
| Accuracy on decided files only | 85.0% (17/20) |
| Abstentions (`_ToReview`), unresolved files | 8/28 |
| Accuracy, strict subset (exactly one accepted category) | 100.0% (5/5) |
| Files omitted by the model | 0/28 |
| Invalid-assignment fallbacks | 0/28 |
| Files without any usable proposal | 0/28 |
| `_ToReview/` rate, all eligible files | 12.1% (8/66) |
| Invalid plan entries | 0 |
| Correction rounds | 0 |
| Classification steps | 1 |
| Classification latency | 530.886 s |
| Classification input tokens | 2891 |
| Classification completion tokens | 1086 |

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
| `Kundenportal_Relaunch_Vertrag.pdf` | rule | `Documents` | `Documents` | PASS | — |
| `Kundenportal_Relaunch_Notizen.txt` | rule | `Documents` | `Documents` | PASS | — |
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
| `Überweisung_2024` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
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
| `backup.gw` | agent | `_ToReview` | `Archives`, `_ToReview` | PASS | — |
| `setup_latest` | agent | `_ToReview` | `Installers`, `_ToReview` | PASS | — |
| `download` | agent | `_ToReview` | `_ToReview`, `Archives`, `Installers` | PASS | — |
| `kundenliste.bak` | agent | `Archives` | `Documents`, `Archives`, `_ToReview` | PASS | — |
| `README` | agent | `Documents` | `Documents`, `Code`, `_ToReview` | PASS | — |
| `projektstand` | agent | `Documents` | `Code`, `Archives`, `Documents`, `_ToReview` | PASS | — |
| `Kritik_Bachelorarbeit_GPT.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Pruefbericht_Bachelorarbeit_v4.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Kundenportal_Relaunch_Briefing.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Review_Klimastudie.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `music_mix.m3u` | agent | `_ToReview` | `_ToReview` | PASS | — |

## Files excluded from category scoring

| File | Group folder | Unresolved by rules |
|---|---|---|
| `Bachelorarbeit_v3.docx` | `Bachelorarbeit` | no |
| `Bachelorarbeit_v4.docx` | `Bachelorarbeit` | no |
| `Bachelorarbeit_v5.docx` | `Bachelorarbeit` | no |
| `Bachelorarbeit_v6.docx` | `Bachelorarbeit` | no |
| `Klimastudie_Daten.csv` | `Klimastudie` | no |
| `Klimastudie_Entwurf.docx` | `Klimastudie` | no |
| `Klimastudie_Auswertung.ipynb` | `Klimastudie` | no |

## Final placement

| File | Expected membership | Destination folder | Placed by |
|---|---|---|---|
| `Bachelorarbeit_v3.docx` | `bachelorarbeit` | `Bachelorarbeit` | group |
| `Bachelorarbeit_v4.docx` | `bachelorarbeit` | `Bachelorarbeit` | group |
| `Bachelorarbeit_v5.docx` | `bachelorarbeit` | `Bachelorarbeit` | group |
| `Bachelorarbeit_v6.docx` | `bachelorarbeit` | `Bachelorarbeit` | group |
| `Kritik_Bachelorarbeit_GPT.md` | `bachelorarbeit` | `Documents` | category |
| `Pruefbericht_Bachelorarbeit_v4.md` | `bachelorarbeit` | `Documents` | category |
| `thesis_draft.docx` | `bachelorarbeit` | `Documents` | category |
| `Kundenportal_Relaunch_Briefing.md` | `kundenportal_relaunch` | `Documents` | category |
| `Kundenportal_Relaunch_Mockup.png` | `kundenportal_relaunch` | `Images` | category |
| `Kundenportal_Relaunch_Vertrag.pdf` | `kundenportal_relaunch` | `Documents` | category |
| `Kundenportal_Relaunch_Notizen.txt` | `kundenportal_relaunch` | `Documents` | category |
| `Klimastudie_Daten.csv` | `klimastudie` | `Klimastudie` | group |
| `Klimastudie_Auswertung.ipynb` | `klimastudie` | `Klimastudie` | group |
| `Klimastudie_Entwurf.docx` | `klimastudie` | `Klimastudie` | group |
| `Review_Klimastudie.md` | `klimastudie` | `Documents` | category |
| `Geburtstagskarte_Lena.pdf` | `scatter:Geburtstagskarte_Lena.pdf` | `Documents` | category |
| `Fahrradverkauf.jpg` | `scatter:Fahrradverkauf.jpg` | `Images` | category |
| `server_inventory.yaml` | `scatter:server_inventory.yaml` | `Code` | category |
| `Kochrezept_Risotto.txt` | `scatter:Kochrezept_Risotto.txt` | `Documents` | category |
| `museum_ticket.pdf` | `scatter:museum_ticket.pdf` | `Documents` | category |
| `music_mix.m3u` | `scatter:music_mix.m3u` | `_ToReview` | category |
