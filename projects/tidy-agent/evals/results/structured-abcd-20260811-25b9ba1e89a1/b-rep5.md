# tidy-agent evaluation — grouping mode

- Status: **ok**
- Run mode: **grouping** (`--group`, content reading `False`)
- Metric family: **clustering metrics + category accuracy for ungrouped files**
- Comparable only with other reports whose run mode line is identical.
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Standard run timeout: 240.0 s
- Clustering run timeout: 600.0 s
- Ground truth: `expected.yaml sha256:dfe0b35b34c6`
- Endpoint: `remote endpoint (address redacted)`
- Model digest: `2a654d98e6fba55d`
- Quantization: `Q4_K_M`
- Parameter size: `4.7B`
- Context length: `8192`
- Model placement: `GPU/VRAM 3341088193 bytes`
- Evaluated: 2026-08-11T19:33:38.289305+00:00
- Warm-up: ok in 19.052 s
- Judge: deterministic group-membership and expected-category comparison (no LLM judge)

### Clustering metrics

Only an accepted semantic group folder counts as a clustering decision. Files the extension rules placed in a fixed category are treated as ungrouped, not as a cluster, so rows about clustering cannot be moved by rule placements. Purity alone is maximised by grouping nothing and must be read together with group cohesion.

| Metric | Result |
|---|---:|
| Clustering purity, files in group folders | 100.0% (11/11) |
| Ground-truth files placed in a group folder | 11/21 |
| Clustering ground truth | 15 group members + 6 scatter files |
| Files in a group folder without clustering ground truth | 0 |
| Destination purity, all evaluated files (includes fixed category folders) | 76.2% (16/21) |
| Fully co-located expected groups | 33.3% (1/3) |
| Scatter files in an accepted group folder | 0/6 |
| Scatter files in a proposed cluster (before executor filtering) | 0/6 |
| …of those, dropped by the minimum-cluster-size filter | 0/6 |
| Scatter files sharing a fixed category folder (not a clustering error) | 5/6 |
| Invalid proposed folder names | 0 |
| Groups proposed / accepted / discarded / rejected | 3 / 3 / 0 / 0 |
| Rule-resolved files overridden by grouping | 10 |
| Unresolved files handled by grouping | 1 |
| Legacy clustering task tokens | 4220 |
| Compact clustering task tokens | 533 |
| Task token reduction | 87.4% |
| Legacy/compact task characters | 11370/1662 |
| Actual clustering input tokens | 3020 |
| Actual clustering completion tokens | 530 |
| Clustering steps | 1 |
| Clustering latency | 13.950 s |
| All agent input tokens | 3304 |
| Average agent latency | 11.201 s |
| Agent runs | 2 |
| Agent steps | 2 |
| Completion tokens | 863 |
| Prompt measurement | LiteLLM token_counter (ollama_chat/qwen3.5:4b) |

### Category metrics for ungrouped files

11 scored file(s) were placed in a semantic group folder and are excluded here; they are listed under *Files excluded from category scoring* below. That count covers files with category ground truth, the same set as the 11 file(s) with clustering ground truth counted above.

| Metric | Result |
|---|---:|
| Category accuracy, ungrouped eligible files | 90.8% (59/65) |
| Category accuracy, ungrouped unresolved files (`_ToReview` accepted) | 80.0% (24/30) |
| Decision rate, unresolved files | 83.3% (25/30) |
| Accuracy on decided files only | 76.0% (19/25) |
| Abstentions (`_ToReview`), unresolved files | 5/30 |
| Accuracy, strict subset (exactly one accepted category) | 66.7% (4/6) |
| Files omitted by the model | 1/30 |
| Invalid-assignment fallbacks | 0/30 |
| Files without any usable proposal | 0/30 |
| `_ToReview/` rate, all eligible files | 7.7% (5/65) |
| Invalid plan entries | 0 |
| Correction rounds | 0 |
| Classification steps | 1 |
| Classification latency | 8.451 s |
| Classification input tokens | 284 |
| Classification completion tokens | 333 |
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
| `meeting-notes` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `invoice_march` | agent | `Documents` | `Documents` | PASS | — |
| `rechnung_august` | agent | `Documents` | `Documents` | PASS | — |
| `report:final` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | — |
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
| `projektstand` | agent | `Documents` | `Code`, `Archives`, `Documents`, `_ToReview` | PASS | — |
| `Kritik_Bachelorarbeit_GPT.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Pruefbericht_Bachelorarbeit_v4.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `Review_Klimastudie.md` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `music_mix.m3u` | agent | `Images` | `_ToReview` | FAIL | — |
| `steuerbescheid_2024` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `sitzungsprotokoll_q3` | agent | `Documents` | `Documents`, `_ToReview` | PASS | — |
| `geraetedump` | agent | `Documents` | `_ToReview` | FAIL | — |

## Files excluded from category scoring

| File | Group folder | Unresolved by rules |
|---|---|---|
| `Kundenportal_Relaunch_Mockup.png` | `Kundenportal_Relaunch` | no |
| `Bachelorarbeit_v3.docx` | `Bachelorarbeit` | no |
| `Bachelorarbeit_v4.docx` | `Bachelorarbeit` | no |
| `Bachelorarbeit_v5.docx` | `Bachelorarbeit` | no |
| `Bachelorarbeit_v6.docx` | `Bachelorarbeit` | no |
| `Kundenportal_Relaunch_Vertrag.pdf` | `Kundenportal_Relaunch` | no |
| `Kundenportal_Relaunch_Notizen.txt` | `Kundenportal_Relaunch` | no |
| `Klimastudie_Daten.csv` | `Klimastudie` | no |
| `Klimastudie_Entwurf.docx` | `Klimastudie` | no |
| `Klimastudie_Auswertung.ipynb` | `Klimastudie` | no |
| `Kundenportal_Relaunch_Briefing.md` | `Kundenportal_Relaunch` | yes |

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
| `Kundenportal_Relaunch_Briefing.md` | `kundenportal_relaunch` | `Kundenportal_Relaunch` | group |
| `Kundenportal_Relaunch_Mockup.png` | `kundenportal_relaunch` | `Kundenportal_Relaunch` | group |
| `Kundenportal_Relaunch_Vertrag.pdf` | `kundenportal_relaunch` | `Kundenportal_Relaunch` | group |
| `Kundenportal_Relaunch_Notizen.txt` | `kundenportal_relaunch` | `Kundenportal_Relaunch` | group |
| `Klimastudie_Daten.csv` | `klimastudie` | `Klimastudie` | group |
| `Klimastudie_Auswertung.ipynb` | `klimastudie` | `Klimastudie` | group |
| `Klimastudie_Entwurf.docx` | `klimastudie` | `Klimastudie` | group |
| `Review_Klimastudie.md` | `klimastudie` | `Documents` | category |
| `Geburtstagskarte_Lena.pdf` | `scatter:Geburtstagskarte_Lena.pdf` | `Documents` | category |
| `Fahrradverkauf.jpg` | `scatter:Fahrradverkauf.jpg` | `Images` | category |
| `server_inventory.yaml` | `scatter:server_inventory.yaml` | `Code` | category |
| `Kochrezept_Risotto.txt` | `scatter:Kochrezept_Risotto.txt` | `Documents` | category |
| `museum_ticket.pdf` | `scatter:museum_ticket.pdf` | `Documents` | category |
| `music_mix.m3u` | `scatter:music_mix.m3u` | `Images` | category |
