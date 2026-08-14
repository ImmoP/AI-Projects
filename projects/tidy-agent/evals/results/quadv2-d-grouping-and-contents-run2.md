# tidy-agent evaluation — grouping mode

- Status: **ok**
- Run mode: **grouping** (`--group`, content reading `True`)
- Metric family: **clustering metrics + category accuracy for ungrouped files**
- Comparable only with other reports whose run mode line is identical.
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Standard run timeout: 900.0 s
- Clustering run timeout: 1200.0 s
- Ground truth: `expected.yaml sha256:dfe0b35b34c6`
- Endpoint: `remote host on a private network (address redacted)`
- Model digest: `2a654d98e6fba55d`
- Quantization: `Q4_K_M`
- Parameter size: `4.7B`
- Context length: `8192`
- Model placement: `GPU/VRAM 3341088193 bytes`
- Evaluated: 2026-08-11T13:11:14.537012+00:00
- Warm-up: ok in 8.051 s
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
| Scatter files sharing a fixed category folder (not a clustering error) | 4/6 |
| Invalid proposed folder names | 0 |
| Legacy clustering task tokens | 4220 |
| Compact clustering task tokens | 533 |
| Task token reduction | 87.4% |
| Legacy/compact task characters | 11370/1662 |
| Actual clustering input tokens | 3087 |
| Actual clustering completion tokens | 653 |
| Clustering steps | 1 |
| Clustering latency | 18.172 s |
| All agent input tokens | 6474 |
| Average agent latency | 41.158 s |
| Agent runs | 2 |
| Agent steps | 2 |
| Completion tokens | 3264 |
| Prompt measurement | LiteLLM token_counter (ollama_chat/qwen3.5:4b) |

### Category metrics for ungrouped files

11 scored file(s) were placed in a semantic group folder and are excluded here; they are listed under *Files excluded from category scoring* below. That count covers files with category ground truth, the same set as the 11 file(s) with clustering ground truth counted above.

| Metric | Result |
|---|---:|
| Category accuracy, ungrouped eligible files | 96.9% (63/65) |
| Category accuracy, ungrouped unresolved files (`_ToReview` accepted) | 93.3% (28/30) |
| Decision rate, unresolved files | 0.0% (0/30) |
| Accuracy on decided files only | n/a (0/0) |
| Abstentions (`_ToReview`), unresolved files | 30/30 |
| Accuracy, strict subset (exactly one accepted category) | 66.7% (4/6) |
| Files omitted by the model | 0/30 |
| Invalid-assignment fallbacks | 0/30 |
| Files without any usable proposal | 30/30 |
| `_ToReview/` rate, all eligible files | 46.2% (30/65) |
| Invalid plan entries | 0 |
| Correction rounds | 0 |
| Classification steps | 1 |
| Classification latency | 64.143 s |
| Classification input tokens | 3387 |
| Classification completion tokens | 2611 |
| Content peeks attempted | 0 |
| Peeks that returned text | 0 |
| Unresolved files peek_file may open | 31/30 |

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
| `Review_Klimastudie.md` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | no-proposal |
| `music_mix.m3u` | agent | `_ToReview` | `_ToReview` | PASS | no-proposal |
| `steuerbescheid_2024` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | no-proposal |
| `sitzungsprotokoll_q3` | agent | `_ToReview` | `Documents`, `_ToReview` | PASS | no-proposal |
| `geraetedump` | agent | `_ToReview` | `_ToReview` | PASS | no-proposal |

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
| `Kritik_Bachelorarbeit_GPT.md` | `bachelorarbeit` | `_ToReview` | category |
| `Pruefbericht_Bachelorarbeit_v4.md` | `bachelorarbeit` | `_ToReview` | category |
| `thesis_draft.docx` | `bachelorarbeit` | `Documents` | category |
| `Kundenportal_Relaunch_Briefing.md` | `kundenportal_relaunch` | `Kundenportal_Relaunch` | group |
| `Kundenportal_Relaunch_Mockup.png` | `kundenportal_relaunch` | `Kundenportal_Relaunch` | group |
| `Kundenportal_Relaunch_Vertrag.pdf` | `kundenportal_relaunch` | `Kundenportal_Relaunch` | group |
| `Kundenportal_Relaunch_Notizen.txt` | `kundenportal_relaunch` | `Kundenportal_Relaunch` | group |
| `Klimastudie_Daten.csv` | `klimastudie` | `Klimastudie` | group |
| `Klimastudie_Auswertung.ipynb` | `klimastudie` | `Klimastudie` | group |
| `Klimastudie_Entwurf.docx` | `klimastudie` | `Klimastudie` | group |
| `Review_Klimastudie.md` | `klimastudie` | `_ToReview` | category |
| `Geburtstagskarte_Lena.pdf` | `scatter:Geburtstagskarte_Lena.pdf` | `Documents` | category |
| `Fahrradverkauf.jpg` | `scatter:Fahrradverkauf.jpg` | `Images` | category |
| `server_inventory.yaml` | `scatter:server_inventory.yaml` | `Code` | category |
| `Kochrezept_Risotto.txt` | `scatter:Kochrezept_Risotto.txt` | `Documents` | category |
| `museum_ticket.pdf` | `scatter:museum_ticket.pdf` | `Documents` | category |
| `music_mix.m3u` | `scatter:music_mix.m3u` | `_ToReview` | category |
