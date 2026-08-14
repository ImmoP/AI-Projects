# tidy-agent evaluation — grouping mode

- Status: **error**
- Run mode: **grouping** (`--group`, content reading `True`)
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
- Evaluated: 2026-08-11T11:22:04.980379+00:00
- Warm-up: ok in 27.104 s
- Judge: deterministic group-membership and expected-category comparison (no LLM judge)

### Clustering metrics

Only an accepted semantic group folder counts as a clustering decision. Files the extension rules placed in a fixed category are treated as ungrouped, not as a cluster, so rows about clustering cannot be moved by rule placements. Purity alone is maximised by grouping nothing and must be read together with group cohesion.

| Metric | Result |
|---|---:|
| Clustering purity, files in group folders | n/a (0/0) |
| Ground-truth files placed in a group folder | 0/21 |
| Clustering ground truth | 15 group members + 6 scatter files |
| Files in a group folder without clustering ground truth | 0 |
| Destination purity, all evaluated files (includes fixed category folders) | 0.0% (0/21) |
| Fully co-located expected groups | 0.0% (0/3) |
| Scatter files in an accepted group folder | 0/6 |
| Scatter files in a proposed cluster (before executor filtering) | 0/6 |
| …of those, dropped by the minimum-cluster-size filter | 0/6 |
| Scatter files sharing a fixed category folder (not a clustering error) | 0/6 |
| Invalid proposed folder names | 0 |
| Legacy clustering task tokens | 4056 |
| Compact clustering task tokens | 511 |
| Task token reduction | 87.4% |
| Legacy/compact task characters | 10932/1609 |
| Actual clustering input tokens | 0 |
| Actual clustering completion tokens | 0 |
| Clustering steps | 0 |
| Clustering latency | 0.000 s |
| All agent input tokens | 0 |
| Average agent latency | 0.000 s |
| Agent runs | 0 |
| Agent steps | 0 |
| Completion tokens | 0 |
| Prompt measurement | LiteLLM token_counter (ollama_chat/qwen3.5:4b) |

### Category metrics for ungrouped files

0 scored file(s) were placed in a semantic group folder and are excluded here; they are listed under *Files excluded from category scoring* below. That count covers files with category ground truth, the same set as the 0 file(s) with clustering ground truth counted above.

| Metric | Result |
|---|---:|
| Category accuracy, ungrouped eligible files | 0.0% (0/73) |
| Category accuracy, ungrouped unresolved files (`_ToReview` accepted) | 0.0% (0/28) |
| Decision rate, unresolved files | 0.0% (0/28) |
| Accuracy on decided files only | n/a (0/0) |
| Abstentions (`_ToReview`), unresolved files | 0/28 |
| Accuracy, strict subset (exactly one accepted category) | 0.0% (0/5) |
| Files omitted by the model | 0/28 |
| Invalid-assignment fallbacks | 0/28 |
| Files without any usable proposal | 0/28 |
| `_ToReview/` rate, all eligible files | 0.0% (0/73) |
| Invalid plan entries | 0 |
| Correction rounds | 0 |
| Classification steps | 0 |
| Classification latency | 0.000 s |
| Classification input tokens | 0 |
| Classification completion tokens | 0 |
| Content peeks attempted | 0 |
| Peeks that returned text | 0 |
| Unresolved files peek_file can read at all | 4/28 |

`_ToReview` is an accepted answer for most ambiguous filenames, so the unresolved-accuracy row credits abstention exactly as much as a correct decision — a model that always abstained would score well on it. **Decision rate** and **accuracy on decided files** separate the two and are the rows to compare against the rules-only baseline, which decides nothing.

## Assignments

| File | Mode | Predicted | Accepted | Correct | Fallback |
|---|---|---|---|---|---|
| `Screenshot 2026-08-10 at 10.15.22.png` | rule | `UNASSIGNED` | `Images` | FAIL | — |
| `IMG_4821.HEIC` | rule | `UNASSIGNED` | `Images` | FAIL | — |
| `urlaub_münchen.jpg` | rule | `UNASSIGNED` | `Images` | FAIL | — |
| `logo-final.svg` | rule | `UNASSIGNED` | `Images` | FAIL | — |
| `scan-receipt.webp` | rule | `UNASSIGNED` | `Images` | FAIL | — |
| `Fahrradverkauf.jpg` | rule | `UNASSIGNED` | `Images` | FAIL | — |
| `Kundenportal_Relaunch_Mockup.png` | rule | `UNASSIGNED` | `Images` | FAIL | — |
| `Rechnung_2025-11.pdf` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `Scan_2024-03-11.pdf` | rule | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `Scan_2025-01-09.pdf` | rule | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `dokument.pdf` | rule | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `dokument_final_final.pdf` | rule | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `thesis_draft.docx` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `Budget Q4.xlsx` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `präsentation-köln.pptx` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `notes.txt` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `unbenannt 3.txt` | rule | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `daten.csv` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `Bachelorarbeit_v3.docx` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `Bachelorarbeit_v4.docx` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `Bachelorarbeit_v5.docx` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `Bachelorarbeit_v6.docx` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `Kundenportal_Relaunch_Vertrag.pdf` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `Kundenportal_Relaunch_Notizen.txt` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `Klimastudie_Daten.csv` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `Klimastudie_Entwurf.docx` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `Geburtstagskarte_Lena.pdf` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `Kochrezept_Risotto.txt` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `museum_ticket.pdf` | rule | `UNASSIGNED` | `Documents` | FAIL | — |
| `project-source.zip` | rule | `UNASSIGNED` | `Archives` | FAIL | — |
| `photos_backup.tar.gz` | rule | `UNASSIGNED` | `Archives` | FAIL | — |
| `archive_2024.7z` | rule | `UNASSIGNED` | `Archives` | FAIL | — |
| `logs.tgz` | rule | `UNASSIGNED` | `Archives` | FAIL | — |
| `tidy_agent.py` | rule | `UNASSIGNED` | `Code` | FAIL | — |
| `app.tsx` | rule | `UNASSIGNED` | `Code` | FAIL | — |
| `config.yaml` | rule | `UNASSIGNED` | `Code` | FAIL | — |
| `notebook.ipynb` | rule | `UNASSIGNED` | `Code` | FAIL | — |
| `build.sh` | rule | `UNASSIGNED` | `Code` | FAIL | — |
| `package.json` | rule | `UNASSIGNED` | `Code` | FAIL | — |
| `server_inventory.yaml` | rule | `UNASSIGNED` | `Code` | FAIL | — |
| `Klimastudie_Auswertung.ipynb` | rule | `UNASSIGNED` | `Code` | FAIL | — |
| `installer.dmg` | rule | `UNASSIGNED` | `Installers` | FAIL | — |
| `setup_windows.exe` | rule | `UNASSIGNED` | `Installers` | FAIL | — |
| `app-release.AppImage` | rule | `UNASSIGNED` | `Installers` | FAIL | — |
| `tool.deb` | rule | `UNASSIGNED` | `Installers` | FAIL | — |
| `mystery` | agent | `UNASSIGNED` | `_ToReview` | FAIL | — |
| `ohne_name` | agent | `UNASSIGNED` | `_ToReview` | FAIL | — |
| `meeting-notes` | agent | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `invoice_march` | agent | `UNASSIGNED` | `Documents` | FAIL | — |
| `rechnung_august` | agent | `UNASSIGNED` | `Documents` | FAIL | — |
| `report:final` | agent | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `Überweisung_2024` | agent | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `vertrag.old` | agent | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `final_final_v2` | agent | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `messwerte.dat` | agent | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `vacation-photo` | agent | `UNASSIGNED` | `Images`, `_ToReview` | FAIL | — |
| `urlaub_2025` | agent | `UNASSIGNED` | `Images`, `_ToReview` | FAIL | — |
| `foto_export` | agent | `UNASSIGNED` | `Images`, `_ToReview` | FAIL | — |
| `design.sketch` | agent | `UNASSIGNED` | `Images`, `_ToReview` | FAIL | — |
| `source-code-snippet` | agent | `UNASSIGNED` | `Code`, `_ToReview` | FAIL | — |
| `quellcode_alt` | agent | `UNASSIGNED` | `Code`, `_ToReview` | FAIL | — |
| `data.backup` | agent | `UNASSIGNED` | `Archives`, `_ToReview` | FAIL | — |
| `backup.gw` | agent | `UNASSIGNED` | `Archives`, `_ToReview` | FAIL | — |
| `setup_latest` | agent | `UNASSIGNED` | `Installers`, `_ToReview` | FAIL | — |
| `download` | agent | `UNASSIGNED` | `_ToReview`, `Archives`, `Installers` | FAIL | — |
| `kundenliste.bak` | agent | `UNASSIGNED` | `Documents`, `Archives`, `_ToReview` | FAIL | — |
| `README` | agent | `UNASSIGNED` | `Documents`, `Code`, `_ToReview` | FAIL | — |
| `projektstand` | agent | `UNASSIGNED` | `Code`, `Archives`, `Documents`, `_ToReview` | FAIL | — |
| `Kritik_Bachelorarbeit_GPT.md` | agent | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `Pruefbericht_Bachelorarbeit_v4.md` | agent | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `Kundenportal_Relaunch_Briefing.md` | agent | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `Review_Klimastudie.md` | agent | `UNASSIGNED` | `Documents`, `_ToReview` | FAIL | — |
| `music_mix.m3u` | agent | `UNASSIGNED` | `_ToReview` | FAIL | — |

## Files excluded from category scoring

| File | Group folder | Unresolved by rules |
|---|---|---|

## Final placement

| File | Expected membership | Destination folder | Placed by |
|---|---|---|---|
| `Bachelorarbeit_v3.docx` | `bachelorarbeit` | `UNASSIGNED` | category |
| `Bachelorarbeit_v4.docx` | `bachelorarbeit` | `UNASSIGNED` | category |
| `Bachelorarbeit_v5.docx` | `bachelorarbeit` | `UNASSIGNED` | category |
| `Bachelorarbeit_v6.docx` | `bachelorarbeit` | `UNASSIGNED` | category |
| `Kritik_Bachelorarbeit_GPT.md` | `bachelorarbeit` | `UNASSIGNED` | category |
| `Pruefbericht_Bachelorarbeit_v4.md` | `bachelorarbeit` | `UNASSIGNED` | category |
| `thesis_draft.docx` | `bachelorarbeit` | `UNASSIGNED` | category |
| `Kundenportal_Relaunch_Briefing.md` | `kundenportal_relaunch` | `UNASSIGNED` | category |
| `Kundenportal_Relaunch_Mockup.png` | `kundenportal_relaunch` | `UNASSIGNED` | category |
| `Kundenportal_Relaunch_Vertrag.pdf` | `kundenportal_relaunch` | `UNASSIGNED` | category |
| `Kundenportal_Relaunch_Notizen.txt` | `kundenportal_relaunch` | `UNASSIGNED` | category |
| `Klimastudie_Daten.csv` | `klimastudie` | `UNASSIGNED` | category |
| `Klimastudie_Auswertung.ipynb` | `klimastudie` | `UNASSIGNED` | category |
| `Klimastudie_Entwurf.docx` | `klimastudie` | `UNASSIGNED` | category |
| `Review_Klimastudie.md` | `klimastudie` | `UNASSIGNED` | category |
| `Geburtstagskarte_Lena.pdf` | `scatter:Geburtstagskarte_Lena.pdf` | `UNASSIGNED` | category |
| `Fahrradverkauf.jpg` | `scatter:Fahrradverkauf.jpg` | `UNASSIGNED` | category |
| `server_inventory.yaml` | `scatter:server_inventory.yaml` | `UNASSIGNED` | category |
| `Kochrezept_Risotto.txt` | `scatter:Kochrezept_Risotto.txt` | `UNASSIGNED` | category |
| `museum_ticket.pdf` | `scatter:museum_ticket.pdf` | `UNASSIGNED` | category |
| `music_mix.m3u` | `scatter:music_mix.m3u` | `UNASSIGNED` | category |

Error: `AgentGenerationError: Error in generating model output:
litellm.APIConnectionError: Ollama_chatException - litellm.Timeout: Connection timed out after 600.0 seconds.`
