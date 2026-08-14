# tidy-agent evaluation — grouping mode

> **Superseded — do not quote these numbers.** This report was produced before the metric definitions were corrected: clustering rows counted files that the extension rules co-located rather than files that were clustered, the unresolved-accuracy row credits abstention like a correct decision, and the grouped-file count is a different set from the group members in this report's own placement table. Kept as a record of the run only; re-run before citing.


- Status: **ok**
- Run mode: **grouping** (`--group`, content reading `False`)
- Metric family: **clustering metrics + category accuracy for ungrouped files**
- Comparable only with other reports whose run mode line is identical.
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Standard run timeout: 300.0 s
- Clustering run timeout: 900.0 s
- Evaluated: 2026-08-10T17:45:29.838384+00:00
- Warm-up: ok in 4.717 s
- Judge: deterministic group-membership and expected-category comparison (no LLM judge)

### Clustering metrics

| Metric | Result |
|---|---:|
| Clustering purity | 90.0% (18/20) |
| Fully co-located expected groups | 100.0% (3/3) |
| Falsely grouped scatter files | 3/6 |
| Invalid proposed folder names | 0 |
| Legacy clustering task tokens | 4056 |
| Compact clustering task tokens | 511 |
| Task token reduction | 87.4% |
| Legacy/compact task characters | 10932/1609 |
| Actual clustering input tokens | 3064 |
| Actual clustering completion tokens | 619 |
| Clustering steps | 1 |
| Clustering latency | 17.270 s |
| All agent input tokens | 5912 |
| Average agent latency | 22.083 s |
| Agent runs | 2 |
| Agent steps | 2 |
| Completion tokens | 1675 |
| Prompt measurement | LiteLLM token_counter (ollama_chat/qwen3.5:4b) |

### Category metrics for ungrouped files

15 file(s) were placed in a semantic group folder and are scored by the clustering metrics above, not here.

| Metric | Result |
|---|---:|
| Category accuracy, ungrouped eligible files | 96.6% (56/58) |
| Category accuracy, ungrouped unresolved files | 91.7% (22/24) |
| Files omitted by the model | 1/24 |
| Invalid-assignment fallbacks | 0/24 |
| Files without any usable proposal | 0/24 |
| `_ToReview/` rate | 15.5% (9/58) |
| Invalid plan entries | 0 |
| Correction rounds | 0 |
| Classification steps | 1 |
| Classification latency | 26.897 s |

Omitted by the model (deterministic `_ToReview/` fallback): `Überweisung_2024`

## Final placement

| File | Expected membership | Destination folder |
|---|---|---|
| `Bachelorarbeit_v3.docx` | `bachelorarbeit` | `Bachelorarbeit` |
| `Bachelorarbeit_v4.docx` | `bachelorarbeit` | `Bachelorarbeit` |
| `Bachelorarbeit_v5.docx` | `bachelorarbeit` | `Bachelorarbeit` |
| `Bachelorarbeit_v6.docx` | `bachelorarbeit` | `Bachelorarbeit` |
| `Kritik_Bachelorarbeit_GPT.md` | `bachelorarbeit` | `Bachelorarbeit` |
| `Pruefbericht_Bachelorarbeit_v4.md` | `bachelorarbeit` | `Bachelorarbeit` |
| `Kundenportal_Relaunch_Briefing.md` | `kundenportal_relaunch` | `Kundenportal_Relaunch` |
| `Kundenportal_Relaunch_Mockup.png` | `kundenportal_relaunch` | `Kundenportal_Relaunch` |
| `Kundenportal_Relaunch_Vertrag.pdf` | `kundenportal_relaunch` | `Kundenportal_Relaunch` |
| `Kundenportal_Relaunch_Notizen.txt` | `kundenportal_relaunch` | `Kundenportal_Relaunch` |
| `Klimastudie_Daten.csv` | `klimastudie` | `Klimastudie` |
| `Klimastudie_Auswertung.ipynb` | `klimastudie` | `Klimastudie` |
| `Klimastudie_Entwurf.docx` | `klimastudie` | `Klimastudie` |
| `Review_Klimastudie.md` | `klimastudie` | `Klimastudie` |
| `Geburtstagskarte_Lena.pdf` | `scatter:Geburtstagskarte_Lena.pdf` | `Documents` |
| `Fahrradverkauf.jpg` | `scatter:Fahrradverkauf.jpg` | `Images` |
| `server_inventory.yaml` | `scatter:server_inventory.yaml` | `Code` |
| `Kochrezept_Risotto.txt` | `scatter:Kochrezept_Risotto.txt` | `Documents` |
| `museum_ticket.pdf` | `scatter:museum_ticket.pdf` | `Documents` |
| `music_mix.m3u` | `scatter:music_mix.m3u` | `_ToReview` |
