# tidy-agent clustering evaluation

> **Superseded — do not quote these numbers.** This report was produced before the metric definitions were corrected: clustering rows counted files that the extension rules co-located rather than files that were clustered, the unresolved-accuracy row credits abstention like a correct decision, and the grouped-file count is a different set from the group members in this report's own placement table. Kept as a record of the run only; re-run before citing.


- Status: **ok**
- Model: `ollama_chat/qwen3.5:4b`
- Grouping: `True`
- Content reading: `False` (part B not implemented)
- Think: `False`
- Standard run timeout: 120.0 s
- Clustering run timeout: 600.0 s
- Evaluated: 2026-08-10T16:11:40.582784+00:00
- Judge: deterministic group-membership comparison (no LLM judge)
- Warm-up: ok in 10.807 s

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
| Clustering latency | 15.638 s |
| All agent input tokens | 66973 |
| Average agent latency | 90.445 s |
| Agent runs | 2 |
| Agent steps | 10 |
| Completion tokens | 5405 |
| Prompt measurement | LiteLLM token_counter (ollama_chat/qwen3.5:4b) |

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
