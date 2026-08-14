# tidy-agent clustering evaluation

> **Superseded — do not quote these numbers.** Produced before the metric definitions were corrected (2026-08-10) and before `thesis_draft.docx` joined the `bachelorarbeit` ground-truth group. Its accuracy rows credit abstention like a correct decision, and any clustering row counts files co-located by extension rule. Kept as a record of the run only; the current results are the `-x3` reports.


- Status: **ok**
- Model: `ollama_chat/qwen3.5:4b`
- Grouping: `False`
- Content reading: `False` (part B not implemented)
- Think: `False`
- Standard run timeout: 120.0 s
- Clustering run timeout: 600.0 s (not used)
- Evaluated: 2026-08-10T16:40:59.887956+00:00
- Judge: deterministic group-membership comparison (no LLM judge)
- Warm-up: ok in 6.181 s

| Metric | Result |
|---|---:|
| Clustering purity | 45.0% (9/20) |
| Fully co-located expected groups | 33.3% (1/3) |
| Falsely grouped scatter files | 5/6 |
| Invalid proposed folder names | 0 |
| Legacy clustering task tokens | 4056 |
| Compact clustering task tokens | 511 |
| Task token reduction | 87.4% |
| Legacy/compact task characters | 10932/1609 |
| Actual clustering input tokens | 0 |
| Actual clustering completion tokens | 0 |
| Clustering steps | 0 |
| Clustering latency | 0.000 s |
| All agent input tokens | 12037 |
| Average agent latency | 45.480 s |
| Agent runs | 1 |
| Agent steps | 3 |
| Completion tokens | 1678 |
| Prompt measurement | LiteLLM token_counter (ollama_chat/qwen3.5:4b) |

## Final placement

| File | Expected membership | Destination folder |
|---|---|---|
| `Bachelorarbeit_v3.docx` | `bachelorarbeit` | `Documents` |
| `Bachelorarbeit_v4.docx` | `bachelorarbeit` | `Documents` |
| `Bachelorarbeit_v5.docx` | `bachelorarbeit` | `Documents` |
| `Bachelorarbeit_v6.docx` | `bachelorarbeit` | `Documents` |
| `Kritik_Bachelorarbeit_GPT.md` | `bachelorarbeit` | `Documents` |
| `Pruefbericht_Bachelorarbeit_v4.md` | `bachelorarbeit` | `Documents` |
| `Kundenportal_Relaunch_Briefing.md` | `kundenportal_relaunch` | `Documents` |
| `Kundenportal_Relaunch_Mockup.png` | `kundenportal_relaunch` | `Images` |
| `Kundenportal_Relaunch_Vertrag.pdf` | `kundenportal_relaunch` | `Documents` |
| `Kundenportal_Relaunch_Notizen.txt` | `kundenportal_relaunch` | `Documents` |
| `Klimastudie_Daten.csv` | `klimastudie` | `Documents` |
| `Klimastudie_Auswertung.ipynb` | `klimastudie` | `Code` |
| `Klimastudie_Entwurf.docx` | `klimastudie` | `Documents` |
| `Review_Klimastudie.md` | `klimastudie` | `Documents` |
| `Geburtstagskarte_Lena.pdf` | `scatter:Geburtstagskarte_Lena.pdf` | `Documents` |
| `Fahrradverkauf.jpg` | `scatter:Fahrradverkauf.jpg` | `Images` |
| `server_inventory.yaml` | `scatter:server_inventory.yaml` | `Code` |
| `Kochrezept_Risotto.txt` | `scatter:Kochrezept_Risotto.txt` | `Documents` |
| `museum_ticket.pdf` | `scatter:museum_ticket.pdf` | `Documents` |
| `music_mix.m3u` | `scatter:music_mix.m3u` | `_ToReview` |
