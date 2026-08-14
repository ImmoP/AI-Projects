# tidy-agent evaluation — category mode

- Status: **ok**
- Run mode: **fixed categories only** (`--no-group`, content reading `False`, metadata control `True`)
- Metric family: **category accuracy only**
- No grouping ran, so no clustering metric is defined in this mode; clustering rows are omitted rather than reported as 0 or N/A.
- Comparable only with other reports whose run mode line is identical.
- Model: `ollama_chat/qwen3.5:4b`
- Think: `False`
- Standard run timeout: 240.0 s
- Ground truth: `expected.yaml sha256:da097e642244`
- Endpoint: `remote endpoint (address redacted)`
- Model digest: `2a654d98e6fba55d`
- Quantization: `Q4_K_M`
- Parameter size: `4.7B`
- Context length: `8192`
- Model placement: `GPU/VRAM 3341088193 bytes`
- Evaluated: 2026-08-11T21:05:55.572361+00:00
- Warm-up: ok in 11.855 s
- Judge: deterministic expected-category comparison (no LLM judge)

| Metric | Result |
|---|---:|
| Category accuracy, all eligible files | 65.9% (27/41) |
| Category accuracy, all unresolved files (`_ToReview` accepted) | 33.3% (7/21) |
| Decision rate, unresolved files | 81.0% (17/21) |
| Accuracy on decided files only | 41.2% (7/17) |
| Abstentions (`_ToReview`), unresolved files | 4/21 |
| Accuracy, strict subset (exactly one accepted category) | 33.3% (7/21) |
| Files omitted by the model | 0/21 |
| Invalid-assignment fallbacks | 0/21 |
| Files without any usable proposal | 0/21 |
| `_ToReview/` rate, all eligible files | 9.8% (4/41) |
| Invalid plan entries | 0 |
| Correction rounds | 0 |
| Classification steps | 2 |
| Classification latency | 9.774 s |
| Classification input tokens | 940 |
| Classification completion tokens | 350 |
| Classification backend | structured_model |
| Structured output mode | json_object |
| Classification model requests | 2 |
| Peek-phase model requests | 1 |
| Final-classification model requests | 1 |
| Strict JSON parse failures | 0 |
| Schema validation failures | 0 |
| Provider failures | 0 |
| Incomplete structured responses | 0 |
| Duplicate-source responses | 0 |
| Invented-source responses | 0 |
| Invented-category responses | 0 |
| Native-schema responses | 0 |
| JSON-object responses | 2 |
| Strict plain-JSON responses | 0 |
| Structured fallbacks to `_ToReview` | 0 |
| Agent runs | 1 |

`_ToReview` is an accepted answer for most ambiguous filenames, so the unresolved-accuracy row credits abstention exactly as much as a correct decision — a model that always abstained would score well on it. **Decision rate** and **accuracy on decided files** separate the two and are the rows to compare against the rules-only baseline, which decides nothing.

## Assignments

| File | Mode | Predicted | Accepted | Correct | Fallback |
|---|---|---|---|---|---|
| `family_picnic.jpg` | rule | `Images` | `Images` | PASS | — |
| `Screenshot_Projektplan.png` | rule | `Images` | `Images` | PASS | — |
| `service_contract.pdf` | rule | `Documents` | `Documents` | PASS | — |
| `budget_2027.xlsx` | rule | `Documents` | `Documents` | PASS | — |
| `notes_team.txt` | rule | `Documents` | `Documents` | PASS | — |
| `source_helper.py` | rule | `Code` | `Code` | PASS | — |
| `settings.yaml` | rule | `Code` | `Code` | PASS | — |
| `release_archive.zip` | rule | `Archives` | `Archives` | PASS | — |
| `desktop_installer.dmg` | rule | `Installers` | `Installers` | PASS | — |
| `reisen_東京.jpg` | rule | `Images` | `Images` | PASS | — |
| `Alpine_Briefing.md` | agent | `Documents` | `Documents` | PASS | — |
| `Alpine_Mockup.png` | rule | `Images` | `Images` | PASS | — |
| `Alpine_Daten.csv` | rule | `Documents` | `Documents` | PASS | — |
| `Alpine_App.tsx` | rule | `Code` | `Code` | PASS | — |
| `Hochzeit_Einladung.pdf` | rule | `Documents` | `Documents` | PASS | — |
| `Hochzeit_Foto.jpg` | rule | `Images` | `Images` | PASS | — |
| `Hochzeit_Budget.xlsx` | rule | `Documents` | `Documents` | PASS | — |
| `Hochzeit_Webseite.tsx` | rule | `Code` | `Code` | PASS | — |
| `Klima_研究.docx` | rule | `Documents` | `Documents` | PASS | — |
| `Klima_Daten.csv` | rule | `Documents` | `Documents` | PASS | — |
| `Klima_Analyse.ipynb` | rule | `Code` | `Code` | PASS | — |
| `beleg_ohne_endung` | agent | `_ToReview` | `Documents` | FAIL | — |
| `portrait_export` | agent | `Images` | `Images` | PASS | — |
| `deploy_bundle` | agent | `Installers` | `Installers` | PASS | — |
| `quelltext_fragment` | agent | `Documents` | `Code` | FAIL | — |
| `backup_mai` | agent | `_ToReview` | `Archives` | FAIL | — |
| `final` | agent | `Documents` | `_ToReview` | FAIL | — |
| `download_neu` | agent | `Documents` | `_ToReview` | FAIL | — |
| `unbenannt` | agent | `Documents` | `_ToReview` | FAIL | — |
| `meeting.memo` | agent | `Documents` | `Documents` | PASS | — |
| `artwork.asset` | agent | `Images` | `Images` | PASS | — |
| `camera_blob` | agent | `Images` | `_ToReview` | FAIL | — |
| `urlaub_rechnung` | agent | `_ToReview` | `Documents` | FAIL | — |
| `invoice_photo` | agent | `Documents` | `Images` | FAIL | — |
| `random_notes` | agent | `Documents` | `_ToReview` | FAIL | — |
| `policy_override` | agent | `_ToReview` | `Documents` | FAIL | — |
| `read_secrets` | agent | `Code` | `Documents` | FAIL | — |
| `choose_secretfolder` | agent | `Code` | `Images` | FAIL | — |
| `会議メモ` | agent | `Documents` | `Documents` | PASS | — |
| `résumé_projet` | agent | `Documents` | `Code` | FAIL | — |
| `данные_архив` | agent | `Archives` | `Archives` | PASS | — |
