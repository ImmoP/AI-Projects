# Content-selection root cause — structured-abcd-20260811-25b9ba1e89a1

This analysis uses only the development fixture and persisted benchmark
metadata. It contains no document excerpts.

The old Phase-1 prompt received only the 31 unresolved filenames. It did not
receive extension, byte size, reader eligibility, or an indication that a file
was empty. Python accepted any unresolved filename returned by the model; the
bounded reader then treated a successfully opened empty text file as readable.
Consequently an empty file consumed one of the four technical calls even though
it returned no evidence.

| File | Ground truth | A result | Requested in C/D | Readable | Size | Content available | Informative | C result |
|---|---|---|---:|---:|---:|---:|---:|---|
| `Kritik_Bachelorarbeit_GPT.md` | Documents / review | Documents | 10 | yes | 0 B | no | no | Documents |
| `Pruefbericht_Bachelorarbeit_v4.md` | Documents / review | Documents | 10 | yes | 0 B | no | no | Documents |
| `Review_Klimastudie.md` | Documents / review | Documents | 5 | yes | 0 B | no | no | Documents |
| `sitzungsprotokoll_q3` | Documents / review | review | 0 | yes | 215 B | yes | yes | review |
| `steuerbescheid_2024` | Documents / review | Documents | 0 | yes | 260 B | yes | yes | Documents |
| `geraetedump` | review | Documents | 0 | no (binary) | 2,056 B | no | no | Code |

The remaining 25 unresolved files were also zero-byte fixture placeholders and
were never selected. Filename semantics made the three review-style Markdown
names look attractive, while the missing size signal made their lack of
evidence invisible to the selector. The two non-empty UTF-8 files that could
provide evidence were extensionless and were never requested.

The fix is property-based rather than fixture-specific: Phase 1 receives only
relative source, normalized extension, byte size, and the bounded reader type.
Python excludes zero-byte, pre-known unsupported, and over-limit candidates
before the model response is validated. Unknown and extensionless candidates
remain eligible for the existing strict UTF-8 fallback; binary detection is not
performed speculatively because that would read every candidate outside the
four-call budget.

## Structured-output incompleteness (not changed here)

All 15 incomplete A/B/D responses omitted the same final candidate,
`Überweisung_2024`, while remaining valid JSON objects. C returned the full
list. The stable last-item omission, combined with `json_object` compatibility
mode not enforcing array cardinality, points to long-list/model completion
behaviour or schema ambiguity rather than parsing. No output-limit finish reason
was persisted, so token truncation cannot be asserted. The deterministic
`_ToReview` fallback remains unchanged to avoid modifying a second experimental
variable.
