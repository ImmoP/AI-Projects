# veto-precision development stress fixture

A reusable **Development** fixture built specifically to refine E4's
deterministic veto. It is not a Holdout: it may be inspected, rerun, and
tuned against as many times as needed. It is also **not independent
validation** — correct terminology for this fixture is "veto-precision
Development stress fixture" or "post-result Development stress fixture,"
never "validation." It was authored after the previous live E3/E4/E5
Development result, using only that result's aggregate findings (never any
Holdout case-level data), and its role is candidate refinement.

72 files, all extensionless and unresolved by `config/rules.yaml`, so every
one reaches whichever unresolved-file mechanism (E3/E4-current/E4-refined)
is under evaluation. All files are 0 bytes — metadata-only throughout.

## Why this fixture exists

The previous live Development result found E4-current's veto had low
precision: 1 true-positive veto against 3 false-positive vetoes (25%
combined), all three false positives caused by the same pattern — two
categories' cue vocabularies co-occurring in a legitimate filename with no
further structure. This fixture is built to measure that specific failure
mode directly and repeatedly, at a scale the two existing Development
fixtures (47 and 66 files, general-purpose) were not designed around.

## Ground-truth distribution

| Category | Count |
|---|---:|
| Code | 12 |
| Documents | 10 |
| Images | 9 |
| Archives | 9 |
| Installers | 8 |
| **Real-category total** | **48 (66.7%)** |
| `_ToReview` | 24 (33.3%) |
| **Total** | **72** |

## Case families

* **Hard negatives for the veto (24 of the 48 real-category cases)** —
  filenames that legitimately contain lexical cues from more than one
  category's vocabulary but still have exactly one correct category once
  the actual file type is considered: a document *about* images, code *for*
  image processing, documentation *of* an installer, source code that
  *manages* archives, a script over photo metadata, code that *builds*
  packages, and further variations distributed across all five categories.
  These measure false-veto burden directly — a precision-oriented veto
  must preserve every one of these.
* **Ordinary classifiable cases (24)** — moderate single-category evidence,
  fresh multilingual examples (Portuguese, Polish, Spanish, Italian,
  Dutch), weak/generic noise words alongside a real cue, category-specific
  compounds, and realistic non-trivial filenames.
* **Explicit category-boundary ambiguity (12 of the 24 `_ToReview` cases)**
  — one case per ordered category pair (plus two repeated pairs with
  different wording), each using an explicit disjunction/uncertainty marker
  over genuinely unresolved two-category semantics.
* **Generic insufficient metadata (6)** — including fresh
  Spanish/Russian/Chinese/Japanese examples.
* **Container/content uncertainty (6)** — metadata that cannot distinguish
  whether the file itself is a container/package or whether the filename
  merely describes contained material.

## Ground-truth principle

Assigned before any inference (none has occurred in this cycle — this is
implementation- and fixture-construction-only), using the same principle as
every prior Development/Holdout fixture in this project: if the permitted
metadata is sufficient for a human evaluator to safely choose one
production category, assign that category; otherwise `_ToReview`. Cases
were never labelled by asking what E3, E4-current, or E4-refined would
predict.

## Independence limitations

Every filename, wording choice, and label here was freshly authored for
this fixture without inspecting `evals/holdout/` or `evals/holdout_v2/`
case-level content, and without copying or reconstructing any case from
`evals/calibration/` or `evals/boundary_calibration/`. This is a process
claim, not a mathematical independence proof, and — unlike a Holdout — that
claim is not this fixture's purpose: it exists to be reused for tuning.

`fixture_manifest.json` is produced by `evals/freeze_datasets.py`, the same
tool used for every other fixture in this project.
