# E3 automatic-error development stress fixture

A reusable **Development** fixture built specifically to generate enough
genuinely difficult E3 automatic decisions to evaluate the frozen
veto-precision evidence threshold. It is not a Holdout: it may be
inspected, rerun, and reused as many times as needed. It is also **not
independent validation** — correct terminology for this fixture is "E3
automatic-error Development stress fixture" or "post-result Development
error stress fixture," never "validation." It was authored after the
corrected 2026-08-12 E4-precision Development cycle, using only that
cycle's aggregate finding (never any Holdout case-level data), and its
role is evidence generation for an already-frozen candidate set — not
candidate tuning.

72 files, all extensionless and unresolved by `config/rules.yaml`, so every
one reaches whichever unresolved-file mechanism (E3/E4-current/E4-refined)
is under evaluation. All files are 0 bytes — metadata-only throughout.

## Why this fixture exists

The corrected 2026-08-12 Development cycle (see
`evals/results/e4-precision-development-20260812-b6095d49ee1d/`) found only
**1 unique E3 automatic error** across 185 Development files, well below
the frozen minimum of **3** required to interpret veto precision/recall
robustly. The `veto_precision_calibration` fixture already answers "can
E4-refined avoid false vetoes on correct E3 decisions?" — but it does not,
by design, generate many wrong E3 automatic decisions to test veto
*recall* against. This fixture asks the complementary question:

> When metadata is semantically deceptive but still plausible enough for
> E3 to automate, does E3 make correlated wrong automatic decisions — and
> can the already-frozen veto strategies catch them?

It deliberately avoids explicit uncertainty markers (`or`, `oder`,
`unclear`, `unklar`, `unknown`, `maybe`, `unbekannt`, `unbestimmt` and
equivalents) in its `_ToReview` cases, because those markers encourage
abstention rather than confident misclassification and so do not exercise
the target failure mode. The goal is not random ambiguity — it is
**plausible confident misinterpretation**.

This fixture does not guarantee that a future live run will observe ≥3
unique E3 automatic errors. It increases the *probability* of observing
them. If a future live run still produces fewer than 3, that is the
result, and the frozen `UNDERPOWERED` verdict applies exactly as before.

## Ground-truth distribution

| Category | Count |
|---|---:|
| Documents | 10 |
| Code | 10 |
| Images | 10 |
| Archives | 9 |
| Installers | 9 |
| **Real-category total** | **48 (66.7%)** |
| `_ToReview` | 24 (33.3%) |
| **Total** | **72** |

## Real-category stress families

* **subject_vs_artifact (16)** — a filename carries a strong cue for
  category A because category A is the file's *subject*, while the actual
  artifact is category B. Correct classification requires distinguishing
  "what the file is about" from "what the file itself is."
* **tool_vs_output (10)** — an action/tool cue points toward the category
  the tool operates *on*, but the file is actually the tool's own
  implementation or documentation (`tool that manipulates X ≠ file of
  type X`).
* **container_lexical_trap (8)** — package/bundle/backup/collection/
  release/distribution vocabulary, deliberately paired with a *non-
  Archives* ground truth throughout, so the vocabulary alone cannot be
  used as a shortcut.
* **installer_driver_trap (8)** — driver/setup/installation/deployment/
  firmware vocabulary, deliberately paired with a *non-Installers* ground
  truth throughout, for the same reason.
* **media_document_trap (6)** — media vocabulary paired with a
  non-Images artifact, balanced by some genuine Images cases that
  themselves carry documentation/workflow vocabulary — the trap is not
  one-directional.

Category × family composition (real-category cases only):

| Family | Documents | Code | Images | Archives | Installers | Total |
|---|---:|---:|---:|---:|---:|---:|
| subject_vs_artifact | 2 | 1 | 4 | 5 | 4 | 16 |
| tool_vs_output | 2 | 5 | 0 | 1 | 2 | 10 |
| container_lexical_trap | 2 | 2 | 2 | 0 | 2 | 8 |
| installer_driver_trap | 3 | 2 | 1 | 2 | 0 | 8 |
| media_document_trap | 1 | 0 | 3 | 1 | 1 | 6 |
| **Total** | **10** | **10** | **10** | **9** | **9** | **48** |

## `_ToReview` stress families

* **latent_dual_role (10)** — two plausible artifact types are present,
  but grammar/context never establishes which one the file actually is.
  No "A or B" wording anywhere.
* **latent_container_content (6)** — metadata cannot establish whether the
  file itself is a container/archive, or is something that merely
  describes/represents the container's contents.
* **dominant_cue_ambiguity (8)** — one category has a strong lexical cue,
  but the file-role relation needed to commit to it is absent, so a
  different interpretation remains equally legitimate. This family is
  specifically designed so a model may confidently pick the dominant cue
  and thereby produce an automatic error.

## Marker discipline

24/24 `_ToReview` cases (exceeding the required ≥20/24) contain no token
matching this fixture's documented explicit-ambiguity-marker vocabulary
(`or`, `oder`, `unclear`, `unklar`, `unknown`, `maybe`, `unbekannt`,
`unbestimmt`) — see `EXPLICIT_AMBIGUITY_MARKERS` in `build_fixture.py` and
the corresponding integrity test.

## Multilingual and morphology stress

51/72 files use a language other than English (German, Spanish,
Portuguese, Dutch, Italian, Polish, French, and one Greek example),
comfortably exceeding the required ≥18. 21 files are tagged
`compound_morphology`, comfortably exceeding the required ≥8 — mostly
German/Dutch filenames built from genuine multi-morpheme compounds (e.g.
`installationsassistent`, `druckertreiber`, `schriftartenverwaltung`),
never malformed spelling or tokenizer-bug exploitation. Every `CASES`
entry's optional `secondary_tags` records these (`multilingual`,
`compound_morphology`) alongside its required primary family; neither the
tags nor the primary family are ever sent to a model.

## Ground-truth principle

Assigned before any inference (none has occurred in this cycle — this is
implementation- and fixture-construction-only), using the same principle
as every prior Development/Holdout fixture in this project: if the
permitted metadata is sufficient for a human evaluator to safely choose
one production category, assign that category; otherwise `_ToReview`.
Cases were never labelled by asking what E3, E4-current, or E4-refined
would predict, and no filename was adjusted after any (nonexistent)
prediction.

## Independence limitations

Every filename, wording choice, and label here was freshly authored for
this fixture without inspecting `evals/holdout/` or `evals/holdout_v2/`
case-level content, and without copying or reconstructing any case (or a
one-character/synonym-only variant of one) from `evals/calibration/`,
`evals/boundary_calibration/`, or `evals/veto_precision_calibration/`.
This is a process claim, not a mathematical independence proof, and —
unlike a Holdout — that claim is not this fixture's purpose: it exists to
be reused for evidence generation and, later, tuning.

`fixture_manifest.json` is produced by `evals/freeze_datasets.py`, the
same tool used for every other fixture in this project.
