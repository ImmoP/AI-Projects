# post-holdout-v2 development boundary-calibration

A reusable **development** fixture for the post-Holdout-v2 E4/E5 candidate
cycle. It is not a Holdout: it may be inspected, rerun, and tuned against as
many times as needed, and it is expected to be reused for E4/E5 tuning.

66 files, all extensionless and unresolved by `config/rules.yaml`, so every
one reaches whichever unresolved-file mechanism (E3/E4/E5) is under
evaluation. All files are 0 bytes — this cycle stays metadata-only, so byte
content is never sent to a model and would be inert either way.

## Why a second Development fixture

The existing 47-file `evals/calibration` fixture was already partly used to
select E3. This fixture exists to reduce reliance on that set and, more
importantly, to deliberately enrich coverage of the general failure mode
motivating this cycle: **category-boundary ambiguity and correlated
semantic confusion** — the failure mode Holdout v2 showed survives E3's
two-pass same-category agreement gate (see the root-cause framing in
`evals/post_holdout_candidates.py`'s module docstring; only the aggregate
Holdout v2 conclusion is used anywhere in this cycle, never case-level
Holdout v2 data).

Future live Development evaluation is expected to report this fixture and
the original 47-file `evals/calibration` fixture **both separately and
combined**, never combined-only.

## Ground-truth distribution

| Category | Count |
|---|---:|
| Documents | 12 |
| Archives | 8 |
| Images | 7 |
| Code | 7 |
| Installers | 6 |
| **Real-category total** | **40 (60.6%)** |
| `_ToReview` | 26 (39.4%) |
| **Total** | **66** |

## Case families

* **Single-category semantic support (10)** — moderately clear, not trivial.
* **Weak-vs-strong cue (10)** — one strong category cue plus generic filler
  noise (`final`, `neu`, `kopie`, `v2`, ...) that must not by itself change
  the outcome.
* **Misleading lexical cue (10)** — a word superficially associated with a
  *different* category coexists with the word(s) that actually determine
  the true category. Exists specifically to measure false-veto/false-flag
  risk for E4's deterministic conflict check and any similarly lexical
  mechanism, not to make E4 look good.
* **Multilingual real-category (8)** — fresh Portuguese, Dutch, Polish,
  French, and Italian examples.
* **Prompt-like but legitimate (2)** — an imperative/urgent-styled wrapper
  around genuinely sufficient topical content.
* **Multi-category cue conflict (10, `_ToReview`)** — content-vs-container,
  document-vs-image, code-vs-archive, installer-vs-archive, and
  document-vs-code pairs (2 each), where a human rater genuinely cannot
  safely choose one category.
* **Generic ambiguity (10, `_ToReview`)** — insufficient metadata, including
  fresh French/Spanish/Russian/Chinese/Japanese examples.
* **Prompt-like filenames (6, `_ToReview`)** — a small Development subset of
  instruction-like/authority-impersonating filenames with no legitimate
  topic (filenames only, nothing destructive; this tests instruction
  resistance, not security).

## Ground-truth principle

Assigned before any model inference, using exactly one test: given only the
metadata a production metadata-only classifier is permitted to see, can a
human evaluator safely choose exactly one production category? If yes, that
category; if no, `_ToReview`. No case was labelled `_ToReview` merely
because a model might struggle with it, and no case was given a real
category merely because a plausible keyword appears — see the
"misleading lexical cue" family above, whose entire point is the opposite:
a plausible-but-wrong keyword coexists with a real, correctly-classifiable
file.

## Independence from both Holdouts

Every filename, wording choice, and label in this fixture was freshly
authored for this cycle without opening or inspecting `evals/holdout/` or
`evals/holdout_v2/` case-level content (filenames, expected labels,
rationales, per-file predictions, or gate outcomes). No filename or phrase
here duplicates one from either. This is a process claim, not a
mathematical independence proof.

`fixture_manifest.json` is produced by `evals/freeze_datasets.py`, the same
tool used for every other fixture in this project.
