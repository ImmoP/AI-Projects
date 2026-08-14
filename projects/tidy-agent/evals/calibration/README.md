# development-calibration

A reusable **development** fixture for the abstention-calibration cycle
(E0/E1/E2/E3). It is not a Holdout: it may be inspected, rerun, and tuned
against as many times as needed.

47 files, all unresolved by `config/rules.yaml` (no extension, or an
extension the rules do not list), so every one reaches the structured
metadata classifier under evaluation. All files are 0 bytes — this cycle is
metadata-only (no `--read-contents`), so byte content is never sent to a
model and would be inert either way.

`expected.yaml` ground truth was written by hand from the filenames, before
any calibration candidate was run against this fixture, following the policy
in that file's header: `_ToReview` only where a filename genuinely lacks
enough evidence, never mixed in next to a real category as a fallback, and
never assigned merely because a model happens to struggle with a name.

Mixture (see `expected.yaml` section comments): clearly classifiable
unresolved names, abbreviated names, unknown extensions, multilingual names,
names honestly compatible with more than one real category, ordinary
difficult cases, benign instruction-styled filenames (filenames only, no
embedded content, nothing destructive — this tests instruction resistance,
not security), and cases where `_ToReview` is genuinely the only correct
answer.

This fixture, its filenames, and its ground truth were constructed
independently of the frozen 41-file Holdout. No filename, wording, or label
here was copied from it or from any of its individual predictions.

`fixture_manifest.json` is produced by `evals/freeze_datasets.py` and hashes
this fixture the same way the development and holdout fixtures are hashed.
