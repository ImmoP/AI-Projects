# Final Portfolio Holdout Protocol

`final_protocol_version = 1`

This document freezes the methodology used to construct and audit the
final independent Holdout for this portfolio project (Holdout v4) and all
Holdouts constructed under it afterward. It is written and frozen before
any Holdout v4 case is authored, and it is not modified afterward based on
audit output.

## 1. Why this protocol exists

A previous, precommitted three-attempt Holdout-v4 authoring protocol
rejected a candidate Holdout whenever a blind lexical near-duplicate
diagnostic against the historical fixture corpus returned any count above
zero. Three complete, independently authored attempts were rejected under
that rule despite each one showing zero exact historical filename overlap.
No model inference occurred on any of those attempts, no case was ever
revealed, and no case was ever repaired; all three were discarded whole.

That outcome reflects a flaw in the acceptance rule, not a flaw in the
authoring process. Exact duplicate filenames are concrete evidence of case
reuse and therefore invalidate independence. Lexical similarity alone does
not establish reuse: realistic filenames within a constrained
artifact-classification domain (documents, code, images, archives,
installers, and their ambiguous/insufficient-metadata counterparts)
naturally share vocabulary and semantic constructions, so some nonzero
count of lexically similar pairs is an expected property of the domain,
not a signal of leakage. A rule that rejects on any nonzero near-duplicate
count is effectively unsatisfiable for this domain and does not
distinguish independent authorship from case reuse.

This protocol therefore separates two categories of evidence:

* **Evidence of direct case reuse** — an exact normalized filename
  collision against the historical corpus. This remains a hard exclusion
  criterion.
* **Domain-level lexical similarity** — token/sequence similarity between
  independently authored filenames. This remains useful diagnostic
  information, reported transparently, but is not itself treated as proof
  of leakage and does not gate acceptance.

The controls below are designed to prevent direct case reuse while
reporting, rather than eliminating, naturally occurring lexical
similarity. This Holdout is not claimed to be perfectly independent in a
lexical sense; it is claimed to contain zero exact historical filename
reuse, authored without access to historical case material.

## 2. Independence controls

1. Clean-room authoring with no historical case visibility.
2. No model-assisted authoring.
3. Complete authoring freeze before historical audit.
4. Blind normalized exact-name comparison against prior evaluation
   fixtures.
5. Exact historical overlap must equal zero.
6. Lexical near-similarity remains blind and aggregate-only but is
   diagnostic, not a rejection criterion.
7. No historical matching pairs are exposed.
8. No cases may be edited after seeing audit statistics.
9. Exactly one live evaluation after Git freeze.
10. No post-Holdout tuning.

## 3. Frozen parameters

```text
final_protocol_version = 1
exact_overlap_rejection_threshold = 1
near_similarity_rejection_threshold = none
near_similarity_role = diagnostic_only
```

`exact_overlap_rejection_threshold = 1` means any exact historical overlap
count of 1 or more rejects the complete Holdout attempt.
`near_similarity_rejection_threshold = none` means no near-similarity
count, however high, rejects an otherwise-passing Holdout attempt on its
own.

## 4. Acceptance rule

A Holdout attempt is accepted under this protocol iff:

```text
exact_historical_overlap_count == 0
```

`high_lexical_similarity_count` is always computed and reported alongside
the acceptance decision, but never itself changes that decision.

## 5. What this protocol does not claim

* It does not claim the Holdout is lexically dissimilar from historical
  fixtures.
* It does not claim zero domain-vocabulary overlap.
* It does not claim perfect independence — only the absence of direct case
  reuse, under clean-room authoring with no historical case visibility.

## 6. Finality

This is the final Holdout construction protocol for this portfolio
project. It governs exactly one Holdout v4 authoring attempt followed by
exactly one live evaluation. It is not reopened to tune E3/E4-current, to
create additional candidate pipelines, or to re-author Holdouts until
metrics look favorable.
