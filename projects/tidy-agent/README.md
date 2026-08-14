# Tidy Agent

Tidy Agent organizes a directory by scanning file metadata, proposing a move
plan (using deterministic rules and, for ambiguous filenames, a
schema-validated local LLM classifier), and showing that plan to a human for
explicit approval before anything on disk changes. The interesting part isn't
"an LLM sorts your files" — it's the boundary drawn around it: the model
never moves, renames, or deletes anything; it returns structured JSON that a
separate, independently validated, deterministic executor either applies or
routes to `_ToReview/`.

## Highlights

- **Human-approved filesystem automation** — every run is a dry-run by
  default; mutation requires an explicit `--apply` and, interactively, typing
  `yes`.
- **LLM separated from the mutation layer** — classification and grouping
  return structured data only; a dedicated deterministic executor performs
  every move, with no model-generated code ever executed.
- **Fail-closed classification** — a two-pass, order-perturbed agreement gate
  (`E3`) sends anything the model can't consistently agree with itself on to
  `_ToReview/` rather than guessing.
- **Deterministic plan validation** — every move and every semantic group is
  independently checked (path containment, no-clobber, naming rules, group
  size) before it ever reaches the executor.
- **Durable journal and undo** — applies are journaled atomically before
  mutation and can be undone by run id, even after partial failure.
- **Privacy-aware, opt-in content access** — file content is read only when
  explicitly enabled, size-capped, parsed in a timing-bounded subprocess, and
  never sent to a non-local endpoint without a second explicit flag.
- **Reproducible evaluation discipline** — Development and Holdout results
  are never mixed, and the final Holdout ran exactly once under a
  precommitted, frozen protocol.

## Architecture

```mermaid
flowchart TD
    A[Target directory] --> B[Metadata scanner]
    B --> C[Extension rules / optional grouping]
    C -->|resolved| G[Combined validated plan]
    C -->|unresolved| D["Structured classifier<br/>E3 agreement gate"]
    D -->|agree| E["E4-current veto<br/>(evaluation candidate)"]
    D -->|disagree / invalid| R[["_ToReview/"]]
    E -->|conflict| R
    E -->|clear| G
    R --> G
    G --> H[Dry-run preview]
    H --> I{Human approval}
    I -- no --> STOP[["Nothing changes"]]
    I -- yes / --apply --> J[Deterministic executor]
    J --> K[Durable journal]
    K --> L[Undo]
```

The model layer (metadata scanner output, classifier, grouping agent) has no
connection to the filesystem mutation layer. It produces proposals; a
separate validation step decides what's safe to route through; only the
executor, driven by an approved plan, writes to disk. See
[`docs/architecture.md`](docs/architecture.md) for the full diagram, including
the grouping path and validation detail.

## Safety and control model

Safety rules live in `src/tidy/executor.py`, entirely outside the model:

- **Dry-run is the default.** A run without `--apply` only prints a plan.
- **Plans are validated, not trusted.** Every source/destination must resolve
  relative to the selected root (`Path.resolve()` plus a containment check
  rejects traversal); hidden paths, symlinks, and files detected as in use
  are skipped or rejected.
- **No-clobber by construction.** Existing and plan-reserved destinations are
  never selected — collisions get `_1`, `_2`, ... suffixes instead. Moves use
  platform-native atomic no-clobber primitives bound to open directory
  descriptors (`renameat2(RENAME_NOREPLACE)` on Linux,
  `renameatx_np(RENAME_EXCL | RENAME_NOFOLLOW_ANY)` on macOS); cross-filesystem
  moves are rejected rather than falling back to copy/delete.
- **No delete operation exists.** Ambiguous or invalid classifier output
  falls back to `_ToReview/`, never to a guess.
- **Re-validation at apply time.** A source that disappears or changes
  between preview and approval invalidates the plan and forces a fresh
  preview.
- **One root-level lock** shared by apply and undo — a second concurrent Tidy
  process on the same root fails closed instead of racing.
- **Durable journaling.** An `in_progress` journal is written before the
  first mutation; each update is fsynced and atomically replacing, moves are
  persisted incrementally, and the run only becomes `committed` after
  `commit_pending` succeeds.
- **Undo is a first-class operation**, not an afterthought: it revalidates
  paths, replays moves in reverse with the same atomic no-clobber primitive,
  and tracks its own state machine (`undo_in_progress` → `partially_undone` /
  `undone`) so failed entries stay retryable by run id.

These are hardened, explicitly-checked controls, not an absolute guarantee —
see [Limitations](#limitations) for the boundaries that remain, including a
documented, deliberately-not-closed semantic prompt-injection case
(`Steuerunterlagen_2024/`-style filename manipulation), platform-specific
race windows on POSIX, and Windows' lack of directory-relative rename/fsync
primitives.

### Content access is a separate, opt-in control

`--read-contents` lets the classifier read the beginning of files extension
rules couldn't resolve; it is off by default. Reading is bound to an exact
allowlist of unresolved filenames, capped at 1,500 characters (first two PDF
pages), parsed for `.pdf`/`.docx` in a subprocess with a 3-second wall
timeout, and budgeted at exactly four reads per classification task — a fifth
request is refused before any filesystem access. A loopback model endpoint
needs only `--read-contents`; any other endpoint additionally requires
`--allow-remote-content`, checked before scanning, classifier construction,
or any file read. Extracted text is wrapped in explicit "untrusted data"
markers in the prompt — a labeling convention, not a sandboxing guarantee (see
[Limitations](#limitations)).

## Classification strategy

Unresolved filenames (those extension rules can't place) go through
**`E3`**, a two-pass agreement gate: the identical model, prompt, and schema
classify the same files twice, with source order reversed on the second
pass. A file automates only when both passes independently return the same
category; disagreement, an invalid response, or a review verdict on either
pass falls back to `_ToReview/`. This favors classification safety over
maximum automation coverage by construction, and it is the strategy actually
wired into production (`StructuredClassifier.classify_with_agreement_gate` in
[`src/tidy/classification.py`](src/tidy/classification.py)).

**`E4-current`** is a separate, deterministic ambiguity/conflict veto layered
on top of `E3`'s automatic decisions, using a small filename-cue vocabulary
to catch cases where both passes agreed but landed on a category that
conflicts with a strongly competing cue. It adds no extra model request.
**`E4-current` is an evaluation candidate, not the production default** — it
was selected as the sole candidate for the final independent Holdout (see
[Evaluation](#evaluation) below), but that Holdout's inconclusive result was
not used to change what ships. Production classification is unmodified `E3`.

Full candidate history (`E0`–`E5`), Development metrics, and the final
Holdout writeup live in [`docs/evaluation.md`](docs/evaluation.md).

## Quickstart

Requires Python 3.10+ and, for the default local model, a running
[Ollama](https://ollama.com) instance.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

`.env` (or `--model`) selects the model; the default is a local Ollama model:

```dotenv
MODEL_ID=ollama_chat/qwen3.5:4b
API_BASE=http://localhost:11434
```

Any model `LiteLLMModel` supports can be used instead by editing `MODEL_ID`
or passing `--model`. Pull and start the configured model before running an
agent-backed plan, e.g. `ollama pull qwen3.5:4b`.

Try it on a disposable directory first, never an important one:

```bash
mkdir -p ~/tidy-demo && cd ~/tidy-demo
touch invoice.pdf photo.jpg archive.zip mystery_file

# Dry-run is always the default — nothing moves yet
tidy --path ~/tidy-demo

# Same thing, purely deterministic, no model call:
tidy --path ~/tidy-demo --no-agent

# Apply after reviewing the printed plan (asks for a typed "yes")
tidy --path ~/tidy-demo --apply

# Undo the most recent run
tidy --undo
```

Other supported flags:

```bash
tidy --path ~/tidy-demo --model ollama_chat/qwen3.5:4b --no-think
tidy --path ~/tidy-demo --group              # opt-in semantic clustering
tidy --path ~/tidy-demo --read-contents      # opt-in content-assisted classification
tidy --path ~/tidy-demo --read-contents --allow-remote-content
tidy --path ~/tidy-demo --apply --yes        # skip the interactive confirmation
tidy --undo 20260810T102030.123456Z-a1b2c3d4 # undo a specific run id
```

Run `tidy --help` (or `python -m tidy.cli --help`) for the complete, current
flag list — that is the source of truth, not this file.

## Demo

A two-minute walkthrough of the core loop, using `--no-agent` so it runs
without needing a model available:

```bash
mkdir -p ~/tidy-demo && cd ~/tidy-demo
touch invoice.pdf photo.jpg archive.zip mystery_file

tidy --path ~/tidy-demo --no-agent
```

```text
[_ToReview]
ORIGIN  SOURCE        DESTINATION             REASON
------  ------------  ----------------------  ------------------------------------------
rule    mystery_file  _ToReview/mystery_file  No matching extension rule; agent disabled

[Archives]
ORIGIN  SOURCE       DESTINATION           REASON
------  -----------  --------------------  ------------------------------------
rule    archive.zip  Archives/archive.zip  Matched extension rule for Archives/

[Documents]
ORIGIN  SOURCE       DESTINATION            REASON
------  -----------  ---------------------  -------------------------------------
rule    invoice.pdf  Documents/invoice.pdf  Matched extension rule for Documents/

[Images]
ORIGIN  SOURCE     DESTINATION       REASON
------  ---------  ----------------  ----------------------------------
rule    photo.jpg  Images/photo.jpg  Matched extension rule for Images/

Dry-run complete: planned=4
```

That's the proposal — nothing has moved. Approve it explicitly:

```bash
tidy --path ~/tidy-demo --no-agent --apply --yes
find ~/tidy-demo -type f   # Archives/, Documents/, Images/, _ToReview/
```

The move is journaled before it happens, and it's reversible:

```bash
tidy --undo
find ~/tidy-demo -type f   # back to the flat directory
```

`mystery_file` has no extension the rules recognize, so with `--no-agent` it
goes to `_ToReview/` rather than being guessed at. Dropping `--no-agent` (the
default) routes files like that through the classifier described in
[Classification strategy](#classification-strategy) instead — an ambiguous
name is still as likely to land in `_ToReview/` as in a category, by design.
`--group` (semantic clustering across extensions) and `--read-contents`
(opt-in content-assisted classification) build on this same
preview-then-approve loop; see [Quickstart](#quickstart) for the flags.

## Tests

- **Smoke** (`pytest -m smoke`, or `python scripts/smoke.py` from the repo
  root): does the pipeline run at all? A handful of tests that call
  `app.main()`/`app.build_combined_plan()` end-to-end -- dry-run, apply
  with confirmation, and the rules-only fallback path -- with `--no-agent`
  so no model call is involved.
- **Unit** (`pytest`, the full suite, 638 tests): is the logic correct?
  Classification validation, the executor's move/undo journal, prompt-
  injection resistance, the `E3`/`E4` calibration harness, tool sandboxing,
  and more. One test
  (`test_agent.py::test_request_timeout_is_explicit_and_configurable`) is
  marked `slow` (>2s, model-client construction overhead) and excluded from
  the default CI run (`pytest -m "not slow"`).
- **Eval** (not in CI): how good is the classifier? Development calibration
  and the one-time Holdout protocol -- see [Evaluation](#evaluation) below.

```bash
pytest -m smoke     # fast end-to-end paths only
pytest -m "not slow"  # everything except the one >2s test (what CI runs)
pytest               # the full suite
```

## Evaluation

**Development.** Iterative calibration across several fixtures selected `E3`
(two-pass agreement gate) as the production classifier — 0 incorrect
automatic classifications out of 47 files in the deciding calibration run —
and later selected `E4-current` (a deterministic veto layered on `E3`) as the
sole candidate for the final Holdout. All Development numbers are labeled and
reported separately from Holdout results; see
[`docs/evaluation.md`](docs/evaluation.md#development-evidence) for the full
table with links to every underlying report.

**Final Holdout v4: `PARTIAL_INCONCLUSIVE`.** The final, one-time,
150-case independent Holdout against `E3`+`E4-current` completed both
provider requests with zero provider errors and zero JSON parse failures,
and both the `E3` gate and `E4-current` veto ran to completion — but one
response failed structured-output schema validation (it was incomplete and
named a source file that wasn't part of the request). The protocol's
precommitted validity rule treats any schema-contract failure on a required
response as invalidating the whole evaluation, so `evaluation_valid = false`
and the run is not a usable accuracy benchmark. Because the evaluation was invalidated,
no final Holdout accuracy or generalization estimate is reported.
 Per this project's one-time-Holdout protocol, the
run is not repeated. Full root-cause analysis, the authoritative status
fields, and what can and cannot be concluded from this run are in
[`docs/evaluation.md`](docs/evaluation.md#independent-final-holdout-v4).

### Holdout protocol: runtime state vs. permanent record

Each one-time Holdout (`evals/holdout_v3/`, `evals/holdout_v4/`) has two
separate files that must not be confused:

- **`CONSUMED.json`** — a local, gitignored runtime marker. The frozen
  runners (`evals/run_holdout_v3_e4.py`, `evals/run_holdout_v4_e4.py`)
  check `.exists()` on this file before permitting a live run and write it
  durably, atomically, immediately before the first measured request. It
  is environment state, not repo state — a fresh checkout must never ship
  as "already consumed", so it's never committed. It also must never be
  deleted or reset on a machine where the Holdout actually ran; doing so
  would let an already-consumed Holdout be run again.
- **`consumption_record.json`** — the committed, permanent record that a
  consumption event happened: date, commit, result directory, and
  designation (e.g. `PARTIAL_INCONCLUSIVE`, or `INTERRUPTED, NO USABLE
  EVIDENCE`). It documents history; it is deliberately *not* read by any
  runner as a consumption gate, so committing it can never itself block or
  permit a run.

`evals/holdout_v3/consumption_record.json` and
`evals/holdout_v4/consumption_record.json` are the audit trail for both
consumption events referenced above.

## Engineering decisions

- **The LLM cannot mutate files** because structured classification/grouping
  output and filesystem execution are different code paths with different
  trust levels; collapsing them would make every prompt-injection risk a
  filesystem risk.
- **Review is preferable to forced classification** because a wrong
  `_ToReview/` costs a human a glance, while a wrong automatic move costs
  them a missing file — the asymmetry is the whole reason `E3` exists.
- **`E3` uses agreement, not confidence,** because model-reported confidence
  isn't independently verifiable; agreement between two independently
  reordered passes is a cheap, deterministic, checkable proxy.
- **`E4-current`'s veto is deterministic** (filename-cue rules, not a third
  model call) so it adds a control-layer check rather than more variance
  from another inference request.
- **Content access is opt-in and separately gated for remote endpoints**
  because reading file content is a materially different privacy posture
  than reading filenames, and sending that content off-device is a further
  step again.
- **Evaluation validity is fail-closed** because a completed run and a valid
  run are not the same claim — Holdout v4 demonstrates the harness enforcing
  that distinction on itself.

## Project status

Implementation is complete for this portfolio's scope, and evaluation is
closed: the final Holdout has been consumed exactly once and is not rerun,
Development selection is closed, and there is no post-Holdout tuning. The
current focus is demonstration and documentation, not a production
deployment — there is no persistence layer, multi-user support, or
scheduled/background operation, and the tool is intended to be run
interactively against one directory at a time.

## Repository layout

```text
src/tidy/
  cli.py             CLI entry point (dry-run, apply, undo)
  agent.py           Classifier + grouping agent construction
  classification.py  Structured classification, E3 agreement gate
  rules.py           Extension-rule loading and matching
  tools.py           peek_file / propose_groups tool implementations
  executor.py        Deterministic plan validation, execution, journal, undo
  content_parser.py  Sandboxed PDF/DOCX/text extraction
config/rules.yaml     Category extensions and exclusion patterns
evals/                Evaluation harnesses, fixtures, and results
  results/            Committed Development and Holdout reports
  holdout_v4/          Frozen final Holdout artifacts (consumed)
tests/                 pytest suite
docs/
  architecture.md      Full architecture diagram and trust boundaries
  evaluation.md         Full evaluation history and final Holdout writeup
```

## Limitations

- **Filename/metadata ambiguity is real.** Many filenames genuinely don't
  encode enough information to classify correctly; `_ToReview/` exists
  because abstention is often the right answer, not a fallback of last
  resort.
- **The default model is small and local** (`qwen3.5:4b` via Ollama).
  Accuracy and calibration numbers throughout this project are specific to
  that model and are not claimed to generalize to other providers or model
  sizes.
- **The production classifier is conservative by design** — `E3`'s agreement
  gate trades automation coverage for a lower incorrect-automation rate, so
  expect a meaningful share of files to land in `_ToReview/` rather than a
  category.
- **Optional content access has real privacy and cost trade-offs.** It's
  opt-in and capped, but reading file content is a stronger action than
  reading a filename, and non-loopback endpoints require a second explicit
  flag for exactly that reason.
- **The final independent Holdout is inconclusive**, not passing or failing
  — a schema-contract violation on one of two required responses caused the
  precommitted validity rule to invalidate the run before any accuracy claim
  could be made. See [Evaluation](#evaluation).
- **A known, documented semantic-injection limit exists:** the executor
  rejects every *formal* safety violation (path traversal, symlinks, invalid
  names) but cannot judge whether a formally valid destination is
  *semantically* correct — untrusted file content that plausibly argues for
  a folder name can still produce a formally valid plan. Human approval of
  the dry-run remains the actual backstop for this case; it is not closed by
  prompt framing.
- **The pre-move identity re-check is currently unverified on Windows, and
  mutation is refused there as a result.** `--apply`/`--undo`'s core
  TOCTOU/tamper-detection guarantee (`SourceMetadata` full-field equality in
  `src/tidy/executor.py`) was found to spuriously disagree on NTFS between
  two `stat()` calls on a file nobody touched (`st_ctime_ns` observed to
  shift between a `stat()` taken immediately after a write-and-close and one
  taken microseconds later on the same, untouched file). Since that's
  exactly the failure mode the check exists to catch, it cannot currently
  tell a real change from filesystem timing noise on this platform, so
  `PlanExecutor.execute()` and `undo()` now raise `UnverifiedPlatformError`
  and refuse to mutate at all there rather than run with what looks like an
  active safety guarantee but isn't verified to be one. Dry-run is
  unaffected on every platform. This was only discovered now because a
  separate Windows checkout failure (an illegal `:` in a fixture filename)
  had blocked Windows CI from ever reaching these tests before. Tracked as a
  follow-up (reproduction steps: create a file, `path.stat()` it, then
  immediately `os.fstat(os.open(path, os.O_RDONLY))` it on Windows and
  compare `st_ctime_ns` — see `tests/test_executor.py`'s module docstring
  and `_METADATA_IDENTITY_VERIFICATION_RELIABLE` in `src/tidy/executor.py`).
- **No claim of universal filesystem safety or classification accuracy is
  made anywhere in this project.** Every safety control above is described
  as hardened, validated, or fail-closed — not as a guarantee.

## Development

```bash
pytest -v
python -m compileall -q src app.py evals tests
```

The test suite uses `tmp_path` fixtures; it never operates on a real user
directory.

### Windows checkout note

One dev-fixture case, `report:final` (a real, empty case in
`evals/expected.yaml`), used a colon in its filename — legal on the
filesystems this project was authored on, but illegal on NTFS, where a
colon starts an Alternate Data Stream. `actions/checkout` on
`windows-latest` failed on it before any project code ran. Fix: the file is
no longer committed; `evals/runtime_only_fixture_files.py` records its
(always-empty) expected content so `evals/freeze_datasets.py` can still
compute a manifest entry for it without reading it from disk, and
`tests/test_abcd.py::test_committed_dev_manifest_matches_every_canonical_file`
materializes it under `tmp_path` at test time on every platform except
Windows, where that test is skipped with a documented reason.

Verifying this also surfaced a second, unrelated Windows issue: two other
dev-fixture files were getting LF rewritten to CRLF on checkout
(`core.autocrlf=true`, the `windows-latest` default), which silently broke
their manifest hash check. That's fixed by the repo-root `.gitattributes`
(`eol=lf` pinned for `evals/**/fixture/**` and `evals/**/expected.yaml`,
with the two frozen fixtures that deliberately contain binary/CRLF bytes
explicitly excepted).
