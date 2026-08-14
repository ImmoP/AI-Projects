# Architecture

Tidy Agent separates three concerns that are easy to blur in an "AI agent that
touches your filesystem" project: **inference** (proposing a category or
grouping), **control** (validating and authorizing a plan), and **mutation**
(actually moving files). The rule enforced throughout the codebase is:

```text
LLM proposal != authorization != execution
```

## Flow

```mermaid
flowchart TD
    A[Target directory] --> B[Metadata scanner]
    B --> C{"--group?"}

    subgraph INF["Inference / proposal layer"]
        direction TB
        C -- yes --> D[CodeAgent: propose_groups]
        E[Structured classifier]
        E1["Pass 1: classify"]
        E2["Pass 2: classify, order perturbed"]
        E1 --> E
        E2 --> E
    end

    subgraph CTRL["Deterministic control layer"]
        direction TB
        D --> F["Executor: validate clusters<br/>(3+ files, name/collision rules)"]
        C -- no --> G[Extension rules]
        G --> H{Resolved by extension?}
        H -- no --> E1
        H -- no --> E2
        E --> I["E3 agreement gate<br/>(pass1 == pass2?)"]
        I -- disagree / invalid --> R[["_ToReview/"]]
        I -- agree --> J["E4-current veto<br/>(deterministic ambiguity check)"]
        J -- conflict detected --> R
        J -- no conflict --> K[Automatic category]
        F --> L[Combined validated plan]
        H -- yes --> L
        K --> L
        R --> L
    end

    L --> M[Dry-run preview]
    M --> N{Human approval}
    N -- no --> STOP[["Nothing changes"]]

    subgraph MUT["Filesystem mutation layer"]
        direction TB
        N -- yes / --apply --> O["Deterministic executor<br/>(atomic no-clobber moves)"]
        O --> P[Durable journal]
        P --> Q["Undo (optional)"]
    end
```

## Trust boundaries

**Inference layer.** The structured classifier and the grouping `CodeAgent`
only ever produce data: a `{"decisions": [...]}` object for classification, or
`propose_groups(groups)` calls for clustering. Neither is given a filesystem
tool, a move/delete tool, or the deterministic executor. The clustering
agent's Python sandbox has no `open`, no filesystem modules, and no access to
the executor — it can only call the one bound `propose_groups` tool, which
checks transport shape and nothing else.

**Control layer.** Everything the inference layer returns is treated as
untrusted proposal data. Application code — not the model — independently
validates category names, source uniqueness and completeness, group folder
naming, group size (3+ distinct files), and membership safety (no
duplicates, overlaps, unknown, nested, or escaping files). The **E3 agreement
gate** requires two independently reordered classification passes to agree
before a file can automate; the **E4-current veto** then re-checks agreed
decisions for deterministic ambiguity/conflict signals and can still redirect
one to `_ToReview/`. Any malformed, duplicate, invalid, disagreeing, or vetoed
outcome fails closed to `_ToReview/` rather than guessing. See
[Classification strategy](../README.md#classification-strategy) in the README
for what E3 and E4-current actually check.

**Mutation layer.** The executor is the only code path that touches the
filesystem, and it only ever receives the combined, validated plan — never a
raw model response. A dry-run is always produced first and shown to a human;
`--apply` is required to execute anything, and interactive runs additionally
require typing `yes` unless `--yes` is passed. Moves use platform no-clobber,
no-follow rename primitives bound to open directory descriptors, reject path
traversal and symlinks, and re-check source identity before committing. Every
mutation is journaled durably (write-fsync-atomic-replace) before and during
execution, and a completed run can be undone by run id.

Full implementation-level detail — the exact peek-authorization pipeline,
platform rename primitives, journal state machine, and known limitations — is
in the [README's safety and control model](../README.md#safety-and-control-model)
and its "Safety and control model" / "Known semantic-injection limit" subsections.
