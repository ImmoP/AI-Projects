# Engineering Notes

Cross-platform findings from getting this repository's CI green on Ubuntu,
macOS, and Windows. Recorded here because both were platform-specific, not
obvious from reading the failing test alone, and easy to reintroduce by
accident.

## NTFS Alternate Data Streams block a Windows checkout

**Symptom.** `actions/checkout` on `windows-latest` failed before any project
code ran. Locally, the same commit checked out and ran without any error.

**Diagnosis.** A [`tidy-agent`](../projects/tidy-agent/) dev fixture file was
named `report:final` — a legal filename on the POSIX filesystems the project
was authored on. On NTFS, a colon in a filename starts an Alternate Data
Stream (ADS): `report:final` is not a file named "report:final", it is a
stream named `final` attached to a file named `report`. Locally,
`core.protectNTFS=false` was set in `.git/config` at the time, which let Git
write that path as an ADS instead of rejecting it — so the problem was
invisible in the exact environment used to develop the fix. CI runs with
Git's default (`core.protectNTFS=true`), which refuses to check out a path
that resolves to an ADS, so the checkout step failed there while the local
checkout kept succeeding.

**Solution.** The file was removed from Git (`git rm --cached`); its
always-empty content is now recorded in
[`evals/runtime_only_fixture_files.py`](../projects/tidy-agent/evals/runtime_only_fixture_files.py)
so the fixture manifest can still be computed without reading it from disk.
The corresponding manifest-integrity test materializes the file under
`tmp_path` on every platform except Windows, where it is skipped with a
documented reason (NTFS cannot hold the name even under `tmp_path`). Local
config that silently changes checkout behavior (`core.protectNTFS=false`) is
no longer relied on — the fix works with Git's default on every platform,
which is what CI actually uses.

Commit: [`ecf643e`](https://github.com/ImmoP/AI/commit/ecf643e489317f187820ee1bd4542ab59aa0d88e).

## `pathlib` absoluteness is not the same fact on every platform

**Symptom.** A single `tidy-agent` prompt-injection test exercised a
different validation branch in `PlanExecutor` depending on the platform it
ran on, even though the test's input and assertions never changed.

**Diagnosis.** The test's fixture destination started with `/tmp/...`. On
POSIX, `Path("/tmp/...").is_absolute()` is `True`. On Windows, the same
string is drive-relative — `pathlib` treats it as absolute-looking but
without a drive, so `is_absolute()` is `False`. `PlanExecutor` has two
separate rejection paths for an unsafe destination: an early `is_absolute()`
check, and a later post-`resolve()` containment check. The one test was
silently hitting the first path on POSIX and the second on Windows, so it
was really two different tests sharing one name and one set of assertions —
a platform-dependent gap that happened to not matter only as long as both
branches rejected the input for other reasons.

**Solution.** Split into one test per branch, each using a fixture
destination that lands in the same branch on every platform: an absolute
`tmp_path`-based destination for the `is_absolute()` check, and a relative
`..`-based destination for the post-`resolve()` containment check. Neither
test's outcome depends anymore on how a given platform's `pathlib`
implementation classifies a hardcoded path string.

Commit: [`7765cf9`](https://github.com/ImmoP/AI/commit/7765cf9d97b1095be75c0d46ff5a5d40f7a00951).
