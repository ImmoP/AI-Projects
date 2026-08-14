#!/usr/bin/env python3
"""pre-commit hook: reject paths that cannot be checked out on Windows.

Written for this repo instead of relying on pre-commit-hooks' upstream
``check-illegal-windows-names`` id, which multiple open issues report as
either missing from released versions or incomplete (doesn't cover the
character set, only reserved device names) -- see
https://github.com/pre-commit/pre-commit-hooks/issues/1046,
https://github.com/pre-commit/pre-commit-hooks/issues/1091, and
https://github.com/pre-commit/pre-commit-hooks/issues/1189. This exists
because exactly this kind of path (a colon in a filename, read as an NTFS
Alternate Data Stream separator) previously broke `actions/checkout` on
windows-latest before any project code could run.

Checks, per path component:
- The reserved character set '<>:"/\\|?*' and control characters (\\x00-\\x1f)
  -- '/' is checked defensively even though path splitting already handles
  it as a separator.
- Windows device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9), with or
  without an extension.
- Trailing dot or trailing space, which Windows silently strips, so
  "name." and "name" collide.
"""
from __future__ import annotations

import argparse
import re
import sys

_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?$", re.IGNORECASE)


def _violations(path: str) -> list[str]:
    reasons = []
    for component in re.split(r"[\\/]", path):
        if not component:
            continue
        if _BAD_CHARS.search(component):
            reasons.append(f"{path!r}: component {component!r} contains a character illegal on Windows")
        if component.endswith(" ") or component.endswith("."):
            reasons.append(f"{path!r}: component {component!r} ends with a space or dot, illegal on Windows")
        if _RESERVED.match(component):
            reasons.append(f"{path!r}: component {component!r} is a reserved Windows device name")
    return reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args(argv)

    violations = [reason for path in args.filenames for reason in _violations(path)]
    for reason in violations:
        print(reason, file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
