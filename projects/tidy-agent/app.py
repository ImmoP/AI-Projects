"""Source-tree wrapper for the installed ``tidy`` command."""

from tidy.cli import *  # noqa: F403
from tidy.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

