"""Project filesystem paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Temporary compatibility alias for callers that have not migrated yet.
data_dir = DATA_DIR
