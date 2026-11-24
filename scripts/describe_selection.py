"""Compatibility entry point for :mod:`s2l_tool_calling.analysis.describe_selection`."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s2l_tool_calling.analysis.describe_selection import (
    composition,
    coverage_slice,
    describe,
    digest,
    main,
    overlap_counts,
    read_rows,
)


if __name__ == "__main__":
    main(default_run=Path("local/runpod/repo/local/course-run"))
