"""Compatibility entry point for :mod:`s2l_tool_calling.analysis.plot_proxy_dynamics`."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s2l_tool_calling.analysis.plot_proxy_dynamics import (
    SERIES,
    THRESHOLD,
    main,
    plot,
    summarize,
)


if __name__ == "__main__":
    main(default_run=Path("local/runpod/repo/local/course-run"))
