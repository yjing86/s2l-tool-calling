"""Compatibility entry point for :mod:`s2l_tool_calling.analysis.plot_course_results`."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s2l_tool_calling.analysis.plot_course_results import (
    ARMS,
    main,
    render,
)


if __name__ == "__main__":
    main()
