"""Compatibility entry point for :mod:`s2l_tool_calling.analysis.summarize_dev_cost`."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s2l_tool_calling.analysis.summarize_dev_cost import (
    NOTES,
    collect,
    main,
    metrics,
    plot,
    read,
    save_rows,
    thresholds,
)


if __name__ == "__main__":
    main()
