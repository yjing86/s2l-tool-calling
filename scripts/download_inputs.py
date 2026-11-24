"""Compatibility entry point for :mod:`s2l_tool_calling.data.download_inputs`."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s2l_tool_calling.data.download_inputs import (
    CURL,
    ROOT,
    SNAPSHOTS,
    fetch,
    main,
    matches,
)


if __name__ == "__main__":
    main()
