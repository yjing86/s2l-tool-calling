"""Run the frozen evaluation under the published November 25 calendar amendment."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import round2_holdout as holdout

AMENDMENT = "docs/round2-opening-amendment.md"
ENTRY = "scripts/round2_amended_opening.py"
ANCHOR = "05e5b906e8e552030f20be9d28bc12687ef49b22"


def amended_release(now):
    holdout.require(now >= datetime(2026, 11, 25, 8, tzinfo=timezone.utc),
                    "Amended opening date has not arrived")
    for path in (AMENDMENT, ENTRY):
        published = subprocess.check_output(
            ["git", "show", f"refs/remotes/origin/main:{path}"], cwd=holdout.REPO)
        holdout.require(published == (holdout.REPO / path).read_bytes(),
                        f"Calendar amendment differs from the published copy: {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "summarize") or "--anchor-commit" in sys.argv:
        sys.exit("Use run or summarize; this amendment fixes the original freeze commit.")
    sys.argv.extend(["--anchor-commit", ANCHOR])
    holdout.require_release = amended_release
    holdout.main()
