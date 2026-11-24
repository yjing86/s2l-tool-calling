"""Command line entry points for the source checkout."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results/turnstile-single-call.json"
COMMANDS = {
    "demo": ("examples/quickstart.py", "Score a small synthetic example on CPU."),
    "score": ("s2l_tool_calling/score.py", "Score your own single-call model responses on CPU."),
    "select": ("s2l_tool_calling/select_trajectories.py", "Select training IDs from your own proxy-loss trajectories."),
    "results": (None, "Read the published Turnstile scores or a new run's summary."),
    "download": ("scripts/download_inputs.py", "Download and verify the pinned corpus and model weights."),
    "setup": ("scripts/bootstrap_tools.py", "Prepare prerequisites for a fresh GPU checkout."),
    "prepare": ("scripts/prepare_experiment.py", "Construct the split manifests from the pinned inputs."),
    "verify": ("s2l_tool_calling/verify.py", "Check split files and a GPU design against the published run."),
    "run": ("scripts/run_course_experiment.py", "Benchmark and run the GPU study with a specified deadline."),
    "plot": ("scripts/plot_course_results.py", "Plot the primary result from an aggregate JSON file."),
}


def print_results(args):
    parser = argparse.ArgumentParser(prog="python3 -m s2l_tool_calling results",
                                     description=COMMANDS["results"][1])
    parser.add_argument("--input", type=Path, default=PUBLISHED,
                        help="A summarize_course.py internal.json (default: published result).")
    source = parser.parse_args(args).input
    if not source.is_file():
        parser.error(f"Summary file does not exist: {source}")
    summary = json.loads(source.read_text(encoding="utf-8"))
    main = summary["phases"]["main"]
    models = summary["models"]
    base = models[main["reference_base"]]
    print(f"Turnstile single-call | {summary['test_requests']:,} held-out requests")
    print(f"Base model: {base['correct']}/{base['requests']} ({base['accuracy_pct']:.2f}%)")
    print("Seed    Random (A)    Whole answer (B)    Separate fields (C)")
    for seed in sorted(main["paired_by_seed"], key=int):
        scores = [models[main["locked"][f"s{seed}-{arm}"]]["accuracy_pct"] for arm in ("A", "B", "C")]
        print(f"{seed:<6} {scores[0]:>8.2f}% {scores[1]:>17.2f}% {scores[2]:>21.2f}%")
    print("\nDifferences in percentage points, with paired 95% intervals for the fixed trained seeds")
    for label in ("C-B", "C-A", "C-Bprime"):
        contrast = main["across_seed_differences_pp"].get(label)
        if contrast:
            low, high = contrast["fixed_seed_mean_paired_cluster_ci95_pp"]
            print(f"{label:9} {contrast['mean']:+.2f} [{low:+.2f}, {high:+.2f}]"
                  f"   seeds={contrast['seeds']}")
    return 0


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    overview = "\n".join(f"  {name:<10} {description}" for name, (_, description) in COMMANDS.items())
    parser = argparse.ArgumentParser(
        prog="python3 -m s2l_tool_calling", description="Data selection for LLM tool calling",
        epilog=overview + "\n\nUse COMMAND --help for options. Start with demo or results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=COMMANDS, metavar="COMMAND")
    if not args:
        parser.print_help()
        return 0
    command = parser.parse_args(args[:1]).command
    options = args[1:]
    if command == "results":
        return print_results(options)
    if command == "run" and not any(arg in ("-h", "--help") for arg in options):
        if not any(arg == "--deadline" or arg.startswith("--deadline=") for arg in options):
            parser.error("run requires --deadline with your own UTC stop time; see docs/reproduce.md")
    return subprocess.call([sys.executable, str(ROOT / COMMANDS[command][0]), *options])
