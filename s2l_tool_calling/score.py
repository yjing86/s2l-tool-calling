"""Score single-call predictions against their references with the Turnstile evaluator."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_experiment import judge, no_duplicates, reject_constant


def grade(record):
    """Adapt the public four-field format to the existing exact-match evaluator."""
    if not isinstance(record, dict):
        raise ValueError("expected a JSON object")
    request_id = record["id"]
    expected = record["expected"]
    prediction = record["prediction"]
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("id must be a nonempty string")
    if not isinstance(prediction, str):
        raise ValueError("prediction must be the decoded response text")
    if not isinstance(expected, dict) or not isinstance(expected.get("arguments"), dict):
        raise ValueError("expected must contain a function name and an arguments object")
    if not isinstance(record["tools"], list):
        raise ValueError("tools must be an array of candidate function schemas")
    names = set()
    for tool in record["tools"]:
        function = tool["function"]
        name = function["name"]
        properties = function["parameters"]["properties"]
        if not isinstance(name, str) or not isinstance(properties, dict):
            raise ValueError("each tool needs a name and a parameters.properties object")
        if name in names or any(not isinstance(schema, dict) for schema in properties.values()):
            raise ValueError("tool names must be unique and property schemas must be objects")
        names.add(name)
    if expected["name"] not in names:
        raise ValueError("expected.name must match one of the candidate tools")
    row = {"id": request_id, "tools": record["tools"], "messages": [{
        "role": "assistant", "content": "", "tool_calls": [{"function": expected}],
    }]}
    return {"id": request_id, **judge(row, prediction)}


def score_file(source, target):
    if target and source.resolve() == target.resolve():
        raise ValueError("choose a different file for --output to preserve the input")
    counts = Counter()
    ids = set()
    with ExitStack() as stack:
        reader = stack.enter_context(source.open("rb"))
        if target:
            writer = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                                 delete=False)
            temp_path = Path(writer.name)
            stack.callback(temp_path.unlink, missing_ok=True)
            stack.enter_context(writer)
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode("utf-8"), object_pairs_hook=no_duplicates,
                                    parse_constant=reject_constant)
                result = grade(record)
                if result["id"] in ids:
                    raise ValueError(f"duplicate id {result['id']!r}")
                ids.add(result["id"])
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            counts[result["reason"]] += 1
            if target:
                writer.write(json.dumps(result, ensure_ascii=False) + "\n")
        if not ids:
            raise ValueError(f"{source}: no input records")
        if target:
            writer.close()
            temp_path.replace(target)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m s2l_tool_calling score", description=__doc__,
        epilog="See examples/calls.jsonl for synthetic examples. Each row needs id, tools, "
               "expected (name and arguments), and prediction (raw Qwen tool-call text).")
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 JSONL with one request per line")
    parser.add_argument("--output", type=Path, help="write one JSON verdict per request to this file")
    args = parser.parse_args(argv)
    try:
        counts = score_file(args.input, args.output)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
    total = sum(counts.values())
    print(f"Exact match: {counts['correct']}/{total} ({100 * counts['correct'] / total:.2f}%)")
    for reason in ("correct", "function", "argument", "count", "parse"):
        print(f"  {reason:<8} {counts[reason]}")
    if args.output:
        print(f"Per-request verdicts: {args.output}")


if __name__ == "__main__":
    main()
