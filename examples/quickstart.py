"""Inspect field spans and score synthetic tool calls with the project's evaluator.

From the repository root: python3 examples/quickstart.py
The four requests and both sets of responses are hand-written teaching fixtures.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_experiment import judge
from summarize_course import paired
from verify_tool_call_loss import DEMO, answer_spans, gold_calls


def render_call(function, order_id):
    return (
        f"<tool_call>\n<function={function}>\n"
        f"<parameter=order_id>\n{order_id}\n</parameter>\n"
        "</function>\n</tool_call>"
    )


def order_request(order_id):
    row = deepcopy(DEMO)
    row["id"] = f"order-{order_id}"
    row["messages"][0]["content"] = f"Please track order {order_id}."
    gold_calls(row)[0]["function"]["arguments"]["order_id"] = order_id
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-calls", action="store_true",
                        help="show the hand-written responses for each request")
    args = parser.parse_args()
    print("Synthetic walkthrough: four hand-written requests and two invented response sets.\n")
    row = DEMO
    gold = gold_calls(row)
    call = gold[0]["function"]
    eos = "<|im_end|>"
    text = render_call(call["name"], call["arguments"]["order_id"]) + eos
    spans, _ = answer_spans(text, 0, gold, eos)
    print(row["messages"][0]["content"])
    for field, label in [("function", "Function name"), ("value", "Argument value")]:
        print(f"  {label} span: {', '.join(text[start:end] for start, end in spans[field])}")
    print("  The scorer can track loss on these spans separately during a real run.\n")

    # Each row contains the expected order ID, X's call and Y's call.
    cases = [
        ("4821", ("track_order", "4821"), ("track_order", "4821")),
        ("5503", ("cancel_order", "5503"), ("track_order", "5503")),
        ("6107", ("track_order", "6107"), ("track_order", "6017")),
        ("7210", ("track_order", "7201"), ("track_order", "7210")),
    ]
    scores_x, scores_y, groups = {}, {}, {}
    print("Expected order    X outcome    Y outcome")
    for expected, response_x, response_y in cases:
        row = order_request(expected)
        x = judge(row, render_call(*response_x))
        y = judge(row, render_call(*response_y))
        key = row["id"]
        scores_x[key], scores_y[key] = x["correct"], y["correct"]
        groups[key] = key  # One illustrative group per request.
        print(f"{expected:17} {x['reason']:12} {y['reason']}")
        if args.show_calls:
            print(f"  X: {response_x[0]}(order_id={response_x[1]!r})")
            print(f"  Y: {response_y[0]}(order_id={response_y[1]!r})")
    print("\n'function' = wrong tool name; 'argument' here = wrong order ID.")
    result = paired(scores_y, scores_x, groups, boot_seed=42)
    print(f"Exact matches: X {sum(scores_x.values())}/{len(cases)}, Y {sum(scores_y.values())}/{len(cases)}")
    print(f"Only Y correct: {result['only_left_correct']}. Only X correct: {result['only_right_correct']}.")
    print(f"Y minus X: {result['difference_pp']:+g} percentage points on these four examples.")


if __name__ == "__main__":
    main()
