# Tool-call scoring and training-data selection

## Score your own tool calls

Run these commands from the repository root with Python 3. Both work offline on CPU. The fixtures are hand-written examples.

```sh
python3 -m s2l_tool_calling demo --show-calls
python3 -m s2l_tool_calling score --input examples/calls.jsonl --output verdicts.jsonl
```

The first command identifies the function-name and argument-value spans in an answer, then compares two sets of predictions on the same four requests. The second reads five different examples and reports one exact match out of five. The optional output file contains one verdict per request.

## Prepare your predictions

Copy [calls.jsonl](calls.jsonl) to `my-calls.jsonl` and replace the records. Use one UTF-8 JSON object per line with these fields.

| Field | What to supply |
| --- | --- |
| `id` | A unique, nonempty string identifying the request. |
| `tools` | An array of available tools. Each object has a `function.name` and a `function.parameters.properties` object defining its argument types. Use `{}` for a function without parameters. |
| `expected` | The single reference call, for example `{"name":"track_order","arguments":{"order_id":"0082"}}`. Its name must match one of the tools. Preserve JSON types, including strings with leading zeros. |
| `prediction` | The model's decoded response as a string in the Qwen tool-call format shown below. Save this text before any API wrapper converts it to a structured object. |

For the reference above, the following decoded response earns an exact match.

```text
<tool_call>
<function=track_order>
<parameter=order_id>
0082
</parameter>
</function>
</tool_call>
```

When preparing JSONL in Python, write each record with `json.dumps(record, ensure_ascii=False) + "\n"`. The JSON encoder escapes line breaks inside the prediction. Then run the scorer with `--input my-calls.jsonl`.

## Read the verdicts

The summary reports exact-match accuracy over every input request. Each output row contains `id`, `correct`, `reason`, and the parsed calls where available. The five possible reasons are `correct`, `function` (wrong tool), `argument` (wrong argument keys, types, or values), `count` (zero or multiple calls), and `parse` (malformed tool-call markup). Argument errors include a JSON path such as `["minutes"]`. For a malformed **input record**, the command exits with an error and prints its filename and line number.

## Select training examples from your own loss trajectories

On macOS or Linux, install the two CPU dependencies into your Python environment. The example selects four IDs from twelve invented trajectories.

```sh
python3 -m pip install -r requirements/selection.txt
python3 -m s2l_tool_calling select --input examples/trajectories.jsonl --count 4 --clusters 3 --seed 42 --method C > selected.json
```

Use the `selected_ids` array in `selected.json` to filter your **training** pool by ID. For a comparison on the same pool, run the command again with `--method A` or `--method B` and a different output filename.

To prepare your own [trajectory file](trajectories.jsonl), score the **same reference response** at five proxy checkpoints, in the same order for every candidate. Each row needs a unique string `id`, a `steps` array (five increasing integer updates, starting at zero), and the following arrays of five per-token mean negative log-likelihoods. A whole-answer mean includes the supervised formatting tokens.

| Input | Tokens averaged at each checkpoint |
| --- | --- |
| `all` | All supervised answer tokens. |
| `function` | Function-name tokens only. |
| `value` | Scalar argument-value tokens only. Use JSON `null` for the entire field when the reference contains zero value tokens. |

**A** samples uniformly from the entire pool. **B** clusters the five whole-answer means. **C** concatenates the five function-name means and the five value means into ten features. B and C standardize each column within its pool, apply K-means, and take turns drawing from each cluster until the requested number is reached. They share a proportional quota of no-value rows, selected using function-name trajectories. Keep the input order, `--cluster-seed` (default 42), and package versions fixed when comparing draw seeds. The command checks for duplicate IDs, nonfinite or negative losses, and inconsistent steps. It reads precomputed means. The [research pipeline](../docs/reproduce.md) produces and verifies the underlying token counts and checkpoint receipts.
