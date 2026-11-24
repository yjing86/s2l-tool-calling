"""Blind annotation sheets, inter-rater agreement and paired single-call diagnostics.

template --data test.jsonl --out labels.jsonl [--size N --seed 31415]
agreement --data test.jsonl --labels rater1.jsonl --second rater2.jsonl --out agreement.json
analyze --data test.jsonl --b B/eval --c C/eval --out report42 [--labels adjudicated.jsonl]

Each annotation row must contain all and only the scalar paths for that request.
Fill source with explicit, normalized, inferred, or uncertain. Keep the two
original sheets and make a separate adjudicated copy before opening predictions.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from evaluate_experiment import digest_json, differences, fingerprint, write_json
from verify_tool_call_loss import gold_calls


SOURCES = ("explicit", "normalized", "inferred", "uncertain")
MISSING = object()


def rows_by_id(path):
    result = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            key = str(rec["id"])
            if key in result:
                raise ValueError(f"Duplicate ID {key}: {path}")
            result[key] = rec
    if not result:
        raise ValueError(f"Empty JSONL: {path}")
    return result


def leaves(value, path=()):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from leaves(v, path + (k,))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from leaves(v, path + (i,))
    else:
        yield path, value


def reference(row):
    calls = gold_calls(row)
    if len(calls) != 1:
        raise ValueError(f"Expected one reference call in {row['id']}")
    func = calls[0]["function"]
    tool = next(t["function"] for t in row["tools"] if t["function"]["name"] == func["name"])
    return func, tool


def labels_template(args, data):
    if args.size is not None and not 0 < args.size <= len(data):
        raise ValueError("Invalid annotation sample size")
    order = sorted(data, key=lambda rid: hashlib.sha256(f"{args.seed}|{rid}".encode()).digest())
    with args.out.open("x") as f:
        for rid in order[:args.size]:
            row = data[rid]
            func, tool = reference(row)
            obj = {"id": rid, "group_id": row["group_id"], "row_sha256": digest_json(row),
                   "messages": row["messages"][:-1], "reference_tool": tool,
                   "leaves": [{"path": list(p), "reference_value": v, "source": None}
                              for p, v in leaves(func["arguments"])]}
            print(json.dumps(obj, ensure_ascii=False), file=f)


def reviewed(path, data):
    labels = {}
    for rid, rec in rows_by_id(path).items():
        if rid not in data or rec["row_sha256"] != digest_json(data[rid]):
            raise ValueError(f"Annotation source changed for {rid}")
        func, _ = reference(data[rid])
        gold = dict(leaves(func["arguments"]))
        categories = {}
        for entry in rec["leaves"]:
            p = tuple(entry["path"])
            if p not in gold or p in categories or digest_json(entry["reference_value"]) != digest_json(gold[p]):
                raise ValueError(f"Invalid reference/path in annotation {rid}:{p}")
            if entry["source"] not in SOURCES:
                raise ValueError(f"Missing or unknown source in annotation {rid}:{p}")
            categories[p] = entry["source"]
        if categories.keys() != gold.keys():
            raise ValueError(f"Incomplete leaf coverage on {rid}")
        labels[rid] = categories
    return labels


def agreement(args, data):
    first = reviewed(args.labels, data)
    second = reviewed(args.second, data)
    if first.keys() != second.keys():
        raise ValueError("Rater sheets must cover identical requests")
    cross = Counter()
    conflicts = []
    for rid in first:
        for path, left in first[rid].items():
            right = second[rid][path]
            cross[left, right] += 1
            if left != right:
                conflicts.append({"id": rid, "path": list(path), "first": left, "second": right})
    n = sum(cross.values())
    if args.out.exists():
        raise ValueError(f"Output exists: {args.out}")
    write_json(args.out, {"requests": len(first), "leaves": n, "agreed": n - len(conflicts),
                          "agreement": 1-len(conflicts)/n if n else None, "conflicts": conflicts,
                          "table": {a: {b: cross[a, b] for b in SOURCES} for a in SOURCES},
                          "rater_file_sha256": [fingerprint(args.labels), fingerprint(args.second)]})


def read_eval(folder, data, source_hash):
    cfg = json.loads((folder / "config.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    records = rows_by_id(folder / "predictions.jsonl")
    if (cfg["source_sha256"] != source_hash or metrics["count"] != len(data)
            or set(records) != set(data) or "accuracy" not in metrics):
        raise ValueError(f"Scored split or completion receipt mismatch: {folder}")
    for rid, pred in records.items():
        if pred["group_id"] != data[rid]["group_id"] or type(pred.get("correct")) is not bool:
            raise ValueError(f"Invalid group/score in {rid}")
    return cfg, records


def prediction_args(pred, function_name):
    calls = pred["parsed"]
    if calls is not None and len(calls) == 1 and calls[0]["name"] == function_name:
        return calls[0]["arguments"]
    return MISSING


def leaf_correct(args, path, expected, schema):
    value = args
    for key in path:
        if isinstance(key, str):
            if not isinstance(value, dict) or key not in value:
                return False
            value = value[key]
            schema = schema.get("properties", {}).get(key, {})
        else:
            if not isinstance(value, list) or key >= len(value):
                return False
            value = value[key]
            schema = schema.get("items", {})
    return not differences(value, expected, schema=schema)


def paired(rows, seed):
    # Each tuple is (group_id, B successes, C successes, shared denominator).
    # One draw samples whole near-duplicate groups with replacement.
    grouped = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    for group, b, c, n in rows:
        grouped[group] += [b, c, n]
    if not grouped:
        return {"requests": 0, "groups": 0, "denominator": 0, "B_correct": 0, "C_correct": 0,
                "B_percent": None, "C_percent": None, "C_minus_B_pp": None, "ci95_pp": None}
    g = np.asarray([grouped[key] for key in sorted(grouped, key=str)])
    b, c, den = map(int, g.sum(axis=0))
    ci = None
    if len(g) > 1:
        rng = np.random.default_rng(seed)
        draws = []
        for start in range(0, 2000, 64):
            take = rng.integers(len(g), size=(min(64, 2000-start), len(g)))
            sums = g[take].sum(axis=1)
            draws.extend(100*(sums[:, 1]-sums[:, 0])/sums[:, 2])
        ci = np.percentile(draws, [2.5, 97.5]).tolist()
    return {"requests": len(rows), "groups": len(g), "denominator": den, "B_correct": b,
            "C_correct": c, "B_percent": 100*b/den, "C_percent": 100*c/den,
            "C_minus_B_pp": 100*(c-b)/den, "ci95_pp": ci}


def fmt(x):
    return "—" if x is None else f"{x:.2f}"


def analyze(args, data):
    source_hash = fingerprint(args.data)
    bc, bp = read_eval(args.b, data, source_hash)
    cc, cp = read_eval(args.c, data, source_hash)
    shared = ("source_sha256", "prompts_sha256", "model_revision", "tokenizer_sha256",
              "template_sha256", "stop_token_ids", "greedy", "script_sha256")
    if any(bc[k] != cc[k] for k in shared) or bc["args"]["max_tokens"] != cc["args"]["max_tokens"]:
        raise ValueError("Evaluation conditions differ between B and C")
    labels = reviewed(args.labels, data) if args.labels else {}
    structure = defaultdict(list)
    diagnostics = defaultdict(list)
    errors = Counter()
    for rid, row in data.items():
        func, tool = reference(row)
        gold = dict(leaves(func["arguments"]))
        b_args = prediction_args(bp[rid], func["name"])
        c_args = prediction_args(cp[rid], func["name"])
        both = b_args is not MISSING and c_args is not MISSING
        group = row["group_id"]
        depth = max(map(len, gold), default=0)
        slice_name = "no_scalar" if not depth else "flat" if depth == 1 else "nested"
        structure[slice_name].append((group, int(bp[rid]["correct"]), int(cp[rid]["correct"]), 1))
        errors[bp[rid]["reason"], cp[rid]["reason"]] += 1
        if rid not in labels:
            continue
        for cls in set(labels[rid].values()):
            paths = [p for p, kind in labels[rid].items() if kind == cls]
            gold_schema = tool["parameters"]
            b = sum(leaf_correct(b_args, p, gold[p], gold_schema) for p in paths)
            c = sum(leaf_correct(c_args, p, gold[p], gold_schema) for p in paths)
            for cohort in (["all", "both_tools_correct"] if both else ["all"]):
                diagnostics[cls, cohort, "request"].append((group, int(b == len(paths)), int(c == len(paths)), 1))
                diagnostics[cls, cohort, "leaf"].append((group, b, c, len(paths)))
    strata = {cls: {cohort: {unit: paired(diagnostics[cls, cohort, unit], args.seed)
                              for unit in ("request", "leaf")} for cohort in ("all", "both_tools_correct")}
              for cls in SOURCES} if labels else {}
    for result in strata.values():
        n = result["all"]["request"]["denominator"]
        result["joint_tool_coverage"] = result["both_tools_correct"]["request"]["denominator"] / n if n else None
    report = {"source_sha256": source_hash, "bootstrap": {"draws": 2000, "seed": args.seed, "unit": "group_id"},
              "predictions_sha256": [fingerprint(p/"predictions.jsonl") for p in (args.b, args.c)],
              "annotation_sha256": fingerprint(args.labels) if args.labels else None,
              "annotation_requests": len(labels), "total_requests": len(data),
              "strata": strata, "structural": {s: paired(rows,args.seed) for s, rows in sorted(structure.items())},
              "error_transitions": [{"B": b, "C": c, "requests": n} for (b,c), n in sorted(errors.items())]}
    lines = ["# 单调用参数诊断", "", f"全卷共 {len(data)} 道，已复核来源标签的题目有 {len(labels)} 道。结构分层覆盖全卷。配对差以百分点计，按近重复题组重采样 2,000 次。", "",
             "| 参考参数结构 | 请求数 | B 整题答对 | C 整题答对 | C−B（百分点） |", "|---|---:|---:|---:|---:|"]
    for s, r in report["structural"].items():
        lines.append(f"| {s} | {r['requests']} | {r['B_correct']} | {r['C_correct']} | {fmt(r['C_minus_B_pp'])} |")
    if strata:
        lines += ["", "每类分别按请求和标量叶值计数。“共同选对工具”只纳入两组均调用一次且选择正确工具的请求。一题可包含多类参数。", "",
                  "| 来源 | 范围 | 单位 | B 正确/分母 | C 正确/分母 | C−B | 95% 区间 |", "|---|---|---|---:|---:|---:|---|"]
        for cls, result in strata.items():
            for cohort in ("all","both_tools_correct"):
                for unit in ("request","leaf"):
                    r = result[cohort][unit]
                    ci = r["ci95_pp"]
                    lines.append(f"| {cls} | {cohort} | {unit} | {r['B_correct']}/{r['denominator']} | {r['C_correct']}/{r['denominator']} | {fmt(r['C_minus_B_pp'])} | {', '.join(fmt(x) for x in ci) if ci else '—'} |")
    else:
        lines += ["", "参数来源分析待人工标注。仅凭预测结果无法区分请求明示、格式规范化和需要推导的值。"]
    if args.out.exists():
        raise ValueError("Select a new output directory")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        out = Path(tmp)/"result"
        out.mkdir()
        write_json(out/"field-diagnostics.json", report)
        (out/"field-diagnostics.md").write_text("\n".join(lines)+"\n")
        out.rename(args.out)


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = cli.add_subparsers(dest="mode", required=True)
    a, b, c = [sub.add_parser(n) for n in ("template", "agreement", "analyze")]
    for p in (a,b,c):
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    a.add_argument("--size", type=int)
    a.add_argument("--seed", type=int, default=31415)
    b.add_argument("--labels", type=Path, required=True)
    b.add_argument("--second", type=Path, required=True)
    c.add_argument("--labels", type=Path)
    c.add_argument("--b", type=Path, required=True)
    c.add_argument("--c", type=Path, required=True)
    c.add_argument("--seed", type=int, default=31415)
    args = cli.parse_args()
    data = rows_by_id(args.data)
    {"template": labels_template, "agreement": agreement, "analyze": analyze}[args.mode](args,data)


if __name__ == "__main__":
    main()
