"""Freeze single-call diagnostic rules, then audit a complete strict-scored run.

freeze --data SPLIT.jsonl --out frozen.json [--reviewed-equivalences rules.json]
audit --data SPLIT.jsonl --freeze frozen.json --eval EVAL_DIR --out NEW_DIR

An optional equivalence rule is a reviewed JSON record with `tool`, `path`
(array offsets written "*"), `kind`, `quote` (verbatim from this property's
description), and `reviewer`. Empty rules are the default. Keep reviewer sheets
local. Changing a rule set requires a new freeze; record its hash before scoring.
"""
import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import re
import tempfile

from evaluate_experiment import differences, digest_json, fingerprint, judge, write_json
from verify_tool_call_loss import gold_calls

VALUE = r"(?P<value>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
# These are the two tool/field pairs independently verified on the pilot.
# Anchor the number to the user concept, units and the field's exact schema.
CONVERSIONS = {
    ("spaceprobetrajectoryplannerapi", "delta_v_budget"): (
        "number", "the total change in velocity the space probe can achieve in its journey in kilometers per second",
        re.compile(r"\bdelta[-‐‑– ]v\s+budget(?:\s+for\s+the\s+mission)?\s*(?:is|of|=|:)\s*"
                   + VALUE + r"\s*(?:m/s|met(?:er|re)s?\s+per\s+second)\b", re.I),
        "m/s", "km/s", Decimal("0.001")),
    ("estimate_product_battery_life", "average_daily_usage"): (
        "integer", "the average amount of daily usage (in minutes)",
        re.compile(r"(?<![\w.\-,])" + VALUE + r"\s+hours?\s+(?:each day|every day|per day|daily)\b", re.I),
        "hours/day", "minutes/day", Decimal("60")),
}

KINDS = ("ascii_case", "omitted_optional_empty_array", "array_order")
# "uniqueItems", "optional", and JSON-Schema's non-operative default annotation
# do not establish equal behavior at execution. Require an explicit tool claim.
EVIDENCE = {
    "ascii_case": re.compile(r"case[- ]insensitive|case does not matter", re.I),
    "array_order": re.compile(r"order (?:of (?:the )?(?:items|elements|values) )?(?:does not matter|is (?:irrelevant|ignored))", re.I),
    "omitted_optional_empty_array": re.compile(r"(?:if|when) omitted.{0,35}(?:empty (?:list|array)|\[\])", re.I),
}


def rows(path):
    indexed = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rid = str(row["id"])
            if rid in indexed:
                raise ValueError(f"Duplicate row {rid}")
            indexed[rid] = row
    if not indexed:
        raise ValueError(f"Empty input: {path}")
    return indexed


def gold(row):
    calls = gold_calls(row)
    if len(calls) != 1:
        raise ValueError("Only single-call Turnstile records are supported")
    func = calls[0]["function"]
    sc = next(t["function"]["parameters"] for t in row["tools"] if t["function"]["name"] == func["name"])
    return func, sc


def conversion_fields(row):
    func, schema = gold(row)
    user = " ".join(m["content"] for m in row["messages"] if m["role"] == "user")
    out = []
    for (tool, field), (kind, desc, expr, source, target, factor) in CONVERSIONS.items():
        if tool != func["name"]:
            continue
        prop = schema.get("properties", {}).get(field, {})
        if prop.get("type") != kind or prop.get("description") != desc:
            continue
        matches = list(expr.finditer(user))
        if len(matches) != 1:
            continue
        m = matches[0]
        expected = func["arguments"].get(field)
        if type(expected) not in (float, int):
            continue
        amount = Decimal(m["value"].replace(",", ""))
        converted = amount * factor
        if amount == converted or Decimal(str(expected)) != converted:
            continue
        out.append({"field": field, "source_text": m[0], "source_start": m.start(),
                    "source_amount": str(amount), "from_unit": source, "to_unit": target,
                    "factor": str(factor), "converted": str(converted), "schema_description": desc})
    return out


def property_at(schema, path):
    for key in path:
        schema = schema.get("items", {}) if key == "*" else schema.get("properties", {}).get(key, {})
    return schema


def freeze(args):
    data = rows(args.data)
    schemas = {}
    for row in data.values():
        func, sc = gold(row)
        other = schemas.setdefault(func["name"], sc)
        if other != sc:
            raise ValueError(f"Conflicting definitions for {func['name']}")
    rules = json.loads(args.reviewed_equivalences.read_text()) if args.reviewed_equivalences else []
    seen = set()
    for r in rules:
        key = (r["tool"], tuple(r["path"]), r["kind"])
        prop = property_at(schemas.get(r["tool"], {}), r["path"])
        quote = r["quote"].strip()
        if (key in seen or not prop or not quote or not r["reviewer"].strip() or r["kind"] not in KINDS
                or quote not in prop.get("description", "") or not EVIDENCE[r["kind"]].search(quote)):
            raise ValueError(f"Invalid, duplicate, or unsupported reviewed rule: {key}")
        seen.add(key)
    qualified = {}
    for rid, row in data.items():
        fields = conversion_fields(row)
        if fields:
            qualified[rid] = fields
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, {"source_sha256": fingerprint(args.data), "script_sha256": fingerprint(__file__),
                          "scorer_sha256": fingerprint(Path(__file__).with_name("evaluate_experiment.py")),
                          "total_requests": len(data), "eligible_conversion_requests": qualified,
                          "reviewed_equivalences": rules, "conversion_scope": [list(k) for k in CONVERSIONS]})
    print(f"Freeze SHA256 {fingerprint(args.out)}; {len(qualified)}/{len(data)} eligible requests.")


def repr_edits(actual, expected, sc, path=()):
    """Return only established representation-only edits; None means a substantive/other difference."""
    if not differences(actual, expected, schema=sc):
        return []
    if isinstance(expected, str) and isinstance(actual, str):
        if expected.isascii() and actual.isascii() and expected.lower() == actual.lower():
            return [("ascii_case", path)]
        return None
    if isinstance(expected, dict) and isinstance(actual, dict):
        if actual.keys() - expected.keys():
            return None
        edits = []
        for k, item in expected.items():
            child = sc.get("properties", {}).get(k, {})
            if k not in actual:
                if (type(item) is list and not item and child.get("type") == "array"
                        and k not in sc.get("required", [])):
                    edits.append(("omitted_optional_empty_array", path+(k,)))
                else:
                    return None
            else:
                nested = repr_edits(actual[k], item, child, path+(k,))
                if nested is None:
                    return None
                edits.extend(nested)
        return edits
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        if Counter(map(digest_json, actual)) == Counter(map(digest_json, expected)):
            return [("array_order", path)]
        edits = []
        for idx, (got, ref) in enumerate(zip(actual, expected)):
            child = repr_edits(got, ref, sc.get("items", {}), path+(idx,))
            if child is None:
                return None
            edits.extend(child)
        return edits
    return None


def rule_for(rules, name, kind, path, schema, predicted):
    normalized = tuple("*" if isinstance(key, int) else key for key in path)
    rule = rules.get((name, normalized, kind))
    if rule and kind == "ascii_case":
        value = predicted
        for key in path:
            value = value[key]
        prop = property_at(schema, normalized)
        # Never infer behavioral equivalence over an explicitly disallowed value.
        if "enum" in prop and value not in prop["enum"]:
            return None
        if "const" in prop and value != prop["const"]:
            return None
    return rule


def audit(args):
    policy = json.loads(args.freeze.read_text())
    data, preds = rows(args.data), rows(args.eval/"predictions.jsonl")
    cfg = json.loads((args.eval/"config.json").read_text())
    metrics = json.loads((args.eval/"metrics.json").read_text())
    if (policy["source_sha256"] != fingerprint(args.data) or policy["source_sha256"] != cfg["source_sha256"]
            or policy["script_sha256"] != fingerprint(__file__)
            or policy["scorer_sha256"] != fingerprint(Path(__file__).with_name("evaluate_experiment.py"))
            or metrics["count"] != len(data) or metrics.get("correct") is None or preds.keys() != data.keys()):
        raise ValueError("Diagnostic freeze, scorer, predictions, or evaluation receipt mismatch")
    rules = {(r["tool"], tuple(r["path"]), r["kind"]): r for r in policy["reviewed_equivalences"]}
    flags = Counter(dict.fromkeys(KINDS, 0))
    unit = Counter(dict.fromkeys(("wrong_tool_count_or_unparsed", "missing_field", "non_numeric_type",
                                  "correct_magnitude", "wrong_magnitude", "eligible_fields", "unconverted_source_magnitude"), 0))
    counts = Counter(dict.fromkeys(("strict_correct", "matched_tool_and_count", "parse_failures",
                                    "length_limited_parse_failures", "representation_candidate_requests",
                                    "all_edits_schema_confirmed"), 0))
    details = []
    for rid, row in data.items():
        func, sc = gold(row)
        pred = preds[rid]
        if type(pred.get("correct")) is not bool or pred["group_id"] != row["group_id"]:
            raise ValueError(f"Missing strict score or wrong group: {rid}")
        replay = judge(row, pred["raw_output"])
        if any(digest_json(replay.get(key)) != digest_json(pred.get(key)) for key in ("correct", "reason", "parsed", "argument_errors")):
            raise ValueError(f"Strict judgment replay differs from original on {rid}")
        counts["strict_correct"] += int(pred["correct"])
        calls = pred["parsed"]
        matched = isinstance(calls, list) and len(calls) == 1 and calls[0]["name"] == func["name"]
        actual = calls[0]["arguments"] if matched else None
        counts["matched_tool_and_count"] += int(matched)
        detail = {"id": rid, "strict_reason": pred["reason"], "unit_checks": [], "representation": []}
        if pred["reason"] == "parse":
            counts["parse_failures"] += 1
            if pred["finish_reason"] == "length":
                counts["length_limited_parse_failures"] += 1
                detail["parse_length"] = True
        for check in policy["eligible_conversion_requests"].get(rid, []):
            field = check["field"]
            value = actual.get(field) if matched else None
            if not matched:
                status = "wrong_tool_count_or_unparsed"
            elif field not in actual:
                status = "missing_field"
            elif type(value) not in (int, float):
                status = "non_numeric_type"
            elif Decimal(str(value)) == Decimal(check["converted"]):
                status = "correct_magnitude"
            else:
                status = "wrong_magnitude"
            copied = status == "wrong_magnitude" and Decimal(str(value)) == Decimal(check["source_amount"])
            unit[status] += 1
            unit["eligible_fields"] += 1
            unit["unconverted_source_magnitude"] += int(copied)
            detail["unit_checks"].append({**check, "status": status, "predicted": value, "copied_source_magnitude": copied})
        if matched and not pred["correct"]:
            edits = repr_edits(actual, func["arguments"], sc)
            if edits:
                known = True
                for kind, path in edits:
                    support = rule_for(rules, func["name"], kind, path, sc, actual)
                    known &= bool(support)
                    detail["representation"].append({"kind": kind, "path": path, "reviewed_rule": support})
                for kind in set(k for k, _ in edits):
                    flags[kind] += 1
                counts["representation_candidate_requests"] += 1
                counts["all_edits_schema_confirmed"] += int(known)
                detail["semantic_status"] = "schema_confirmed" if known else "not_established"
        if detail["representation"] or detail["unit_checks"] or detail.get("parse_length"):
            details.append(detail)
    if counts["strict_correct"] != metrics["correct"]:
        raise ValueError("Strict accuracy receipt differs from immutable predictions")
    summary = {"benchmark": "Turnstile fixed single-call exact match; no tool execution",
               "scope": "Unit checks cover only the two pilot-verified fields and exact schema/text conditions. A wrong magnitude is an observed mismatch, not evidence of the model's internal reasoning.",
               "semantic_scope": "Representation candidates remain strict errors. A schema-confirmed equivalent requires reviewed tool-specific evidence for every edit; aliases, timezones and unit-bearing strings are not normalized.",
               "total": len(data), "counts": dict(counts), "eligible_conversion_requests": len(policy["eligible_conversion_requests"]),
               "strict_eval_script_sha256": cfg["script_sha256"], "judge_replay_verified": len(data),
               "evaluable_numeric_fields": unit["correct_magnitude"] + unit["wrong_magnitude"],
               "unit_field_counts": dict(unit), "representation_candidate_requests_by_kind": dict(flags),
               "freeze_sha256": fingerprint(args.freeze), "predictions_sha256": fingerprint(args.eval/"predictions.jsonl")}
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        root = Path(tmp)/"diagnostics"
        root.mkdir()
        write_json(root/"summary.json",summary)
        (root/"evidence.jsonl").write_text("".join(json.dumps(rec,ensure_ascii=False)+"\n" for rec in details))
        root.rename(args.out)
    print(json.dumps(summary,ensure_ascii=False))


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = cli.add_subparsers(dest="mode", required=True)
    a, b = sub.add_parser("freeze"), sub.add_parser("audit")
    for p in (a,b):
        p.add_argument("--data",type=Path,required=True)
        p.add_argument("--out",type=Path,required=True)
    a.add_argument("--reviewed-equivalences",type=Path)
    b.add_argument("--freeze",type=Path,required=True)
    b.add_argument("--eval",type=Path,required=True)
    args = cli.parse_args()
    (freeze if args.mode=="freeze" else audit)(args)


if __name__ == "__main__":
    main()
