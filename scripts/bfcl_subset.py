"""Freeze the external BFCL v4 subset and score blind vLLM generations.

Use the pinned BFCL checkout and its dependencies in a separate CPU environment.
Run 'prepare' before opening model results; only prompts.jsonl goes to generation.
After locking models, run evaluate_experiment.py with --generate-only when
preparing token IDs and generating predictions. Feed predictions.jsonl to
'score' and compare matched B/C results with 'compare'. The CPU scorer needs
BFCL's declared dependencies plus soundfile (imported by qwen-agent).
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

REVISION = "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000"
CATEGORIES = ("simple_python", "multiple")
SOURCE_HASHES = {
    "data/BFCL_v4_simple_python.json": "82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991",
    "data/BFCL_v4_multiple.json": "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
    "data/possible_answer/BFCL_v4_simple_python.json": "90cd5bc653690ee8e459b5b3f3fc9458606f7f3fcbf795bb51b7dc581f8c86dc",
    "data/possible_answer/BFCL_v4_multiple.json": "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047",
    "eval_checker/ast_eval/ast_checker.py": "2aae7a68461a8f76c0be3894c8901b66b56967a1989d3ab066051e3fb97f1538",
    "model_handler/utils.py": "f78fd3edce603b333dc9a88ee2c041dc547d51f71aa449ffebc044c4b1e353f3",
}
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "local/tools/bfcl"
MODEL = "Qwen/Qwen3.5-2B"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def keyed(path):
    result = {}
    for row in rows(path):
        if row["id"] in result:
            raise ValueError(f"{path}: duplicate {row['id']}")
        result[row["id"]] = row
    return result


def write(path, text):
    # A rerun may reproduce an artifact, but must not overwrite an earlier result.
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"{path} already contains different content")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def lines(records):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)


def text_key(text):
    # Mask numeric IDs and dates so a copied question with new numbers is flagged.
    return " ".join(re.findall(r"\w+", re.sub(r"\d+", " NUMBER ", text.casefold())))


def signature(fn):
    def shape(schema):
        body = {"type": schema.get("type")}
        if "properties" in schema:
            body["properties"] = {key: shape(item) for key, item in schema["properties"].items()}
        if isinstance(schema.get("items"), dict):
            body["items"] = shape(schema["items"])
        if "required" in schema:
            required = schema["required"]
            body["required"] = sorted(required) if isinstance(required, list) else required
        return body

    return json.dumps([fn["name"].casefold(), shape(fn["parameters"])], sort_keys=True)


def prepare(args, source):
    from bfcl_eval.constants.enums import ModelStyle
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.model_handler.utils import convert_to_tool

    train_prompts, train_names, train_signatures = set(), set(), set()
    for record in rows(args.train):
        train_prompts.update(text_key(m["content"]) for m in record["messages"] if m["role"] == "user")
        for tool in record["tools"]:
            fn = tool["function"]
            train_names.add(fn["name"].casefold())
            train_signatures.add(signature(fn))

    prompts, items = [], []
    for category in CATEGORIES:
        pool = keyed(source / "data" / f"BFCL_v4_{category}.json")
        def rank(key):
            return hashlib.sha256(f"31415|{category}|{key}".encode()).digest(), key
        for key in sorted(pool, key=rank)[:100]:
            question = pool[key]
            messages, = question["question"]
            if [m["role"] for m in messages] != ["user"]:
                raise ValueError(f"{key}: expected single user request")
            tools = convert_to_tool(question["function"], GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
            functions = [t["function"] for t in tools]
            if len(set(f["name"] for f in functions)) != len(functions):
                raise ValueError(f"{key}: function names collide after BFCL normalization")
            prompts.append({"id": key, "messages": messages, "tools": tools})
            names = sorted({f["name"] for f in functions if f["name"].casefold() in train_names})
            items.append({"id": key, "category": category, "matching_tool_names": names,
                          "prompt_overlap": text_key(messages[0]["content"]) in train_prompts,
                          "signature_overlap": any(signature(f) in train_signatures for f in functions)})
    write(args.out / "prompts.jsonl", lines(prompts))
    manifest = {"revision": REVISION, "seed": 31415, "source_sha256": SOURCE_HASHES,
                "training_pool_sha256": digest(args.train), "prompts_sha256": digest(args.out / "prompts.jsonl"),
                "sensitivity_exclusion": "normalized user text or named schema matches any training-pool item",
                "items": items}
    write(args.out / "manifest.json", json.dumps(manifest, indent=2) + "\n")


def score_entry(functions, calls, truth, category):
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    # Only this setting is read from the model registry by the official AST checker.
    # It matches our use of official OPENAI_COMPLETIONS conversion (dot -> underscore).
    MODEL_CONFIG_MAPPING.setdefault(MODEL, SimpleNamespace(underscore_to_dot=True))
    if not MODEL_CONFIG_MAPPING[MODEL].underscore_to_dot:
        raise ValueError("Model config does not match the function-name normalization")
    if not isinstance(calls, list) or any(not isinstance(c, dict) or
            not isinstance(c.get("name"), str) or not isinstance(c.get("arguments"), dict) for c in calls):
        return {"valid": False, "error_type": "ast_decoder:decoder_failed"}
    decoded = [{call["name"]: call["arguments"]} for call in calls]
    return ast_checker(functions, decoded, truth, Language.PYTHON, category, MODEL)


def score(args, source):
    manifest = json.loads(args.manifest.read_text())
    if manifest["revision"] != REVISION or manifest["source_sha256"] != SOURCE_HASHES:
        raise ValueError("manifest disagrees with pinned sources")
    predictions = keyed(args.predictions)
    expected = {entry["id"] for entry in manifest["items"]}
    if set(predictions) != expected:
        raise ValueError(f"prediction IDs differ: missing={len(expected-set(predictions))}, extra={len(set(predictions)-expected)}")
    scores = []
    for category in CATEGORIES:
        filename = f"BFCL_v4_{category}.json"
        questions = keyed(source / "data" / filename)
        answers = keyed(source / "data/possible_answer" / filename)
        for item in manifest["items"]:
            if item["category"] != category:
                continue
            key = item["id"]
            prediction = predictions[key]
            verdict = score_entry(questions[key]["function"], prediction["parsed"],
                                  answers[key]["ground_truth"], category)
            scores.append({**item, "valid": verdict["valid"],
                           "error_type": None if verdict["valid"] else verdict.get("error_type"),
                           "error": verdict.get("error", [])})
    write(args.out / "bfcl-scores.jsonl", lines(scores))
    summary = {}
    for category in CATEGORIES:
        group = [r for r in scores if r["category"] == category]
        for suffix, subset in (("all", group), ("excluding_overlap", [r for r in group if not (r["prompt_overlap"] or r["signature_overlap"])])):
            correct = sum(r["valid"] for r in subset)
            summary[f"{category}/{suffix}"] = {"n": len(subset), "correct": correct,
                    "accuracy": correct / len(subset) if subset else None,
                    "errors": dict(Counter(r["error_type"] for r in subset if not r["valid"]))}
    summary.update(manifest_sha256=digest(args.manifest), predictions_sha256=digest(args.predictions))
    write(args.out / "bfcl-summary.json", json.dumps(summary, indent=2) + "\n")


def compare(args):
    baseline, candidate = keyed(args.baseline), keyed(args.candidate)
    if not baseline or baseline.keys() != candidate.keys():
        raise ValueError("Both models must have exactly the same nonempty set of question IDs")
    summary = {}
    rng = random.Random(31415)
    for category in CATEGORIES:
        for suffix, exclude_overlap in (("all", False), ("excluding_overlap", True)):
            diff = []
            for key, left in sorted(baseline.items()):
                right = candidate[key]
                if any(left[k] != right[k] for k in ("category", "prompt_overlap", "signature_overlap")):
                    raise ValueError(f"Different benchmark metadata for {key}")
                if left["category"] != category or (exclude_overlap and (left["prompt_overlap"] or left["signature_overlap"])):
                    continue
                diff.append(int(right["valid"]) - int(left["valid"]))
            n = len(diff)
            samples = sorted(100 * sum(rng.choices(diff, k=n)) / n for _ in range(10000)) if n else []
            summary[f"{category}/{suffix}"] = {
                "n": n, "candidate_only_correct": diff.count(1), "baseline_only_correct": diff.count(-1),
                "delta_pp": 100 * sum(diff) / n if n else None,
                "paired_question_bootstrap_95_pp": [samples[249], samples[9749]] if n else None,
            }
    summary.update(baseline_sha256=digest(args.baseline), candidate_sha256=digest(args.candidate),
                   interval_scope="paired benchmark items with trained models held fixed", seed=31415)
    write(args.out / "bfcl-comparison.json", json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--baseline", required=True, type=Path, help="bfcl-scores.jsonl of baseline")
    p.add_argument("--candidate", required=True, type=Path, help="bfcl-scores.jsonl of candidate")
    p.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "compare":
        compare(args)
        return
    sys.path.insert(0, str(args.bfcl_root.resolve()))
    source = args.bfcl_root / "bfcl_eval"
    for path, sha in SOURCE_HASHES.items():
        if digest(source / path) != sha:
            raise ValueError(f"unexpected BFCL revision for {path}")
    (prepare if args.action == "prepare" else score)(args, source)


if __name__ == "__main__":
    main()
