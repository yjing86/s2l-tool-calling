"""Prepare, freeze and independently score the Round 2 BFCL Live external panel.

Run candidates with the pinned BFCL environment, freeze with the pinned training
environment (CPU is sufficient), and score in the BFCL environment after inference.

    ROOT=local/followup/round2/bfcl-live
    "$BFCL_PY" scripts/bfcl_live.py candidates --train local/prepared/train.jsonl --out "$ROOT/candidates"
    "$TRAIN_PY" scripts/bfcl_live.py freeze --candidates "$ROOT/candidates" --model local/models/Qwen3.5-2B --out "$ROOT/panel"
    "$TRAIN_PY" scripts/evaluate_experiment.py --generate-only --prepare-only --model local/models/Qwen3.5-2B \
        --data "$ROOT/panel/prompts.jsonl" --swift-run-args local/course-run/training/s42-A/args.json --out "$ROOT/native-cache"

The public reference contains only checksums and category counts in
results/followup/round2-bfcl-live-panel.json. See run_round2_bfcl.py for the
resumable generation, scoring and reporting commands for the registered models.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from bfcl_subset import DEFAULT_ROOT, REVISION, SOURCE_HASHES, digest, keyed, lines, rows
from bfcl_subset import score_entry, signature, text_key, write
from evaluate_experiment import BASE_REVISION, digest_json, parse_calls


CATEGORIES = ("live_simple", "live_multiple")
MAX_PER_CATEGORY = 300
MAX_INPUT_TOKENS = 8192 - 512
TRAIN_SHA = "fd0ee69020ff537165b0fc405883f33ea2269ccc14cd4a231f62e5aff32e6001"
TOKENIZER_SHA = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
TEMPLATE_SHA = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
NATIVE_ARGS_SHA = "5118dc5ba9377c035f868a6fb541149b31a9678eeb7d7a8a474991be72d68688"
SAVED_TEMPLATE_ARGS = {
    "add_non_thinking_prefix": True, "agent_template": "qwen3_5", "enable_thinking": False,
    "is_binary_loss_scale": None, "loss_scale": "default+ignore_empty_think", "max_length": 8192,
    "max_pixels": None, "norm_bbox": None, "padding_free": False, "padding_side": "right",
    "preserve_thinking": None, "remove_unused_columns": True, "response_prefix": None,
    "sequence_parallel_size": 1, "system": None, "template": "qwen3_5", "template_backend": "swift",
    "truncation_strategy": "delete", "use_chat_template": True,
}
SOURCES = {
    "data/BFCL_v4_live_simple.json": "1af2ac87dca47556db7b7e37e51e28b459a38b594e3c7b3c792b4903598ca0c4",
    "data/BFCL_v4_live_multiple.json": "fd8ccfad4d911420d0e3341dbe2fff77d1d341da934248b9bb2bda24ab3a10c8",
    "data/possible_answer/BFCL_v4_live_simple.json": "fec9cfa9744a936f9126981e85a2023da1e63e273eafebc81923a1162fad70ce",
    "data/possible_answer/BFCL_v4_live_multiple.json": "97e90d59c5bd76c55a2920ce93e5566e9046307d3f558578f085f9d3a56c3084",
    "constants/enums.py": "2182becfa2a1d071ee1db30db593b4758c6bf866aa12d2d4b8daf09175ea518a",
    "constants/type_mappings.py": "1702fb67afbe2c492608e58e2b7d02e46381f50166b47f3c952f76e34c7cd3bd",
    **{name: sha for name, sha in SOURCE_HASHES.items() if not name.startswith("data/")},
}


def checked_sha(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Unexpected file contents: {path}")


def save_json(path, obj):
    write(path, json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def rank(item):
    category, key = item["category"], item["id"]
    return hashlib.sha256(f"31415|{category}|{key}".encode()).digest(), key


def candidate(category, question, tools, train_prompts, train_signatures):
    if len(question["question"]) != 1:
        return None, "unsupported_turns"
    messages = question["question"][0]
    if [m["role"] for m in messages] not in (["user"], ["system", "user"]) or any(
        not isinstance(m["content"], str) for m in messages
    ):
        return None, "unsupported_messages"
    functions = [tool["function"] for tool in tools]
    if len({f["name"] for f in functions}) != len(functions):
        return None, "colliding_names"
    if text_key(messages[-1]["content"]) in train_prompts:
        return None, "training_prompt_overlap"
    signatures = sorted(signature(f) for f in functions)
    if any(s in train_signatures for s in signatures):
        return None, "training_signature_overlap"
    normalized = [[m["role"], text_key(m["content"])] for m in messages]
    return {
        "id": question["id"], "category": category,
        "duplicate_key": digest_json([normalized, signatures]),
        "messages": messages, "tools": tools,
    }, None


def candidates(args, source):
    from bfcl_eval.constants.enums import ModelStyle
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.model_handler.utils import convert_to_tool

    checked_sha(args.train, TRAIN_SHA)
    train_prompts, train_signatures = set(), set()
    for row in rows(args.train):
        train_prompts.update(text_key(m["content"]) for m in row["messages"] if m["role"] == "user")
        train_signatures.update(signature(tool["function"]) for tool in row["tools"])
    candidates_out, audit = [], {}
    for category in CATEGORIES:
        pool = keyed(source / f"data/BFCL_v4_{category}.json")
        reasons = Counter()
        for row in pool.values():
            tools = convert_to_tool(row["function"], GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
            item, reason = candidate(category, row, tools, train_prompts, train_signatures)
            if reason:
                reasons[reason] += 1
            else:
                candidates_out.append(item)
        audit[category] = {"source": len(pool), "excluded": dict(reasons)}
    write(args.out / "candidates.jsonl", lines(candidates_out))
    save_json(args.out / "candidates-audit.json", {
        "bfcl_revision": REVISION, "training_sha256": TRAIN_SHA, "categories": audit,
        "candidates_sha256": digest(args.out / "candidates.jsonl"),
    })


def choose(eligible, count_tokens):
    """Apply the context limit, then deduplicate in hash order and take each cap."""
    chosen, audit, seen = [], {}, set()
    for category in CATEGORIES:
        items = sorted((c for c in eligible if c["category"] == category), key=rank)
        reasons, kept = Counter(), []
        for item in items:
            size = count_tokens(item)
            if size > MAX_INPUT_TOKENS:
                reasons["context"] += 1
                continue
            if item["duplicate_key"] in seen:
                reasons["duplicate"] += 1
                continue
            seen.add(item["duplicate_key"])
            if len(kept) == MAX_PER_CATEGORY:
                reasons["beyond_cap"] += 1
                continue
            kept.append((item, size))
        audit[category] = {
            "excluded_after_overlap_filter": dict(reasons), "selected": len(kept),
            "selected_with_system": sum(r["messages"][0]["role"] == "system" for r, _ in kept),
            "max_selected_input_tokens": max((n for _, n in kept), default=0),
        }
        chosen.extend(kept)
    return chosen, audit


def native_counter(model, swift_args):
    from swift import get_processor, get_template
    from swift.arguments.base_args.template_args import TemplateArguments

    if importlib.metadata.version("ms-swift") != "4.6.1":
        raise ValueError("Use the pinned ms-swift 4.6.1 environment")
    checked_sha(model / "tokenizer.json", TOKENIZER_SHA)
    checked_sha(model / "chat_template.jinja", TEMPLATE_SHA)
    kwargs = TemplateArguments.get_template_kwargs(SimpleNamespace(**SAVED_TEMPLATE_ARGS))
    if swift_args:
        checked_sha(swift_args, NATIVE_ARGS_SHA)
        original = TemplateArguments.get_template_kwargs(SimpleNamespace(**json.loads(swift_args.read_text())))
        if original != kwargs:
            raise ValueError("Public template arguments differ from the frozen training run")
    processor = get_processor(str(model), model_type="qwen3_5", use_hf=True, download_model=False)
    template = get_template(processor, **kwargs)
    template.set_mode("transformers")
    # Count full, untruncated native IDs even for candidates beyond the training window.
    template.max_length = None

    def count(item):
        prompt = {"tools": json.dumps(item["tools"], ensure_ascii=False), "messages": item["messages"]}
        ids = template.encode(prompt)["input_ids"]
        if not ids:
            raise ValueError(f"Empty prompt: {item['id']}")
        return len(ids)

    return count


def freeze(args):
    source = args.candidates
    audit = json.loads((source / "candidates-audit.json").read_text())
    checked_sha(source / "candidates.jsonl", audit["candidates_sha256"])
    if audit["bfcl_revision"] != REVISION or audit["training_sha256"] != TRAIN_SHA:
        raise ValueError("Unexpected candidate input metadata")
    planned, stats = choose(list(rows(source / "candidates.jsonl")), native_counter(args.model, args.swift_run_args))
    prompts = [{key: item[key] for key in ("id", "messages", "tools")} for item, _ in planned]
    items = [{"id": item["id"], "category": item["category"], "input_tokens": n} for item, n in planned]
    if len({row["id"] for row in items}) != len(items):
        raise ValueError("Duplicate selected IDs across BFCL categories")
    write(args.out / "prompts.jsonl", lines(prompts))
    manifest = {
        "protocol": "round2-bfcl-live-v1", "bfcl_revision": REVISION, "source_sha256": SOURCES,
        "training_sha256": TRAIN_SHA, "tokenizer_sha256": TOKENIZER_SHA, "template_sha256": TEMPLATE_SHA,
        "training_args_sha256": NATIVE_ARGS_SHA, "candidates_sha256": audit["candidates_sha256"],
        "prompts_sha256": digest(args.out / "prompts.jsonl"), "items": items,
        "ordered_ids_sha256": digest_json([i["id"] for i in items]),
        "filter_counts": {c: {**audit["categories"][c], **stats[c]} for c in CATEGORIES},
    }
    save_json(args.out / "manifest.json", manifest)
    save_json(args.out / "panel-summary.json", {
        **{key: manifest[key] for key in ("protocol", "bfcl_revision", "ordered_ids_sha256", "filter_counts",
                                         "prompts_sha256", "training_sha256", "tokenizer_sha256", "template_sha256")},
        "manifest_sha256": digest(args.out / "manifest.json"),
        "source_sha256": SOURCES, "scope": "Secondary external benchmark; fixed subset of two live single-turn categories.",
    })


def score(args, source):
    manifest_path = args.panel / "manifest.json"
    checked_sha(manifest_path, args.panel_sha256)
    manifest = json.loads(manifest_path.read_text())
    if manifest["bfcl_revision"] != REVISION or manifest["source_sha256"] != SOURCES:
        raise ValueError("Panel and official sources disagree")
    prompt_path = args.panel / "prompts.jsonl"
    checked_sha(prompt_path, manifest["prompts_sha256"])
    prompts = keyed(prompt_path)
    predictions_path = args.generation / "predictions.jsonl"
    predictions, config = keyed(predictions_path), json.loads((args.generation / "config.json").read_text())
    if not (config["args"]["generate_only"] and config["source_sha256"] == manifest["prompts_sha256"]
            and config["model_revision"] == BASE_REVISION and config["tokenizer_sha256"] == TOKENIZER_SHA
            and config["template_sha256"] == TEMPLATE_SHA and config["args"]["max_tokens"] == 512
            and config["args"]["max_model_len"] == 8192 and config["greedy"] and config["seed"] == 42):
        raise ValueError("Generation settings differ from the frozen external protocol")
    items = manifest["items"]
    expected = {item["id"] for item in items}
    if len(expected) != len(items) or expected != set(prompts) or expected != set(predictions):
        raise ValueError("Need exactly one prediction and one prompt per frozen item")
    scores, summary = [], {}
    for category in CATEGORIES:
        questions = keyed(source / f"data/BFCL_v4_{category}.json")
        answers = keyed(source / f"data/possible_answer/BFCL_v4_{category}.json")
        selected = [item for item in items if item["category"] == category]
        errors = Counter()
        correct = 0
        for item in selected:
            key = item["id"]
            pred = predictions[key]
            if pred["input_tokens"] != item["input_tokens"]:
                raise ValueError(f"Native prompt length differs from freeze: {key}")
            try:
                calls = parse_calls(pred["raw_output"], prompts[key]["tools"])
            except ValueError:
                calls = None
            if digest_json(calls) != digest_json(pred["parsed"]):
                raise ValueError(f"Stored parse differs from replay: {key}")
            verdict = score_entry(questions[key]["function"], calls, answers[key]["ground_truth"], category)
            reason = None if verdict["valid"] else verdict["error_type"]
            correct += int(verdict["valid"])
            if reason:
                errors[reason] += 1
            scores.append({"id": key, "category": category, "correct": bool(verdict["valid"]), "error_type": reason})
        summary[category] = {"n": len(selected), "correct": correct, "accuracy": correct / len(selected),
                             "errors": dict(errors)}
    write(args.out / "bfcl-live-scores.jsonl", lines(scores))
    save_json(args.out / "bfcl-live-summary.json", {
        "categories": summary, "panel_sha256": args.panel_sha256, "predictions_sha256": digest(predictions_path),
        "generation_config_sha256": digest(args.generation / "config.json"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="action", required=True)
    cmd = sub.add_parser("candidates")
    cmd.add_argument("--train", required=True, type=Path)
    cmd.add_argument("--out", required=True, type=Path)
    cmd = sub.add_parser("freeze")
    cmd.add_argument("--candidates", required=True, type=Path)
    cmd.add_argument("--model", required=True, type=Path)
    cmd.add_argument("--swift-run-args", type=Path, help="Optionally cross-check the private original s42-A args.json")
    cmd.add_argument("--out", required=True, type=Path)
    cmd = sub.add_parser("score")
    cmd.add_argument("--panel", required=True, type=Path)
    cmd.add_argument("--panel-sha256", required=True)
    cmd.add_argument("--generation", required=True, type=Path)
    cmd.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args)
    else:
        sys.path.insert(0, str(args.bfcl_root.resolve()))
        source = args.bfcl_root / "bfcl_eval"
        for rel, expected in SOURCES.items():
            checked_sha(source / rel, expected)
        (candidates if args.action == "candidates" else score)(args, source)


if __name__ == "__main__":
    main()
