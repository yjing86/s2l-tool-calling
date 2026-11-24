"""Prepare verified prompt IDs in the training environment, then generate in vLLM.

python scripts/evaluate_experiment.py --model MODEL --data SPLIT --out RUN --prepare-only --swift-run-args ARGS
python scripts/evaluate_experiment.py --model MODEL --data SPLIT --out RUN --prepared RUN/prompts.jsonl
"""

import argparse
import collections
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback
from types import SimpleNamespace

from verify_tool_call_loss import gold_calls, swift_row


BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
CALL_RE = re.compile(r"<tool_call>\s*<function=([^<>\s]+)>\s*(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^<>\s]+)>\r?\n?(.*?)\r?\n?</parameter>\s*", re.S)


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def no_duplicates(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate key {key}")
        obj[key] = value
    return obj


def reject_constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


def typed_argument(raw, kind):
    if kind == "string":
        return raw
    if kind == "boolean" and raw in ("True", "False"):
        return raw == "True"
    try:
        return json.loads(raw, object_pairs_hook=no_duplicates,
                          parse_constant=reject_constant)
    except json.JSONDecodeError:
        return raw


def differences(actual, expected, path=(), schema=None):
    """Keep missing, extra, type and value errors at their JSON paths."""
    errors = []
    schema = schema or {}
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if (not isinstance(actual, (float, int)) or isinstance(actual, bool)
                or (schema.get("type") == "integer" and type(actual) is not int)):
            return [("type", path)]
        return [] if actual == expected else [("value", path)]
    if type(actual) is not type(expected):
        return [("type", path)]
    if isinstance(expected, dict):
        for key in sorted(expected.keys() ^ actual.keys()):
            errors.append(("key", path + (key,)))
        for key in sorted(expected.keys() & actual.keys()):
            errors.extend(differences(actual[key], expected[key], path + (key,), schema.get("properties", {}).get(key)))
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            errors.append(("value", path))
        for index, (a, e) in enumerate(zip(actual, expected)):
            errors.extend(differences(a, e, path + (index,), schema.get("items")))
    elif actual != expected:
        errors.append(("value", path))
    return errors


def parse_calls(text, tools):
    """Parse the pinned Qwen tool-call serialization using only the candidate schemas."""
    schemas = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in tools}
    text = text.strip()
    first = text.find("<tool_call>")
    if first == -1:
        if any(marker in text for marker in ("<function=", "<parameter=", "</tool_call>", "<tool_call")):
            raise ValueError("Malformed tool-call wrapper")
        return []
    # The model card permits a prose preamble; malformed tool markup ahead of a call is an error.
    if any(token in text[:first] for token in ("<function=", "<tool_call", "</tool_call>", "<parameter=")):
        raise ValueError("Malformed call prefix")
    calls = []
    remaining = text[first:]
    while remaining:
        match = CALL_RE.match(remaining)
        if not match:
            raise ValueError("Malformed function call or trailing text")
        name, body = match.groups()
        args = {}
        while body:
            param = PARAM_RE.match(body)
            if not param or param[1] in args:
                raise ValueError("Malformed or duplicate argument")
            key, raw = param.groups()
            kind = schemas.get(name, {}).get(key, {}).get("type")
            args[key] = typed_argument(raw, kind)
            body = body[param.end():]
        calls.append({"name": name, "arguments": args})
        remaining = remaining[match.end():].strip()
    return calls


def judge(row, text):
    gold = gold_calls(row)
    if len(gold) != 1:
        raise ValueError(f"Expected a single reference call in {row['id']}")
    result = {"correct": False, "reason": None, "parsed": None}
    try:
        calls = parse_calls(text, row["tools"])
    except ValueError as exc:
        return {**result, "reason": "parse", "parse_error": str(exc)}
    result["parsed"] = calls
    if len(calls) != 1:
        result["reason"] = "count"
    elif calls[0]["name"] != gold[0]["function"]["name"]:
        result["reason"] = "function"
    else:
        schema = next(t["function"]["parameters"] for t in row["tools"] if t["function"]["name"] == calls[0]["name"])
        errs = differences(calls[0]["arguments"], gold[0]["function"]["arguments"], schema=schema)
        result["argument_errors"] = errs
        if errs:
            result["reason"] = "argument"
        else:
            result.update(correct=True, reason="correct")
    return result


def row_reader(p):
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def versions():
    out = {"python": sys.version, "platform": platform.platform()}
    for key in ("vllm", "torch", "transformers", "peft", "ms-swift"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            out[key] = importlib.metadata.version(key)
    return out


def prepare(args, rows, cfg):
    from swift import get_processor, get_template
    processor = get_processor(str(args.model), model_type="qwen3_5", use_hf=True, download_model=False)
    if args.swift_run_args:
        from swift.arguments.base_args.template_args import TemplateArguments
        effective = json.loads(args.swift_run_args.read_text())
        kwargs = TemplateArguments.get_template_kwargs(SimpleNamespace(**effective))
    else:
        kwargs = dict(template_type="qwen3_5", agent_template="qwen3_5",
                      enable_thinking=False, loss_scale="default+ignore_empty_think")
    template = get_template(processor, **kwargs)
    eos = processor.tokenizer.eos_token
    path = args.out / "prompts.jsonl"
    cfg.update(prepare_versions=versions(), template_kwargs={k: str(v) for k, v in kwargs.items()})
    # Read and compare the reference serialization, but always feed inference-only IDs to vLLM.
    with path.with_suffix(".partial").open("w") as f:
        for row in rows:
            tools = json.dumps(row["tools"], ensure_ascii=False)
            messages = row["messages"] if args.generate_only else row["messages"][:-1]
            if not messages or any(m["role"] not in ("system", "user") for m in messages):
                raise ValueError(f"{row['id']}: expected a prompt without assistant messages")
            template.set_mode("transformers")
            ids = template.encode({"tools": tools, "messages": messages})["input_ids"]
            if not args.generate_only:
                template.set_mode("train")
                train = template.encode(swift_row(row))
                boundary = len(ids)
                train_ids, labels = train["input_ids"], train["labels"]
                if (not ids or ids != train_ids[:boundary] or
                        any(v != -100 for v in labels[:boundary]) or
                        boundary >= len(labels) or labels[boundary] == -100):
                    raise ValueError(f"{row['id']}: generation prefix differs from native answer boundary")
                answer = processor.tokenizer.decode(train_ids[boundary:], skip_special_tokens=False)
                answer, sep, tail = answer.partition(eos)
                if not sep or tail.strip() or not judge(row, answer)["correct"]:
                    raise ValueError(f"{row['id']}: gold assistant output failed parser roundtrip")
            record = {"id": row["id"], "row_sha256": digest_json(row), "prompt_token_ids": ids}
            print(json.dumps(record, separators=(",", ":")), file=f)
    path.with_suffix(".partial").replace(path)
    write_json(args.out / "prepare.json", {**cfg, "prompts_sha256": fingerprint(path)})
    return path


def audit_first_tokens(args, cache, cfg):
    """Compare the shortest and longest prompts against a separate HF/PEFT forward."""
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    preds = row_reader(args.out / "predictions.jsonl")
    pred_by_id = {p["id"]: p for p in preds}
    prompts = sorted(row_reader(cache), key=lambda r: len(r["prompt_token_ids"]))
    selected = {r["id"]: r for r in (prompts[0], prompts[-1])}
    with (args.out / "config.json").open() as f:
        gpu_run = json.load(f)
    if (gpu_run["prompts_sha256"] != fingerprint(cache)
            or gpu_run["source_sha256"] != cfg["source_sha256"]
            or gpu_run["model_revision"] != args.model_revision):
        raise ValueError("HF audit must use the same prompts, split and base revision")
    hashes = {p.name: fingerprint(p) for p in args.adapter.glob("adapter*") if p.is_file()} if args.adapter else {}
    if hashes != gpu_run["adapter_sha256"]:
        raise ValueError("HF audit adapter differs from vLLM")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(args.model), dtype=torch.bfloat16)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model = model.to("cuda").eval()
    results = []
    with torch.inference_mode():
        for key, record in selected.items():
            ids = torch.tensor([record["prompt_token_ids"]], device="cuda")
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1]
            values, indices = torch.topk(logits.float(), k=2)
            gpu_ids = pred_by_id[key]["output_token_ids"]
            results.append({"id": key, "hf_first_token": indices[0].item(),
                            "vllm_first_token": gpu_ids[0] if gpu_ids else None,
                            "hf_logit_margin": (values[0] - values[1]).item(),
                            "matched": bool(gpu_ids and indices[0].item() == gpu_ids[0])})
    write_json(args.out / "first-token-audit.json", {"checks": results, "versions": versions()})
    if not all(row["matched"] for row in results):
        raise ValueError("Greedy first tokens differ; see first-token-audit.json")


def generate(args, rows, cache, cfg):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    cached = row_reader(cache)
    if len(cached) != len(rows):
        raise ValueError("Cache/split length mismatch")
    prep = json.loads((cache.parent / "prepare.json").read_text())
    if (prep["prompts_sha256"] != fingerprint(cache) or prep["tokenizer_sha256"] != cfg["tokenizer_sha256"]
            or prep["template_sha256"] != cfg["template_sha256"] or prep["source_sha256"] != cfg["source_sha256"]):
        raise ValueError("Prepared cache provenance mismatch")
    by_length = []
    for row, rec in zip(rows, cached):
        if row["id"] != rec["id"] or digest_json(row) != rec["row_sha256"]:
            raise ValueError(f"{row['id']}: cached prompt belongs to different input")
        by_length.append((len(rec["prompt_token_ids"]), row, rec))
    if not by_length or max(n for n, _, _ in by_length) + args.max_tokens > args.max_model_len:
        raise ValueError("Prompt plus generation limit exceeds context; increase --max-model-len")
    adapter = None
    hashes = {}
    if args.adapter:
        from vllm.lora.request import LoRARequest
        for file in sorted(args.adapter.glob("adapter*")):
            if file.is_file():
                hashes[file.name] = fingerprint(file)
        if "adapter_config.json" not in hashes or not any(n.endswith((".safetensors", ".bin")) for n in hashes):
            raise ValueError("Checkpoint lacks adapter weights or config")
        key = digest_json(hashes)
        adapter = LoRARequest(key[:16], int(key[:8], 16) + 1, str(args.adapter.resolve()))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    eos = tokenizer.eos_token_id
    cfg.update(adapter_sha256=hashes, prompts_sha256=fingerprint(cache), stop_token_ids=[eos],
               greedy=True, seed=42, dtype="bfloat16", gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    write_json(args.out / "config.json", cfg)
    start = time.monotonic()
    engine = LLM(model=str(args.model), dtype="bfloat16", enable_lora=bool(adapter),
                 max_model_len=args.max_model_len, seed=42, gpu_memory_utilization=args.gpu_memory_utilization,
                 generation_config="vllm", enforce_eager=args.enforce_eager)
    load_s = time.monotonic() - start
    params = SamplingParams(temperature=0, n=1, max_tokens=args.max_tokens,
                            stop_token_ids=[eos], skip_special_tokens=False)
    ordered = sorted(by_length, key=lambda x: x[0])
    counts = collections.Counter()
    with (args.out / "predictions.jsonl").open("x", buffering=1) as pred, (args.out / "batches.jsonl").open("x", buffering=1) as batches:
        for offset in range(0, len(ordered), args.batch_size):
            group = ordered[offset:offset + args.batch_size]
            requests = [{"prompt_token_ids": r["prompt_token_ids"]} for n, row, r in group]
            tick = time.monotonic()
            outputs = engine.generate(requests, params, lora_request=adapter, use_tqdm=False)
            sec = time.monotonic() - tick
            if len(outputs) != len(group):
                raise RuntimeError("vLLM returned the wrong number of requests")
            out_tokens = 0
            for (n, row, rec), out in zip(group, outputs):
                if list(out.prompt_token_ids) != rec["prompt_token_ids"]:
                    raise RuntimeError("vLLM prompt mismatch")
                seq = out.outputs[0]
                ids = list(seq.token_ids)
                # Preserve tool markers even if this tokenizer defines them as special tokens.
                stripped = ids[:-1] if ids and ids[-1] == eos else ids
                text = tokenizer.decode(stripped, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                if args.generate_only:
                    try:
                        result = {"parsed": parse_calls(text, row["tools"]), "reason": "generated"}
                    except ValueError as exc:
                        result = {"parsed": None, "reason": "parse", "parse_error": str(exc)}
                else:
                    result = judge(row, text)
                counts[result["reason"]] += 1
                out_tokens += len(ids)
                record = {"id": row["id"], "group_id": row.get("group_id"), "raw_output": text,
                          "output_token_ids": ids, "input_tokens": n, "output_tokens": len(ids),
                          "finish_reason": seq.finish_reason, "stop_reason": seq.stop_reason, **result}
                print(json.dumps(record, ensure_ascii=False), file=pred)
            print(json.dumps({"size": len(group), "shortest": group[0][0], "longest": group[-1][0],
                              "input_tokens": sum(n for n, _, _ in group), "output_tokens": out_tokens,
                              "seconds": sec, "requests_per_s": len(group) / sec}), file=batches)
    report = {"count": len(rows), "correct": counts["correct"], "accuracy": counts["correct"] / len(rows),
              "reason_counts": dict(counts), "engine_load_seconds": load_s, "wall_seconds": time.monotonic() - start}
    if args.generate_only:
        del report["correct"], report["accuracy"]
    write_json(args.out / "metrics.json", report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--model-revision", default=BASE_REVISION)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--swift-run-args", type=Path)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--hf-audit", action="store_true", help="After generation, check HF/PEFT first tokens using the training environment")
    p.add_argument("--prepared", type=Path)
    p.add_argument("--generate-only", action="store_true", help="Generate for prompt-only BFCL inputs; use the external scorer")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--enforce-eager", action="store_true", help="Skip CUDA graph compilation for short runs")
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    if args.batch_size <= 0 or args.max_tokens <= 0:
        p.error("Positive --batch-size and --max-tokens required")
    if args.hf_audit and not args.prepared:
        p.error("--hf-audit requires --prepared pointing to the original token IDs")
    if not args.hf_audit and (args.out / "predictions.jsonl").exists():
        p.error("Output already contains predictions; specify a new attempt directory")
    start = time.time()
    cfg = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "started_unix": start, "versions": versions(), "script_sha256": fingerprint(__file__),
           "model_revision": args.model_revision, "source_sha256": fingerprint(args.data),
           "tokenizer_sha256": fingerprint(args.model / "tokenizer.json"),
           "template_sha256": fingerprint(args.model / "chat_template.jinja")}
    try:
        rows = row_reader(args.data)
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Empty data or duplicate IDs")
        if args.prepared is None:
            cache = prepare(args, rows, cfg)
        else:
            cache = args.prepared
        if args.hf_audit:
            audit_first_tokens(args, cache, cfg)
        elif not args.prepare_only:
            generate(args, rows, cache, cfg)
    except Exception:
        write_json(args.out / "error.json", {**cfg, "wall_seconds": time.time() - start,
                                             "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
