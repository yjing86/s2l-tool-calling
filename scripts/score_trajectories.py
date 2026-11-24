"""Measure native, teacher-forced ms-swift train-token losses on a fixed candidate pool.

One JSONL per checkpoint, plus the exact run fingerprint, timings, and a completion
receipt. Resume by reusing the same directory and arguments after interruption.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace

from verify_tool_call_loss import encode, score


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def checkpoint(spec):
    try:
        step, adapter = spec.split("=", 1)
        if int(step) < 0 or (int(step) == 0) != (adapter == "base"):
            raise ValueError
        return int(step), None if adapter == "base" else Path(adapter)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected 0=base or positive-step=adapter-directory") from error


def plan_batches(rows, batch_size, max_batch_tokens):
    batch = []
    for row in sorted(rows, key=lambda item: len(item[1])):
        size = len(row[1])
        if batch and (len(batch) == batch_size or size * (len(batch) + 1) > max_batch_tokens):
            yield batch
            batch = []
        batch.append(row)
    if batch:
        yield batch


def batched_scores(model, batch, pad_id, device):
    import torch
    import torch.nn.functional as F

    # Right padding leaves gold positions unchanged. Restrict the vocabulary
    # projection to predicted positions (one token before each supervised token).
    width = max(len(tokens) for _, tokens, _ in batch)
    tokens = torch.tensor([ids + [pad_id] * (width - len(ids)) for _, ids, _ in batch], device=device)
    lengths = torch.tensor([len(ids) for _, ids, _ in batch], device=device)
    attention = torch.arange(width, device=device)[None, :] < lengths[:, None]
    positions = sorted({p for _, _, fields in batch for group in fields.values() for p in group})
    by_pos = {pos: index for index, pos in enumerate(positions)}
    keep = torch.tensor(positions, dtype=torch.long, device=device) - 1
    output = model(input_ids=tokens, attention_mask=attention.long(), logits_to_keep=keep, use_cache=False)
    results = []
    for row_no, (row_id, ids, fields) in enumerate(batch):
        order = fields["function"] + fields["value"] + fields["other"]
        # Only one sample's selected logits at a time are promoted to float32.
        idx = [by_pos[pos] for pos in order]
        losses = F.cross_entropy(output.logits[row_no, idx].float(), tokens[row_no, order], reduction="none").detach().double().cpu()
        if not torch.isfinite(losses).all():
            raise ValueError(f"Nonfinite loss on row {row_id}")
        result = {"id": row_id, "input_tokens": len(ids), "all_tokens": len(order),
                  "all_sum": losses.sum().item()}
        offset = 0
        for field in ("function", "value", "other"):
            count = len(fields[field])
            result[field + "_tokens"] = count
            result[field + "_sum"] = losses[offset:offset + count].sum().item()
            offset += count
        results.append(result)
    return results


def completed_ids(path, expected):
    done = set()
    if path.exists():
        with path.open("rb+") as stream:
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(start)
                    break
                row = json.loads(line)
                key = row["id"]
                if key not in expected or key in done:
                    raise ValueError(f"Unexpected or duplicate scored row: {key}")
                done.add(key)
    return done


def run(args):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    from swift import get_processor, get_template
    from swift.arguments.base_args.template_args import TemplateArguments
    from peft import PeftModel

    native_args = json.loads(args.swift_run_args.read_text())
    if (native_args.get("loss_scale") or "").split("+")[0] not in ("default", "last_round"):
        raise ValueError("Unexpected training loss scale")
    processor = get_processor(str(args.model), model_type="qwen3_5", use_hf=True, download_model=False)
    tokenizer = processor.tokenizer
    template = get_template(processor, **TemplateArguments.get_template_kwargs(SimpleNamespace(**native_args)))
    template.set_mode("train")

    rows = []
    start = time.perf_counter()
    with args.train.open() as stream:
        for line in stream:
            row = json.loads(line)
            try:
                ids, masks = encode(tokenizer, row, template)
            except Exception as exc:
                raise RuntimeError(f"Encoding or field mask failed for candidate {row['id']}") from exc
            rows.append((str(row["id"]), ids, masks))
    expected = {item[0] for item in rows}
    if not rows or len(expected) != len(rows):
        raise ValueError("Expected nonempty pool with unique IDs")
    ranks = [step for step, _ in args.checkpoint]
    if len(ranks) != len(set(ranks)):
        raise ValueError("Duplicate checkpoint step")
    fingerprint = {
        "train_sha256": file_hash(args.train), "swift_args_sha256": file_hash(args.swift_run_args),
        "model": str(args.model.resolve()), "model_config_sha256": file_hash(args.model / "config.json"),
        "dtype": args.dtype, "device": args.device, "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens, "code_sha256": file_hash(__file__),
        "mask_code_sha256": file_hash(Path(__file__).with_name("verify_tool_call_loss.py")),
        "python": platform.python_version(), "packages": {x: importlib.metadata.version(x) for x in
            ("ms-swift", "transformers", "torch", "peft")}, "cuda": torch.version.cuda,
        "model_index": {p.name: [p.stat().st_size, file_hash(p)] for p in sorted(args.model.glob("*.safetensors"))},
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None}
    if not fingerprint["model_index"]:
        raise ValueError("No local safetensors model weights found")
    manifest_path = args.out / "score-config.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != fingerprint:
        raise RuntimeError("Existing scores have different inputs/settings; choose a new output directory")
    atomic_json(manifest_path, fingerprint)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    print(json.dumps({"encoded_rows": len(rows), "setup_seconds": time.perf_counter()-start}), flush=True)

    base = Qwen3_5ForConditionalGeneration.from_pretrained(str(args.model), dtype=getattr(torch, args.dtype)).to(args.device).eval()
    with torch.inference_mode():
        for step, adapter in sorted(args.checkpoint):
            prefix = args.out / f"step-{step}"
            identity = {"step": step, "adapter_files": {} if adapter is None else
                        {name: file_hash(adapter / name) for name in ("adapter_model.safetensors", "adapter_config.json")}}
            identity_path = prefix.with_suffix(".identity.json")
            if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
                raise RuntimeError(f"Different checkpoint weights for step {step}")
            atomic_json(identity_path, identity)
            path = prefix.with_suffix(".jsonl")
            done = completed_ids(path, expected)
            if len(done) == len(rows) and prefix.with_suffix(".complete.json").exists():
                continue
            model = PeftModel.from_pretrained(base, str(adapter)).eval() if adapter else base
            pending = [r for r in rows if r[0] not in done]
            checked = False
            with path.open("a") as stream, (args.out / "timings.jsonl").open("a") as timings:
                for batch in plan_batches(pending, args.batch_size, args.max_batch_tokens):
                    tick = time.perf_counter()
                    results = batched_scores(model, batch, pad, args.device)
                    if not checked:
                        # Independent unpadded forward catches shifted/right-padding mistakes.
                        rid, ids, fields = batch[0]
                        ref = score(model, ids, fields, args.device, False)
                        deltas = {}
                        for field in ("all", "function", "value"):
                            count = results[0][field + "_tokens"]
                            if count:
                                diff = abs(results[0][field + "_sum"] / count - ref[field])
                                deltas[field] = diff
                                if diff > 0.03 + 0.02 * abs(ref[field]):
                                    raise RuntimeError(f"Padded/singleton disagreement on {rid}, {field}: {diff}")
                        repeated = batched_scores(model, batch, pad, args.device)
                        repeat_max = {}
                        for field in ("all", "function", "value"):
                            repeat_max[field] = max((abs(a[field + "_sum"] - b[field + "_sum"]) /
                                                     a[field + "_tokens"]
                                                     for a, b in zip(results, repeated)
                                                     if a[field + "_tokens"]), default=None)
                        diagnostic = {"step": step, "parity_id": rid, "abs_diff": deltas,
                                      "repeat_ids": [r["id"] for r in results],
                                      "repeat_max_abs_mean_diff": repeat_max}
                        print(json.dumps(diagnostic), flush=True)
                        with prefix.with_suffix(".checks.jsonl").open("a") as checks:
                            print(json.dumps(diagnostic), file=checks)
                        checked = True
                    for record in results:
                        print(json.dumps(record, allow_nan=False), file=stream)
                        done.add(record["id"])
                    stream.flush()
                    print(json.dumps({"step": step, "completed": len(done), "batch": len(batch),
                                      "seconds": time.perf_counter()-tick, "unix_time": time.time()}), file=timings, flush=True)
                    if len(done) % 100 < len(batch):
                        print(f"step={step} scored={len(done)}/{len(rows)}", flush=True)
            if adapter:
                base = model.unload()
            atomic_json(prefix.with_suffix(".complete.json"), {"step": step, "rows": len(done), "sha256": file_hash(path)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ("model", "train", "swift-run-args", "out"):
        p.add_argument("--" + opt, type=Path, required=True)
    p.add_argument("--checkpoint", type=checkpoint, action="append", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batch-tokens", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    args = p.parse_args()
    if min(args.batch_size, args.max_batch_tokens) < 1:
        p.error("Batch limits must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".scoring.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
