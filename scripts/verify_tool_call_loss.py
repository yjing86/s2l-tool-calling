"""Measure teacher-forced Qwen3.5 tool-call losses on single-turn examples."""

import argparse
import ast
import json
import math
import random
import re
from pathlib import Path
from types import SimpleNamespace


FUNCTION = re.compile(r"<function=([^<>\n]+)>\n")
PARAMETER = re.compile(r"<parameter=([^<>\n]+)>\n(.*?)\n</parameter>\n", re.DOTALL)
JSON_SCALAR = re.compile(r'"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null')
DEMO = {
    "id": "demo",
    "tools": [{"type": "function", "function": {
        "name": "track_order", "description": "Track the delivery of an order.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]},
    }}],
    "messages": [
        {"role": "user", "content": "Please track order 4821."},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
            "name": "track_order", "arguments": {"order_id": "4821"},
        }}]},
    ],
}


def gold_calls(example):
    messages = example["messages"]
    if not messages or any(m["role"] not in ("system", "user") for m in messages[:-1]):
        raise ValueError("Expected a single gold assistant turn after the prompt")
    answer = messages[-1]
    if answer["role"] != "assistant" or answer.get("content") or not answer.get("tool_calls"):
        raise ValueError("Expected a gold assistant tool call with empty content")
    return answer["tool_calls"]


def swift_row(example):
    calls = gold_calls(example)
    return {
        "messages": example["messages"][:-1] + [
            {"role": "tool_call", "content": json.dumps(c["function"], ensure_ascii=False)}
            for c in calls],
        "tools": json.dumps(example["tools"], ensure_ascii=False),
    }


def turnstile_examples(directory):
    """Reconstruct the two audited fixtures from the official release."""
    definitions = json.loads((directory / "api_definitions.json").read_text(encoding="utf-8"))

    def schema(value):
        if isinstance(value, list):
            return [schema(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: schema(item) for key, item in value.items()}
        if isinstance(result.get("type"), str):
            result["type"] = {"float": "number", "dict": "object"}.get(result["type"], result["type"])
        return result

    result = []
    with (directory / "data.jsonl").open(encoding="utf-8") as stream:
        for idx, line in enumerate(stream):
            if idx > 49:
                break
            if idx not in (42, 49):
                continue
            row = json.loads(line)
            names = row["api_names"] + row["distractors"]
            random.Random(idx).shuffle(names)
            tools = [{"type": "function", "function": {
                key: schema(definitions[name][key]) for key in ("name", "description", "parameters")
            }} for name in names]
            messages, calls = [], []
            for entry in row["interaction"]:
                for role in ("SYSTEM", "USER"):
                    if role in entry:
                        messages.append({"role": role.lower(), "content": entry[role]})
                if "API_CALL" in entry:
                    call = ast.parse(entry["API_CALL"], mode="eval").body
                    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.args:
                        raise ValueError("Unexpected reference call in Turnstile fixture")
                    arguments = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
                    calls.append({"type": "function", "function": {"name": call.func.id, "arguments": arguments}})
            messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            result.append({"id": str(idx), "tools": tools, "messages": messages})
    if len(result) != 2:
        raise ValueError("Turnstile snapshot does not contain both fixtures")
    return result


def expect(text, offset, literal):
    if not text.startswith(literal, offset):
        raise ValueError(f"Unexpected Qwen tool-call serialization at character {offset}")
    return offset + len(literal)


def answer_spans(text, start, calls, eos):
    """Walk the known gold structure and check the rendered names and values."""
    cursor = start
    fields = {"function": [], "value": []}
    for index, call in enumerate(calls):
        cursor = expect(text, cursor, ("\n" if index else "") + "<tool_call>\n")
        match = FUNCTION.match(text, cursor)
        gold = call["function"]
        if not match or match[1] != gold["name"]:
            raise ValueError("Rendered function name differs from gold")
        fields["function"].append(match.span(1))
        cursor = match.end()
        for key, value in gold["arguments"].items():
            param = PARAMETER.match(text, cursor)
            if not param or param[1] != key:
                raise ValueError("Rendered parameter name differs from gold")
            raw = param[2]
            if isinstance(value, (list, dict)):
                if json.loads(raw) != value:
                    raise ValueError("Nested argument differs from gold")
                for leaf in JSON_SCALAR.finditer(raw):
                    if raw[leaf.end():].lstrip().startswith(":"):
                        continue  # JSON object key
                    quote = int(leaf[0].startswith('"'))
                    left = param.start(2) + leaf.start() + quote
                    right = param.start(2) + leaf.end() - quote
                    if left < right:
                        fields["value"].append((left, right))
            else:
                spellings = {str(value)}
                if isinstance(value, bool) or value is None:
                    spellings.add(json.dumps(value))
                if raw not in spellings:
                    raise ValueError("Scalar argument differs from gold")
                if raw:
                    fields["value"].append(param.span(2))
            cursor = param.end()
        cursor = expect(text, cursor, "</function>\n</tool_call>")
    end = expect(text, cursor, eos)
    if text[end:].strip():
        raise ValueError("Unexpected content after the assistant answer")
    return fields, end


def assign_fields(offsets, supervised, fields):
    by_field = {"function": [], "value": [], "other": []}
    for pos in supervised:
        left, right = offsets[pos]
        matched = [k for k, spans in fields.items()
                   if any(left < stop and right > begin for begin, stop in spans)]
        if len(matched) > 1 or pos == 0 or left == right:
            raise ValueError("Ambiguous supervised token")
        by_field[matched[0] if matched else "other"].append(pos)
    if not by_field["function"]:
        raise ValueError("Empty function-name mask")
    return by_field


def encode(tokenizer, example, swift_template=None):
    messages = example["messages"]
    calls = gold_calls(example)
    if swift_template is not None:
        native = swift_template.encode(swift_row(example))
        if native.get("loss_scale") is not None:
            raise ValueError("The verifier requires binary supervision encoded in labels")
        ids, labels = native["input_ids"], native["labels"]
        if len(ids) != len(labels) or any(y not in (-100, x) for x, y in zip(ids, labels)):
            raise ValueError("Unexpected ms-swift training labels")
        supervised = [i for i, y in enumerate(labels) if y != -100]
        if not supervised:
            raise ValueError("No supervised answer")
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if enc["input_ids"] != ids:
            raise ValueError("Re-encoding changed native training IDs; offsets cannot be reused")
        first = enc["offset_mapping"][supervised[0]][0]
        start = text.index("<tool_call>\n", first)
        fields, _ = answer_spans(text, start, calls, tokenizer.eos_token)
        return ids, assign_fields(enc["offset_mapping"], supervised, fields)
    settings = {"tools": example["tools"], "tokenize": False, "enable_thinking": False}
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **settings)
    text = tokenizer.apply_chat_template(messages, **settings)
    if not text.startswith(prompt):
        raise ValueError("Training text and inference prefix disagree")
    fields, end = answer_spans(text, len(prompt), calls, tokenizer.eos_token)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    supervised = []
    for pos, (left, right) in enumerate(enc["offset_mapping"]):
        if right <= len(prompt) or left >= end or left == right:
            continue
        if left < len(prompt) or right > end:
            raise ValueError("A token crosses the supervised answer boundary")
        supervised.append(pos)
    return enc["input_ids"], assign_fields(enc["offset_mapping"], supervised, fields)


def score(model, ids, fields, device, check_hf):
    import torch
    import torch.nn.functional as F

    tokens = torch.tensor([ids], device=device)
    positions = sorted(pos for group in fields.values() for pos in group)
    targets = torch.tensor(positions, device=device)
    if check_hf:
        labels = torch.full_like(tokens, -100)
        labels[0, targets] = tokens[0, targets]
        out = model(input_ids=tokens, labels=labels, use_cache=False)
        scores = out.logits[0].index_select(0, targets - 1)
        hf_mean = out.loss.item()
    else:
        # Compute the vocabulary projection only at the positions we need.
        out = model(input_ids=tokens, logits_to_keep=targets - 1, use_cache=False)
        scores = out.logits[0]
        hf_mean = None
    losses = F.cross_entropy(scores.float(), tokens[0, targets], reduction="none").cpu()
    if not torch.isfinite(losses).all():
        raise ValueError("Non-finite token loss")
    offsets = {pos: i for i, pos in enumerate(positions)}
    result = {"input_tokens": len(ids), "answer_tokens": len(positions), "all": losses.mean().item()}
    for field, group in fields.items():
        result[field + "_tokens"] = len(group)
        result[field] = losses[[offsets[pos] for pos in group]].mean().item() if group else None
    if hf_mean is not None:
        result["hf_loss"] = hf_mean
        if not math.isclose(result["all"], hf_mean, rel_tol=0, abs_tol=0.0001):
            raise ValueError(f"Mean token loss and the model loss disagree: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Local Qwen3.5 weights or Hugging Face model ID")
    parser.add_argument("--adapter", help="Optional PEFT adapter checkpoint matching the base")
    parser.add_argument("--encoder", choices=["hf", "swift"], default="hf",
                        help="Use swift for a checkpoint trained with ms-swift's native template")
    parser.add_argument("--swift-run-args", type=Path, help="Effective args.json saved by the native swift sft run")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--examples", type=Path, help="JSONL with id, tools, messages (HF tool_calls)")
    source.add_argument("--turnstile-dir", type=Path, help="Directory with official data.jsonl and api_definitions.json; reads rows 42 and 49")
    parser.add_argument("--write-swift-jsonl", type=Path, help="Export selected examples for swift sft and exit without loading weights")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--check-hf", action="store_true", help="Also compute the native full-logit loss; use on short inputs")
    args = parser.parse_args()
    rows = [DEMO]
    if args.examples is not None:
        with args.examples.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    elif args.turnstile_dir is not None:
        rows = turnstile_examples(args.turnstile_dir)
    if not rows:
        parser.error("Input contains no examples")
    if args.write_swift_jsonl:
        lines = [json.dumps(swift_row(row), ensure_ascii=False) for row in rows]
        with args.write_swift_jsonl.open("x", encoding="utf-8") as stream:
            for line in lines:
                print(line, file=stream)
        return
    if not args.model:
        parser.error("--model is required for scoring")
    if args.encoder == "swift" and not args.swift_run_args:
        parser.error("--encoder swift requires --swift-run-args so scoring uses the training run's labels")
    if args.threads < 1:
        parser.error("--threads must be positive")
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    torch.set_num_threads(args.threads)
    swift_template = None
    if args.encoder == "swift":
        from swift import get_processor, get_template
        from swift.arguments.base_args.template_args import TemplateArguments

        run = json.loads(args.swift_run_args.read_text(encoding="utf-8"))
        if (run.get("loss_scale") or "").split("+")[0] not in ("default", "last_round"):
            parser.error("Use a run with answer-only loss_scale (default or last_round)")
        try:
            template_kwargs = TemplateArguments.get_template_kwargs(SimpleNamespace(**run))
        except AttributeError as error:
            parser.error(f"Incomplete saved ms-swift args.json: {error}")
        processor = get_processor(args.model, model_type="qwen3_5", use_hf=True, download_model=False)
        tokenizer = processor.tokenizer
        swift_template = get_template(processor, **template_kwargs)
        swift_template.set_mode("train")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.is_fast:
        raise ValueError("This script needs fast-tokenizer character offsets")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(args.device).eval()
    with torch.inference_mode():
        for row in rows:
            ids, fields = encode(tokenizer, row, swift_template)
            result = score(model, ids, fields, args.device, args.check_hf)
            print(json.dumps({"id": row["id"], **result}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
