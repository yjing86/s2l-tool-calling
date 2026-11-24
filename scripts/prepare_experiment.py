"""Prepare the single-call Turnstile splits. Run from any working directory."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import re
import sys
from tempfile import TemporaryDirectory
import unicodedata

from download_inputs import ROOT, SNAPSHOTS
from verify_tool_call_loss import encode

SEED = 31415
AUDITED = {42, 48, 49}
SIZES = {"pilot": 200, "train": 4000, "dev": 400, "test": 2000}
TYPES = {"array": list, "object": dict, "boolean": bool, "string": str,
         "integer": int, "number": (float, int)}


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_rows(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            print(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")), file=f)


def schema(value):
    if isinstance(value, list):
        return [schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {k: schema(v) for k, v in value.items()}
    if isinstance(result.get("type"), str):
        result["type"] = {"float": "number", "dict": "object"}.get(result["type"], result["type"])
    return result


def validate_value(value, spec):
    if "type" in spec:
        kind = spec["type"]
        if not isinstance(value, TYPES[kind]) or (type(value) is bool and kind in {"integer", "number"}):
            raise ValueError("argument type")
    if "enum" in spec and value not in spec["enum"]:
        raise ValueError("argument enum")
    if isinstance(value, list):
        for item in value:
            validate_value(item, spec.get("items", {}))
    if isinstance(value, dict) and "properties" in spec:
        properties = spec["properties"]
        if set(spec.get("required", [])) - value.keys() or value.keys() - properties.keys():
            raise ValueError("argument keys")
        for key, item in value.items():
            validate_value(item, properties[key])


def parse_call(code):
    call = ast.parse(code, mode="eval").body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.args:
        raise ValueError("call syntax")
    kwargs = {}
    for kw in call.keywords:
        if kw.arg is None or kw.arg in kwargs:
            raise ValueError("keyword arguments")
        kwargs[kw.arg] = ast.literal_eval(kw.value)
    return {"name": call.func.id, "arguments": kwargs}


def example(row, i, defs):
    names = row["api_names"] + row["distractors"]
    if len(row["api_names"]) != 1 or len(names) != 6 or len(set(names)) != 6:
        raise ValueError("candidate count")
    random.Random(i).shuffle(names)
    tools = [{"type": "function", "function": defs[n]} for n in names]
    messages = []
    calls = []
    for entry in row["interaction"]:
        for key in ("SYSTEM", "USER"):
            if key in entry:
                messages.append({"role": key.lower(), "content": entry[key]})
        if "API_CALL" in entry:
            calls.append(parse_call(entry["API_CALL"]))
    if [m["role"] for m in messages] != ["system", "user"] or len(calls) != 1:
        raise ValueError("turn or call count")
    call, = calls
    if call["name"] != row["api_names"][0]:
        raise ValueError("function label mismatch")
    validate_value(call["arguments"], defs[call["name"]]["parameters"])
    messages.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]})
    return {"id": str(i), "tools": tools, "messages": messages}


def leaves(x):
    if isinstance(x, dict):
        for y in x.values():
            yield from leaves(y)
    elif isinstance(x, list):
        for y in x:
            yield from leaves(y)
    elif isinstance(x, str):
        yield x


def words(s):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", s).casefold(), flags=re.UNICODE)


def fingerprints(query, arguments):
    # Normalize case, punctuation, copied strings, weekdays/months and numeric/ID tokens.
    raw = " ".join(words(query))
    masked = " " + raw + " "
    values = {" ".join(words(v)) for v in leaves(arguments)}
    for value in sorted(values, key=lambda v: (-len(v), v)):
        if value:
            masked = masked.replace(" " + value + " ", " SLOT ")
    masked = re.sub(r"\b\w*\d\w*\b", "SLOT", masked)
    masked = re.sub(r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", "SLOT", masked)
    return raw, masked.strip()


def assign_groups(fingerprints_by_id):
    """Connected components of exact or masked matches and high-overlap masked token trigrams."""
    parents = {i: i for i in fingerprints_by_id}

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def unite(i, j):
        a, b = sorted([root(i), root(j)])
        parents[b] = a

    exact, masked = {}, {}
    shingles = {}
    for i, (raw, template) in fingerprints_by_id.items():
        for key, index in [(raw, exact), (template, masked)]:
            unite(i, index.setdefault(key, i))
        tokens = template.split()
        if len(tokens) >= 3 and masked[template] == i:
            shingles[i] = set(zip(tokens, tokens[1:], tokens[2:]))
    freq = Counter(tri for grams in shingles.values() for tri in grams)
    index = defaultdict(list)
    for i, grams in shingles.items():
        # A pair with Jaccard >= .8 shares a shingle in both rarity-sorted prefixes.
        prefix_len = len(grams) - (4 * len(grams) + 4) // 5 + 1
        prefix = sorted(grams, key=lambda g: (freq[g], g))[:prefix_len]
        for j in {other for g in prefix for other in index[g]}:
            other = shingles[j]
            if 5 * min(len(other), len(grams)) >= 4 * max(len(other), len(grams)):
                if 5 * len(grams & other) >= 4 * len(grams | other):
                    unite(i, j)
        for gram in prefix:
            index[gram].append(i)
    return {i: str(root(i)) for i in parents}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/turnstile")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3.5-2B")
    parser.add_argument("--output", type=Path, default=ROOT / "prepared")
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--max-output", type=int, default=512)
    args = parser.parse_args()
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.model / "tokenizer.json"), eos_token="<|im_end|>")
    tokenizer.chat_template = (args.model / "chat_template.jinja").read_text()
    raw_defs = json.loads((args.source / "api_definitions.json").read_text())
    defs = {name: {k: schema(v[k]) for k in ("name", "description", "parameters")} for name, v in raw_defs.items()}
    source = args.source / "data.jsonl"
    audit = {}
    rows, fps = {}, {}
    single_count = 0
    with source.open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            single = row["interaction_template_name"] == "TURNS_1_APIS_1"
            if not single and i not in AUDITED:
                continue
            single_count += int(single)
            audit[i] = {"id": str(i), "source_row_sha256": digest(line.rstrip("\n"))}
            query = next((step["USER"] for step in row["interaction"] if "USER" in step), "")
            if not single:
                # Include audited multi-call fixtures in the duplicate-detection graph.
                calls = [parse_call(e["API_CALL"]) for e in row["interaction"] if "API_CALL" in e]
                fps[i] = fingerprints(query, [c["arguments"] for c in calls])
                audit[i]["reason"] = "audited out-of-scope"
                continue
            fps[i] = fingerprints(query, {})
            try:
                item = example(row, i, defs)
                call = item["messages"][-1]["tool_calls"][0]["function"]
                fps[i] = fingerprints(query, call["arguments"])
                ids, fields = encode(tokenizer, item)
                prompt_len = min(position for positions in fields.values() for position in positions)
                audit[i].update(full_tokens=len(ids), prompt_tokens=prompt_len, fn=call["name"])
                if len(ids) > args.max_context or prompt_len + args.max_output > args.max_context:
                    raise ValueError("context window")
                if len(ids) - prompt_len > args.max_output:
                    raise ValueError("output window")
            except (SyntaxError, ValueError, KeyError, TypeError) as exc:
                audit[i]["reason"] = str(exc) if type(exc) is ValueError else type(exc).__name__
                continue
            rows[i] = item
    groups = assign_groups(fps)
    members = defaultdict(list)
    for i, group in groups.items():
        audit[i]["group_id"] = group
        if i in rows:
            members[group].append(i)
            rows[i]["group_id"] = group
    reserved = {groups[i] for i in AUDITED if i in groups}
    if 49 not in rows:
        raise RuntimeError("Audited row 49 failed validation")
    representatives = []
    for group, ids in members.items():
        if group in reserved:
            for i in ids:
                audit[i]["reason"] = "audited group"
        else:
            rep = min(ids, key=lambda i: digest(f"{SEED}|representative|{i}"))
            representatives.append(rep)
            for i in ids:
                if i != rep:
                    audit[i]["reason"] = "duplicate of group representative"
    ordered_length = sorted(representatives, key=lambda i: (audit[i]["full_tokens"], i))
    debug = [49, ordered_length[len(ordered_length) // 10], ordered_length[9 * len(ordered_length) // 10]]
    eligible = sorted(set(representatives) - set(debug), key=lambda i: digest(f"{SEED}|split|{groups[i]}"))
    if len(eligible) < sum(SIZES.values()):
        raise ValueError(f"Only {len(eligible)} eligible groups; review fallback before running")
    splits = {"debug": debug}
    offset = 0
    for split, count in SIZES.items():
        splits[split] = eligible[offset:offset + count]
        offset += count
    for split, ids in splits.items():
        for i in ids:
            audit[i]["split"] = split
            audit[i].pop("reason", None)
    for i in eligible[offset:]:
        audit[i]["reason"] = "unused eligible group"
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output) as tmp_name:
        tmp = Path(tmp_name)
        for split, ids in splits.items():
            write_rows(tmp / f"{split}.jsonl", (rows[i] for i in ids))
        write_rows(tmp / "rows.audit.jsonl", audit.values())
        manifest = {"seed": SEED, "source_revision": SNAPSHOTS[0][2], "tokenizer_revision": SNAPSHOTS[2][2],
                    "data_source_sha256": file_hash(source), "definitions_sha256": file_hash(args.source / "api_definitions.json"),
                    "template_sha256": file_hash(args.model / "chat_template.jinja"), "tokenizer_sha256": file_hash(args.model / "tokenizer.json"),
                    "transforms": "Exact/masked query and masked word trigram Jaccard >= 0.8 components; one representative per group",
                    "preparation_script_sha256": file_hash(Path(__file__)), "python": sys.version,
                    "packages": {k: version(k) for k in ["transformers", "tokenizers", "jinja2"]},
                    "max_context": args.max_context, "max_output": args.max_output,
                    "single_candidate_count": single_count, "single_valid_count": len(rows),
                    "counts": {k: len(v) for k, v in splits.items()}, "debug_ids": debug,
                    "exclusions": dict(Counter(row["reason"] for row in audit.values() if "reason" in row)),
                    "sha256": {p.name: file_hash(p) for p in sorted(tmp.iterdir())}}
        for path in tmp.iterdir():
            path.replace(args.output / path.name)
        out = tmp / "manifest.json"
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        out.replace(args.output / out.name)
    print(json.dumps({k: manifest[k] for k in ["counts","single_candidate_count","exclusions","debug_ids"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
