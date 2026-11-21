"""Fetch and verify the pinned public Hugging Face snapshots using Python and curl."""
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1] / "local"
SNAPSHOTS = [
    ("datasets", "amazon/Turnstile-Synthetic-Domains", "70bf9e9a6234e694efa36d1ec6c207c65a695f5b", "data/turnstile"),
    ("models", "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", "models/Qwen3.5-0.8B"),
    ("models", "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", "models/Qwen3.5-2B"),
]
CURL = ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "4"]


def matches(path, entry):
    size = entry["size"]
    if not path.is_file() or path.stat().st_size != size:
        return False
    if "lfs" in entry:
        digest = hashlib.sha256()
        expected = entry["lfs"]["oid"]
    else:
        digest = hashlib.sha1(f"blob {size}\0".encode())
        expected = entry["oid"]
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def fetch(snapshot):
    kind, repo, revision, folder = snapshot
    dest = ROOT / folder
    dest.mkdir(exist_ok=True, parents=True)
    api = f"https://huggingface.co/api/{kind}/{repo}/tree/{revision}"
    entries = json.loads(subprocess.check_output([*CURL, api]))
    prefix = "datasets/" if kind == "datasets" else ""
    for entry in entries:
        name = entry["path"]
        if entry["type"] != "file" or Path(name).name != name:
            raise ValueError(f"Unexpected snapshot layout: {name}")
        target = dest / name
        if matches(target, entry):
            continue
        url = f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{name}?download=true"
        print(f"Downloading {repo}/{name}", flush=True)
        with TemporaryDirectory(dir=dest) as tmp:
            partial = Path(tmp) / name
            subprocess.run([*CURL, "--output", str(partial), url], check=True)
            if not matches(partial, entry):
                raise ValueError(f"Checksum mismatch: {partial}")
            partial.replace(target)
    print(f"Verified {repo} @ {revision} ({len(entries)} files)", flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as workers:
        list(workers.map(fetch, SNAPSHOTS))
