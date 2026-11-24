"""Measure synchronized warm training and proxy scoring using only candidate-pool rows."""
import argparse
from collections import defaultdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from score_trajectories import atomic_json, batched_scores, file_hash
from verify_tool_call_loss import encode, swift_row


def put(path, record):
    print(json.dumps(record, allow_nan=False), flush=True)
    with path.open("a") as f:
        print(json.dumps(record, allow_nan=False), file=f)


def register():
    import torch
    from swift.callbacks import TrainerCallback, callbacks_map

    class WarmStep(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.out = Path(os.environ["TOOL_BENCH_OUT"])
            self.count = defaultdict(int)
            params = [(n, p.numel()) for n, p in kwargs["model"].named_parameters() if p.requires_grad]
            if not params or any("lora_" not in n or "visual." in n for n, _ in params):
                raise RuntimeError("Unexpected trainable parameter set")
            atomic_json(self.out / "runtime.json", {"gpu": torch.cuda.get_device_name(),
                        "trainable": sum(v for n, v in params), "dtype": str(torch.bfloat16),
                        "cuda": torch.version.cuda, "model_params": params})
            original = self.trainer.training_step

            def counted(model, inputs, *pos, **kw):
                ids = inputs["input_ids"]
                self.count["examples"] += ids.shape[0]
                self.count["padded_tokens"] += ids.numel()
                mask = inputs.get("attention_mask")
                if mask is None and ids.shape[0] > 1:
                    raise ValueError("Missing multi-example attention mask")
                self.count["tokens"] += mask.sum().item() if mask is not None else ids.numel()
                self.count["answer_tokens"] += (inputs["labels"] != -100).sum().item()
                return original(model, inputs, *pos, **kw)

            self.trainer.training_step = counted

        def on_step_begin(self, args, state, control, **kwargs):
            self.count.clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            put(self.out / "steps.jsonl", {"update": state.global_step, "warmup": state.global_step == 1,
                    "seconds": time.perf_counter()-self.start, **self.count,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated()})

    callbacks_map["tool_bench_warm"] = WarmStep


def sample_rows(train, audit, n):
    lengths = {r["id"]: r["full_tokens"] for r in map(json.loads, audit.open()) if r.get("split") == "train"}
    rows = list(map(json.loads, train.open()))
    # Accept reduced candidate pools only when every ID belongs to the frozen training split.
    if len({r["id"] for r in rows}) != len(rows) or any(r["id"] not in lengths for r in rows):
        raise ValueError("Expected distinct, train-only candidate IDs")
    if len(rows) < n:
        raise ValueError("Not enough training rows")
    rows.sort(key=lambda r:(lengths[r["id"]], r["id"]))
    sample = [rows[(2*i+1)*len(rows)//(2*n)] for i in range(n)]
    return sample, {r["id"]: lengths[r["id"]] for r in sample}, len(rows)


def projection(timings, examples):
    total = sum(t["seconds"] for t in timings)
    count = sum(t["examples"] for t in timings)
    unit = [t["seconds"]/t["examples"] for t in timings]
    return {"rows_per_second": count / total, "projected_examples": examples,
            "warm_hours_at_pooled_rate": examples*total/count/3600,
            "warm_hours_at_fastest_and_slowest_observed_batch": [examples*x/3600 for x in (min(unit),max(unit))],
            "scope": "Warm compute only. Excludes setup, checkpoints, evaluation and failures. Observed range is not a statistical bound."}


def train_benchmark(args, data, pool_size):
    output = args.out / "swift"
    command = ["swift", "sft", "--model", str(args.model), "--dataset", str(data),
        "--model_type", "qwen3_5", "--template", "qwen3_5", "--agent_template", "qwen3_5",
        "--tuner_type", "lora", "--target_modules", "up_proj", "down_proj", "--freeze_vit", "true",
        "--lora_rank", "16", "--lora_alpha", "32", "--lora_dropout", "0.05", "--loss_scale", "default",
        "--enable_thinking", "false", "--use_hf", "true", "--use_cpu", "false", "--torch_dtype", "bfloat16",
        "--bf16", "true", "--fp16", "false", "--gradient_checkpointing", "true", "--attn_impl", "sdpa",
        "--optim", "adamw_torch", "--learning_rate", "0.0001", "--lr_scheduler_type", "constant_with_warmup",
        "--warmup_steps", "5", "--split_dataset_ratio", "0", "--strict", "true", "--truncation_strategy", "delete",
        "--max_length", "4096", "--per_device_train_batch_size", str(args.micro_batch),
        "--gradient_accumulation_steps", str(16//args.micro_batch), "--max_steps", str(1+args.updates),
        "--save_strategy", "no", "--logging_steps", "1", "--logging_nan_inf_filter", "false",
        "--seed", "42", "--data_seed", "42", "--report_to", "none", "--check_model", "false",
        "--add_version", "false", "--output_dir", str(output),
        "--external_plugins", str(Path(__file__).resolve()), "--callbacks", "tool_bench_warm"]
    (args.out / "command.txt").write_text(shlex.join(command)+"\n")
    env = dict(os.environ, TOOL_BENCH_OUT=str(args.out), PYTHONUNBUFFERED="1")
    with (args.out / "swift.log").open("w") as log:
        proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if proc.wait():
            raise RuntimeError(f"Benchmark training failed (exit {proc.returncode})")
    steps = list(map(json.loads,(args.out/"steps.jsonl").open()))
    timed = [s for s in steps if not s["warmup"]]
    if len(timed) != args.updates or any(s["examples"]!=16 for s in steps):
        raise RuntimeError("Warm training counts differ from expected effective batch")
    study_rows = pool_size if args.stage == "proxy" else 2 * (pool_size//4)
    result = projection(timed, study_rows)
    atomic_json(args.out / "training-projection.json", result)
    print(json.dumps(result), flush=True)


def score_benchmark(args, sample, pool_size):
    import torch
    from swift import get_processor, get_template
    from swift.arguments.base_args.template_args import TemplateArguments
    from transformers import Qwen3_5ForConditionalGeneration
    cfg = json.loads(args.swift_args.read_text())
    processor = get_processor(str(args.model), model_type="qwen3_5", use_hf=True, download_model=False)
    template = get_template(processor, **TemplateArguments.get_template_kwargs(SimpleNamespace(**cfg)))
    template.set_mode("train")
    tokenizer = processor.tokenizer
    rows = [(r["id"], *encode(tokenizer, r, template)) for r in sample]
    rows.sort(key=lambda r: len(r[1]))
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(args.model), dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter)).eval()
    batches=[rows[i:i+args.micro_batch] for i in range(0,len(rows),args.micro_batch)]
    timings=[]
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.inference_mode():
        for i,batch in enumerate([batches[len(batches)//2],*batches]):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            records=batched_scores(model,batch,pad,"cuda")
            torch.cuda.synchronize()
            stats={"warmup":i==0,"seconds":time.perf_counter()-start,"examples":len(batch),
                   "tokens":sum(r["input_tokens"] for r in records),"peak_cuda_bytes":torch.cuda.max_memory_allocated(),
                   "ids":[r["id"] for r in records]}
            put(args.out/"score.jsonl",stats)
            if i:
                timings.append(stats)
    result=projection(timings,5*pool_size)
    result["adapter"]=str(args.adapter) if args.adapter else None
    atomic_json(args.out/"score-projection.json",result)
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["proxy","target","score"],required=True)
    for name in ["train","audit","model","out"]:
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--micro-batch",type=int,choices=[1,2,4,8],default=4)
    p.add_argument("--rows",type=int,choices=[16,32],default=32)
    p.add_argument("--updates",type=int,choices=[1,2,3,4],default=2)
    p.add_argument("--swift-args",type=Path,help="A real swift training args.json, required in score mode")
    p.add_argument("--adapter",type=Path,help="Optional real proxy adapter to include in scoring benchmark")
    args=p.parse_args()
    if args.stage=="score" and not args.swift_args:
        p.error("--stage score requires --swift-args")
    if args.stage!="score" and not shutil.which("swift"):
        p.error("Activate the training environment")
    args.model=args.model.resolve()
    args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    sample,lens,pool_size=sample_rows(args.train,args.audit,args.rows)
    cfg={"stage":args.stage,"micro_batch":args.micro_batch,"pool_size":pool_size,"selected_id_lengths":lens,
         "model":str(args.model),"model_config_sha256":file_hash(args.model/"config.json"),"train_sha256":file_hash(args.train),
         "code_sha256":file_hash(__file__),"start_unix":time.time(),"python":sys.version,
         "packages":{n:version(n) for n in ("ms-swift","torch","peft","transformers")}}
    atomic_json(args.out/"config.json",cfg)
    if args.stage=="score":
        score_benchmark(args,sample,pool_size)
    else:
        data=args.out/"sample.swift.jsonl"
        data.write_text("".join(json.dumps(swift_row(r),ensure_ascii=False)+"\n" for r in sample))
        train_benchmark(args,data,pool_size)


if __name__ == "__main__":
    main()
elif os.environ.get("TOOL_BENCH_OUT"):
    register()
