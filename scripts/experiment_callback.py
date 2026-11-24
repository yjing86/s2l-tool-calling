"""ms-swift 4.6.1 plugin for token accounting and protocol checkpoints."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from swift.callbacks import TrainerCallback, callbacks_map

from verify_tool_call_loss import swift_row


class ExperimentCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.out = Path(args.output_dir)
        self.run = json.loads(Path(os.environ["TOOL_TRAIN_MANIFEST"]).read_text())
        self.started = time.monotonic()
        self.input_count = self.answer_count = self.seen = 0
        self.token_goal = None
        self.save_steps = set(self.run["save_steps"])
        if not torch.cuda.is_available() or args.world_size != 1:
            raise RuntimeError("This protocol requires a single CUDA worker")
        params = [(n, p.numel()) for n, p in kwargs["model"].named_parameters() if p.requires_grad]
        if not params or any("lora_" not in n or "visual." in n for n, _ in params):
            raise RuntimeError(f"Unexpected trainable parameters: {[n for n, _ in params][:8]}")
        if len(self.trainer.train_dataset) != len(self.run["ids"]):
            raise ValueError("Training dataset count changed during preprocessing")
        if self.run["stage"] == "target":
            total = count = 0
            pool_ids = set()
            with Path(self.run["pool"]).open() as f:
                for line in f:
                    row = json.loads(line)
                    pool_ids.add(str(row["id"]))
                    encoded = self.trainer.template.encode(swift_row(row))
                    total += len(encoded["input_ids"])
                    count += 1
            if len(pool_ids) != count or not set(self.run["ids"]).issubset(pool_ids):
                raise ValueError("Target subset IDs did not match pool IDs")
            self.token_goal = math.ceil(2 * len(self.run["ids"]) * total / count)
        torch.cuda.reset_peak_memory_stats()
        info = {"gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                "trainable_parameter_count": sum(v for _, v in params), "trainable_parameters": params,
                "token_goal": self.token_goal, "starting_model_step": state.global_step}
        (self.out / "runtime.json").write_text(json.dumps(info, indent=2) + "\n")
        original = self.trainer.training_step

        def counted_step(model, inputs, *a, **kw):
            # HF passes the unsharded batch to training_step; count its non-padding
            # positions before Swift converts the attention mask for the architecture.
            mask = inputs.get("attention_mask")
            ids = inputs["input_ids"]
            if mask is None and ids.shape[0] != 1:
                raise RuntimeError("Cannot count a padded multi-example batch without a mask")
            if mask is not None and mask.shape != ids.shape:
                raise RuntimeError("Unexpected attention mask shape in non-packed trainer")
            count = ids.numel() if mask is None else int(mask.sum().item())
            supervised = int((inputs["labels"] != -100).sum().item())
            if not supervised:
                raise ValueError("Batch has no supervised answer tokens")
            loss = original(model, inputs, *a, **kw)
            if not torch.isfinite(loss).all().item():
                raise FloatingPointError("Non-finite training loss")
            self.input_count += count
            self.answer_count += supervised
            self.seen += ids.shape[0]
            return loss
        self.trainer.training_step = counted_step

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        epoch_complete = step == self.run["planned_steps_one_epoch"]
        target = self.run["stage"] == "target"
        reached = self.input_count >= self.token_goal if target else epoch_complete
        if step in self.save_steps or reached or (target and step % 50 == 0):
            control.should_save = True
        if reached:
            control.should_training_stop = True
        record = {"step": step, "examples": self.seen, "input_tokens": self.input_count,
                  "answer_tokens": self.answer_count, "wall_seconds": time.monotonic() - self.started,
                  "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "checkpoint_due": bool(control.should_save)}
        with (self.out / "steps.jsonl").open("a") as f:
            print(json.dumps(record), file=f)

    def on_train_end(self, args, state, control, **kwargs):
        if self.token_goal is not None and self.input_count < self.token_goal:
            raise RuntimeError("Epoch guard hit before common token goal")
        if self.run["stage"] != "target" and self.seen != len(self.run["ids"]):
            raise RuntimeError("Single-epoch training did not see exactly the requested row count")
        expected = self.save_steps | {state.global_step}
        snapshots = {}
        for step in sorted(expected):
            adapter = self.out / f"checkpoint-{step}" / "adapter_model.safetensors"
            snapshots[str(step)] = hashlib.sha256(adapter.read_bytes()).hexdigest()
        (self.out / "completed.json").write_text(json.dumps({
            "final_step": state.global_step, "input_tokens": self.input_count, "answer_tokens": self.answer_count,
            "seen_examples": self.seen, "adapter_sha256": snapshots}, indent=2) + "\n")


callbacks_map["tool_experiment"] = ExperimentCallback
