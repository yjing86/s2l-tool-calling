"""Summarize the fixed 2x2 selection-seed × target-seed diagnostic on development only."""
import argparse
from collections import Counter
import json
from pathlib import Path

from summarize_course import cluster_interval, digest, indexed, paired, read_json, read_model, read_rows


# Cell order is (selection, target): (42,42), (42,43), (43,42), (43,43).
CELLS = ((42, 42), (42, 43), (43, 42), (43, 43))
CONTRASTS = [
    ("grid_mean", "四格平均 C−B", (.25, .25, .25, .25)),
    ("selection_at_target42", "训练固定 42，选样由 42 换到 43", (-1, 0, 1, 0)),
    ("selection_at_target43", "训练固定 43，选样由 42 换到 43", (0, -1, 0, 1)),
    ("target_at_selection42", "选样固定 42，训练由 42 换到 43", (-1, 1, 0, 0)),
    ("target_at_selection43", "选样固定 43，训练由 42 换到 43", (0, 0, -1, 1)),
    ("selection_marginal", "选样种子 43−42，对两种训练种子平均", (-.5, -.5, .5, .5)),
    ("target_marginal", "训练种子 43−42，对两种选样种子平均", (-.5, .5, -.5, .5)),
    ("interaction", "交互：训练 43 时的选样变化 − 训练 42 时的选样变化", (1, -1, -1, 1)),
]


def seed_contrasts(deltas, groups, boot_seed):
    """Compute the contrast per request first, then jointly resample entire request groups."""
    sizes = Counter(groups.values())
    effects = {}
    for key, description, coeff in CONTRASTS:
        totals = Counter()
        for i, g in groups.items():
            totals[g] += sum(w * deltas[cell][i] for w, cell in zip(coeff, CELLS))
        effects[key] = {"description": description, "estimate_pp": 100 * sum(totals.values()) / len(groups),
                        "group_paired_ci95_pp": cluster_interval({g: (totals[g], n) for g, n in sizes.items()}, boot_seed)}
    return effects


def summarize(primary, crossed, dev, boot_seed):
    if dev.name != "dev.jsonl":
        raise ValueError("--dev must point to the prepared dev.jsonl")
    dev_hash = digest(dev)
    if read_json(dev.parent / "manifest.json")["sha256"]["dev.jsonl"] != dev_hash:
        raise ValueError("Development file differs from the prepared split manifest")
    rows = indexed(({"id": r["id"], "group_id": r["group_id"]} for r in read_rows(dev)), "id", dev)
    if len(rows) != 400 or any(r["group_id"] is None or r["group_id"] == "" for r in rows.values()):
        raise ValueError("Expected 400 unique frozen development IDs with nonempty group IDs")
    groups = {i: str(r["group_id"]) for i, r in rows.items()}
    primary_lock = read_json(primary / "locked-models.json")["main"]
    cross_map = read_json(crossed / "crossed-models.json")
    checksums = {"dev_split": dev_hash, "primary_lock": digest(primary / "locked-models.json"),
                 "crossed_lock": digest(crossed / "crossed-models.json")}
    stats, deltas, protocol, training_identity = {}, {}, None, None
    for sel, target in CELLS:
        subset = primary / "selection" / str(sel)
        by_arm, accuracies = {}, {}
        for arm in ("B", "C"):
            diagonal = sel == target
            run_id = f"s{sel}-{arm}" if diagonal else f"{arm}-selection{sel}-target{target}"
            root = primary if diagonal else crossed
            train = root / "training" / run_id
            done = read_json(train / "completed.json")
            meta = read_json(train / "run.json")
            step = done["final_step"]
            name = f"{run_id}-step{step}"
            key = run_id if diagonal else f"select{sel}-train{target}-{arm}"
            actual = primary_lock[key] if diagonal else cross_map[key]
            if name != actual:
                raise ValueError(f"{key}: checkpoint differs from predeclared equal-token endpoint")
            if meta["stage"] != "target" or meta["seed"] != target or meta["data_sha256"] != digest(subset / f"{arm}.jsonl"):
                raise ValueError(f"{key}: training data or target seed mismatch")
            runtime = read_json(train / "runtime.json")
            goal = runtime["token_goal"]
            end_steps = list(read_rows(train / "steps.jsonl"))[-2:]
            previous = end_steps[-2]["input_tokens"] if len(end_steps) > 1 else 0
            if (not end_steps or end_steps[-1]["step"] != step
                    or end_steps[-1]["input_tokens"] != done["input_tokens"]
                    or not previous < goal <= done["input_tokens"]):
                raise ValueError(f"{key}: expected the first optimizer boundary crossing the token goal")
            identity = (meta["model_config_sha256"], meta["pool_sha256"], meta["batch"], goal)
            if training_identity is not None and training_identity != identity:
                raise ValueError(f"{key}: comparison changed training pool, model config or effective batch")
            training_identity = identity
            eval_dir = root / "evaluation" / "dev" / name
            local = {}
            pred, perf, cfg = read_model(eval_dir, groups, dev_hash, local)
            if done["adapter_sha256"][str(step)] != perf["adapter_sha256"]["adapter_model.safetensors"]:
                raise ValueError(f"{key}: evaluated adapter differs from training endpoint")
            cfg_raw = read_json(eval_dir / "config.json")
            cfg.update({k: cfg_raw[k] for k in ("seed", "language_model_only", "max_lora_rank")})
            cfg.update({k: cfg_raw["args"][k] for k in ("max_model_len", "batch_size", "enforce_eager")})
            if protocol is not None and cfg != protocol:
                raise ValueError(f"{key}: generation or scoring protocol changed")
            protocol = cfg
            checksums[key] = {"training_record": digest(train / "run.json"), "completion": digest(train / "completed.json"),
                              "selected_training_rows": meta["data_sha256"],
                              "evaluation": {Path(k).name: v for k, v in local.items()}}
            by_arm[arm] = pred
            accuracies[arm] = {"correct": perf["correct"], "accuracy_pct": perf["accuracy_pct"], "step": step,
                               "input_tokens": done["input_tokens"], "answer_tokens": done["answer_tokens"], "token_goal": goal}
        deltas[sel, target] = {i: int(by_arm["C"][i]) - int(by_arm["B"][i]) for i in groups}
        stats[f"selection{sel}-target{target}"] = {"arms": accuracies,
            "C_minus_B": paired(by_arm["C"], by_arm["B"], groups, boot_seed)}
    return {"split": "dev", "requests": len(groups), "request_groups": len(set(groups.values())), "cells": stats,
            "contrasts": seed_contrasts(deltas, groups, boot_seed),
            "bootstrap": {"seed": boot_seed, "draws": 2000, "unit": "near-duplicate request group",
                          "intervals": "pointwise paired percentile 95%; common group draws for all cells"},
            "scope": "Exploratory, conditional on the fixed two seeds per axis and this development split; no population variance decomposition.",
            "provenance": checksums, "evaluation_protocol": protocol}


def markdown(report):
    out = ["# 选样种子与训练种子的交叉诊断", "",
           f"四格都在同一份开发集上评估，含 {report['requests']} 道请求、{report['request_groups']} 个近重复题组。"
           "B 表示整答轨迹选样，C 表示分字段轨迹选样。差值以百分点计。",
           "", "| 选样种子 | 训练种子 | B 正确数 | C 正确数 | 仅 C 对 | 仅 B 对 | C−B | 95% 区间 |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, t in CELLS:
        cell = report["cells"][f"selection{s}-target{t}"]
        p = cell["C_minus_B"]
        lo, hi = p["paired_cluster_ci95_pp"]
        out.append(f"| {s} | {t} | {cell['arms']['B']['correct']} | {cell['arms']['C']['correct']} | "
                   f"{p['only_left_correct']} | {p['only_right_correct']} | {p['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    out += ["", "| 对 C−B 的比较 | 点估计（百分点） | 95% 区间 |", "|---|---:|---:|"]
    for v in report["contrasts"].values():
        lo, hi = v["group_paired_ci95_pp"]
        out.append(f"| {v['description']} | {v['estimate_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    out += ["", "每次重抽同时保留同一题组在四格、两种方法下的结果。区间来自 2,000 次题组配对重采样，各行分别计算。"
            "这项探索性诊断只覆盖 42、43 两个种子和这份固定开发集。"
            "交互衡量一个因素的变化在另一因素的两个设定下相差多少。"
            "这些数值无法估计两类随机性在总体方差中的占比，也无法代替封存考试成绩。"]
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--crossed", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--bootstrap-seed", type=int, default=31415)
    args = p.parse_args()
    report = summarize(args.primary, args.crossed, args.dev, args.bootstrap_seed)
    for name, text in (("summary.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                       ("summary.md", markdown(report))):
        dest = args.crossed / name
        temp = dest.with_suffix(".tmp")
        temp.write_text(text)
        temp.replace(dest)
    print(f"Wrote four matched seed cells and eight contrasts to {args.crossed}")


if __name__ == "__main__":
    main()
