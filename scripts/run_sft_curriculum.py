#!/usr/bin/env python3
"""Run the fixed Pure V4 SFT curriculum with existing train/merge scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# 阶段不再写死 a/b/c：课程清单说了算（2026-10-22 起 M10-R 用两段倒序课程）。
DEFAULT_STAGES = ("a", "b", "c")


def stage_order(manifest) -> list[str]:
    stages = manifest.get("stages") or {}
    if not stages:
        raise ValueError("课程清单里没有 stages")
    return list(stages.keys())


def build_stage_commands(
    manifest,
    *,
    manifest_path,
    source,
    base_model,
    output_root,
    python,
    start_stage=None,
    stop_after_stage=None,
    swanlab=False,
    swanlab_project="shopping-grpo-sft-curriculum",
    qlora=False,
    liger_kernel=False,
    resume_from_checkpoint=None,
    loss_scope="assistant",
    max_steps=-1,
    save_steps=0,
    save_total_limit=3,
):
    stages = stage_order(manifest)
    start_stage = start_stage or stages[0]
    stop_after_stage = stop_after_stage or stages[-1]
    for name in (start_stage, stop_after_stage):
        if name not in stages:
            raise ValueError(
                f"阶段 {name!r} 不在课程清单里（可选：{', '.join(stages)}）"
            )
    start = stages.index(start_stage)
    stop = stages.index(stop_after_stage)
    if stop < start:
        raise ValueError("--stop-after-stage must be at or after --start-stage")

    commands = []
    for index, stage in enumerate(stages[start : stop + 1]):
        stage_config = manifest["stages"][stage]
        stage_root = Path(output_root) / f"stage-{stage}"
        model = (
            base_model
            if index == 0
            else str(
                Path(output_root)
                / f"stage-{stages[stages.index(stage) - 1]}"
                / "merged"
            )
        )
        train = [
            str(python),
            str(ROOT / "scripts/train_lora_sft.py"),
            "--model",
            str(model),
            "--train",
            str(source),
            "--validation",
            str(source),
            "--curriculum-manifest",
            str(manifest_path),
            "--curriculum-stage",
            stage,
            "--loss-scope",
            str(loss_scope),
            "--output",
            str(stage_root / "adapter"),
            "--epochs",
            str(stage_config["epochs"]),
            "--learning-rate",
            str(stage_config["learning_rate"]),
            "--max-length",
            "24576",
            "--dtype",
            "bf16",
            "--attention-implementation",
            "sdpa",
            "--gradient-checkpointing",
            "--swanlab-run-name",
            f"pure-v4-stage-{stage}",
        ]
        if swanlab:
            train.extend(["--swanlab", "--swanlab-project", swanlab_project])
        if qlora:
            train.append("--qlora")
        if liger_kernel:
            train.append("--liger-kernel")
        if int(max_steps) > 0:
            train.extend(["--max-steps", str(int(max_steps))])
        # 定时关机场景要的是「中断粒度」而不是「省磁盘」：LoRA checkpoint 只有几十 MB，
        # 所以把存盘频率和保留个数都透传下去，默认仍是历史行为（epoch 末存一次）。
        if int(save_steps) > 0:
            train.extend(["--save-steps", str(int(save_steps))])
        train.extend(["--save-total-limit", str(int(save_total_limit))])
        if index == 0 and resume_from_checkpoint:
            train.extend(["--resume-from-checkpoint", str(resume_from_checkpoint)])
        merge = [
            str(python),
            str(ROOT / "scripts/merge_lora_adapter.py"),
            "--base-model",
            str(model),
            "--adapter",
            str(stage_root / "adapter"),
            "--output",
            str(stage_root / "merged"),
            "--bf16",
        ]
        commands.append({"stage": stage, "train": train, "merge": merge})
    return commands


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/sft_pure_v4/all.jsonl"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/sft_curriculum/manifest.json"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/models/sft-curriculum"
    )
    parser.add_argument(
        "--start-stage",
        default=None,
        help="从哪个阶段开始（默认课程清单的第一段；可选值由清单决定）",
    )
    parser.add_argument(
        "--stop-after-stage",
        default=None,
        help="到哪个阶段结束（默认课程清单的最后一段）",
    )
    parser.add_argument(
        "--loss-scope",
        choices=("assistant", "tool-call-only"),
        default="assistant",
        help="M5 A/B 的 loss 口径；两个臂必须使用同一课程与同一口径之外的相同超参。",
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="每多少步存一次 checkpoint（透传给每个阶段）；0=只在 epoch 末存。"
        "定时关机 / 可能中断的长跑用它换中断粒度。",
    )
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="shopping-grpo-sft-curriculum")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--liger-kernel", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="透传给每个阶段的 --max-steps；仅用于冒烟测试（-1=按 manifest 的 epochs 跑完整阶段）",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "shopping-sft-curriculum-v1":
        raise SystemExit("不支持的课程清单 schema_version")
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_sha != manifest.get("source", {}).get("sha256"):
        raise SystemExit("Pure V4 数据与课程清单 SHA256 不一致；请先重新生成并审查清单")
    try:
        commands = build_stage_commands(
            manifest,
            manifest_path=args.manifest,
            source=args.source,
            base_model=args.base_model,
            output_root=args.output_root,
            python=sys.executable,
            start_stage=args.start_stage,
            stop_after_stage=args.stop_after_stage,
            swanlab=args.swanlab,
            swanlab_project=args.swanlab_project,
            qlora=args.qlora,
            liger_kernel=args.liger_kernel,
            resume_from_checkpoint=args.resume_from_checkpoint,
            loss_scope=args.loss_scope,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    for command in commands:
        print(f"\n[stage {command['stage'].upper()}] train")
        print(shlex.join(command["train"]))
        print(f"[stage {command['stage'].upper()}] merge")
        print(shlex.join(command["merge"]))
        if args.dry_run:
            continue
        merged = args.output_root / f"stage-{command['stage']}" / "merged"
        if merged.exists() and any(merged.iterdir()):
            raise SystemExit(f"拒绝覆盖已完成阶段：{merged}")
        subprocess.run(command["train"], check=True)
        subprocess.run(command["merge"], check=True)
    last_stage = commands[-1]["stage"]
    final_stage = stage_order(manifest)[-1]
    label = "最终 GRPO 起点" if last_stage == final_stage else "本次最后阶段输出"
    print(f"\n{label}：{args.output_root / f'stage-{last_stage}/merged'}")


if __name__ == "__main__":
    main()
