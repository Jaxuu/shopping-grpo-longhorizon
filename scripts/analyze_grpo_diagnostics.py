#!/usr/bin/env python3
"""分析 GRPO 训练诊断文件：量化真正更新了几步、动态采样为什么重试、以及优化信号有多弱。

用途
----
M6 第一轮 50 步里 `skipped_updates_total` 累积到 32+。**注意口径**：补丁在跳过时是
`continue` 重试同一步，不是 `break` 丢步（只有连续 10 次跳过才 `RuntimeError` 中止）。
所以那 32 条 `skipped_update` 是**重试轮次**，不是丢失的梯度步。本脚本把这件事量化清楚：

  1. 步数总览：完成 optimizer 更新的步数 / 重试轮次 / 完全没有更新的步数
  2. 动态采样实测的组级有效率（与全池探针的预测值对比）
  3. 每步用掉几个 generation batch（用来判断 `max_num_gen_batches` 是不是卡在阈值上）
  4. 组被丢弃的原因分布（constant_reward / sampling_invalid / …）
  5. 训练动力学：`actor/ppo_kl`、`actor/pg_clipfrac` 等指标的首/末/极值/均值

只读文件，不加载 torch / veRL。

用法
----
    python scripts/analyze_grpo_diagnostics.py outputs/models/grpo-m6/training_diagnostics.jsonl
    python scripts/analyze_grpo_diagnostics.py <path> --json
    # 与探针结果对比（可选）
    python scripts/analyze_grpo_diagnostics.py <path> --probe-labels outputs/probe/m6r-full/difficulty_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("diagnostics", type=Path, help="training_diagnostics.jsonl 路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（供报告引用）")
    parser.add_argument(
        "--probe-labels",
        type=Path,
        default=None,
        help="可选：全池探针的 difficulty_labels.jsonl，用于交叉核对有效率",
    )
    parser.add_argument(
        "--max-gen-batches",
        type=int,
        default=None,
        help="可选：本次训练配置的 shopping_dynamic_sampling.max_num_gen_batches",
    )
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"诊断文件不存在：{path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
    if not records:
        raise SystemExit(f"诊断文件为空：{path}")
    return records


def analyse(records: list[dict]) -> dict:
    by_event = Counter(record.get("event") for record in records)

    generation_batches: dict[int, list[dict]] = defaultdict(list)
    skipped_rounds: list[int] = []
    updated_metrics: dict[int, dict] = {}

    for record in records:
        event = record.get("event")
        step = record.get("global_step")
        if event == "generation_batch":
            generation_batches[int(step)].append(record)
        elif event == "skipped_update":
            skipped_rounds.append(int(step))
        elif event == "optimizer_step":
            metrics = record.get("metrics")
            updated_metrics[int(step)] = metrics if isinstance(metrics, dict) else {}

    # 组级统计（动态采样的实测有效率）
    seen_groups = 0
    kept_groups = 0
    accepted_groups = 0
    drop_reasons: Counter[str] = Counter()
    per_batch_kept: list[float] = []
    per_batch_groups: list[int] = []
    step_targets: dict[int, int] = {}
    step_filled: dict[int, int] = {}

    for step, batches in generation_batches.items():
        for batch in batches:
            target = batch.get("target_prompts")
            if isinstance(target, int) and target > 0:
                step_targets[step] = target
            filled = batch.get("accepted_prompts_after_batch")
            if isinstance(filled, int):
                step_filled[step] = max(step_filled.get(step, 0), filled)

            groups = batch.get("groups") or []
            if not groups:
                continue
            batch_kept = 0
            for group in groups:
                seen_groups += 1
                if group.get("kept"):
                    kept_groups += 1
                    batch_kept += 1
                if group.get("accepted"):
                    accepted_groups += 1
                reason = group.get("drop_reason")
                drop_reasons[str(reason) if reason else "kept"] += 1
            per_batch_groups.append(len(groups))
            per_batch_kept.append(batch_kept / len(groups))

    batches_per_step = {step: len(batches) for step, batches in generation_batches.items()}

    # 「每步是否收满目标组数」—— 判断"过滤训练集到底有没有信号可捡"的**决定性证据**。
    # 补丁在每个 generation_batch 里写了 `accepted_prompts_after_batch`（该步累计已收满的
    # 非退化组数）与 `target_prompts`（= train_batch_size）。若**每一步都收满**，
    # 说明模型每步都拿到了完整的 4 个非退化组 ⇒ 过滤数据集**不会增加梯度信息**，只省 rollout 成本。
    target_values = Counter(step_targets.values())
    target_prompts = target_values.most_common(1)[0][0] if target_values else None
    comparable_steps = sorted(set(step_filled) & set(step_targets))
    reached_steps = [s for s in comparable_steps if step_filled[s] >= step_targets[s]]
    below_steps = [s for s in comparable_steps if step_filled[s] < step_targets[s]]
    filled_values = [step_filled[s] for s in comparable_steps]

    # ⚠️ `skipped_update` 是**重试轮次**，不是「丢掉的一步」：补丁在跳过时 reset 计数后
    # `continue`，只有连续 10 次跳过才会 RuntimeError 中止。所以「总步数」必须按
    # optimizer_step 的**不同 global_step** 数来算，不能把两者相加。
    distinct_skipped_steps = sorted(set(skipped_rounds))
    steps_with_retry = sorted(set(skipped_rounds) & set(updated_metrics))
    retry_only_steps = sorted(set(skipped_rounds) - set(updated_metrics))

    summary: dict = {
        "records": len(records),
        "events": dict(sorted(by_event.items())),
        "steps": {
            "total_steps": len(updated_metrics),
            "optimizer_updated_steps": len(updated_metrics),
            "skipped_rounds": len(skipped_rounds),
            "distinct_steps_with_retry": len(distinct_skipped_steps),
            "retry_rounds_per_step": (
                len(skipped_rounds) / len(updated_metrics) if updated_metrics else 0.0
            ),
            "steps_with_retry_that_still_updated": len(steps_with_retry),
            "steps_skipped_without_any_update": len(retry_only_steps),
            "first_skipped_step": min(skipped_rounds) if skipped_rounds else None,
            "last_skipped_step": max(skipped_rounds) if skipped_rounds else None,
        },
        "batch_fill": {
            "target_prompts": target_prompts,
            "steps_with_fill_info": len(comparable_steps),
            "steps_reached_target": len(reached_steps),
            "steps_below_target": len(below_steps),
            "below_target_steps": below_steps[:20],
            "mean_filled_per_step": (
                statistics.mean(filled_values) if filled_values else 0.0
            ),
            "min_filled_per_step": min(filled_values) if filled_values else None,
        },
        "groups": {
            "seen": seen_groups,
            "kept": kept_groups,
            "accepted": accepted_groups,
            "kept_rate": kept_groups / seen_groups if seen_groups else 0.0,
            "accepted_rate": accepted_groups / seen_groups if seen_groups else 0.0,
            "drop_reason_counts": dict(sorted(drop_reasons.items())),
        },
        "generation_batches": {
            "total": sum(batches_per_step.values()),
            "per_step_counts": dict(sorted(Counter(batches_per_step.values()).items())),
            "mean_per_step": (
                statistics.mean(batches_per_step.values()) if batches_per_step else 0.0
            ),
            "max_per_step": max(batches_per_step.values()) if batches_per_step else 0,
            "mean_kept_rate_per_batch": (
                statistics.mean(per_batch_kept) if per_batch_kept else 0.0
            ),
            "mean_groups_per_batch": (
                statistics.mean(per_batch_groups) if per_batch_groups else 0.0
            ),
        },
        "optimizer_metrics": summarize_metrics(updated_metrics),
    }
    return summary


INTERESTING_METRICS = (
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "actor/pg_clipfrac_lower",
    "actor/grad_norm",
    "actor/entropy",
    "actor/loss",
    "group/effective_ratio",
    "group/all_equal_ratio",
    "group/resample_batches",
    "reward/terminal_utility_mean",
    "critic/score/mean",
)


def summarize_metrics(per_step: dict[int, dict]) -> dict:
    """对 optimizer_step 的数值指标做首/末/极值/均值汇总（只看关心的键）。"""
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for step in sorted(per_step):
        for key, value in (per_step[step] or {}).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number != number:  # NaN
                continue
            series[str(key)].append((step, number))

    def stat(points: list[tuple[int, float]]) -> dict:
        values = [value for _, value in points]
        return {
            "first_step": points[0][0],
            "first": values[0],
            "last_step": points[-1][0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
        }

    picked = {key: stat(points) for key, points in series.items() if key in INTERESTING_METRICS}
    return {
        "num_steps_with_metrics": len(per_step),
        "available_metric_keys": len(series),
        "picked": picked,
    }


def compare_with_probe(summary: dict, labels_path: Path) -> dict:
    rows = []
    with Path(labels_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    total = len(rows)
    eligible = sum(1 for row in rows if row.get("eligible"))
    varying = sum(1 for row in rows if row.get("reward_varying"))
    return {
        "probe_tasks": total,
        "probe_reward_varying": varying,
        "probe_reward_varying_rate": varying / total if total else 0.0,
        "probe_eligible": eligible,
        "probe_eligible_rate": eligible / total if total else 0.0,
        "training_kept_rate": summary["groups"]["kept_rate"],
    }


def render(summary: dict, probe: dict | None, max_gen_batches: int | None) -> str:
    steps = summary["steps"]
    groups = summary["groups"]
    batches = summary["generation_batches"]
    total_steps = steps["total_steps"]
    wasted_batches = steps["skipped_rounds"] * (max_gen_batches or 0)

    # 章节编号动态生成，避免可选章节缺失时出现「4 → 6」的断号
    section = {"n": 0}

    def heading(title: str) -> str:
        section["n"] += 1
        return f"## {section['n']}. {title}"

    lines = [
        "# GRPO 训练诊断分析",
        "",
        f"- 记录数：{summary['records']}；事件：{summary['events']}",
        "",
        heading("步数总览"),
        "",
        "> ⚠️ **`skipped_update` 是「重试轮次」，不是「丢掉的一步」**：补丁在跳过时 reset 计数后 "
        "`continue`，只有**连续 10 次**跳过才会 `RuntimeError` 中止。所以总步数按 "
        "`optimizer_step` 的不同 `global_step` 数算，**不能把两者相加**。",
        "",
        "| 项 | 值 |",
        "|---|---:|",
        f"| **总步数（完成 optimizer 更新）** | **{total_steps}** |",
        f"| 重试轮次（`skipped_update` 记录数） | {steps['skipped_rounds']} |",
        f"| 平均每步重试轮次 | {steps['retry_rounds_per_step']:.2f} |",
        f"| 发生重试但仍更新成功的步 | {steps['steps_with_retry_that_still_updated']} |",
        f"| **完全没有任何更新的步** | **{steps['steps_skipped_without_any_update']}** |",
        f"| 首个 / 末个重试步 | {steps['first_skipped_step']} / {steps['last_skipped_step']} |",
        "",
        "> 如果「完全没有任何更新的步」是 0，说明**每一步都产出了梯度**；"
        "重试只浪费 rollout，不损失梯度步。",
        "",
    ]

    fill = summary.get("batch_fill") or {}
    if fill.get("steps_with_fill_info"):
        target = fill["target_prompts"]
        all_reached = fill["steps_below_target"] == 0
        lines += [
            heading("每步是否收满目标组数（判断「过滤训练集有没有信号可捡」）"),
            "",
            "> 补丁在每个 `generation_batch` 里记录 `accepted_prompts_after_batch`（该步累计收满的"
            "非退化组数）与 `target_prompts`（= `train_batch_size`）。这一步直接回答"
            "**「模型每步到底拿到几个有梯度的组」**。",
            "",
            "| 项 | 值 |",
            "|---|---:|",
            f"| 每步目标组数（`target_prompts`） | {target} |",
            f"| 有填充信息的步数 | {fill['steps_with_fill_info']} |",
            f"| **收满目标的步数** | **{fill['steps_reached_target']}** |",
            f"| **未收满目标的步数** | **{fill['steps_below_target']}** |",
            f"| 平均每步收满组数 | {fill['mean_filled_per_step']:.2f} |",
            f"| 最少收满组数 | {fill['min_filled_per_step']} |",
            "",
        ]
        if all_reached:
            lines += [
                f"> ✅ **每一步都收满了 {target} 个非退化组** ⇒ 模型每步拿到的梯度信息"
                "**已经是完整的**。",
                "> ⇒ **过滤训练集不会增加任何梯度信息** —— `dynamic_sampling` 本身就是那个过滤器，"
                "它每步已经只把「有差异的题」喂给优化器。过滤只能省 rollout 成本（占单步耗时约 4%）。",
                "",
            ]
        else:
            lines += [
                f"> ⚠️ 有 **{fill['steps_below_target']}** 步**没**收满 {target} 个非退化组"
                f"（步号：{fill['below_target_steps']}）"
                "⇒ 这些步的梯度信息确实不完整，此时**扩大池子**或**放宽采样上限**才有意义。",
                "",
            ]

    lines += [
        heading("动态采样实测的组级有效率"),
        "",
        "| 项 | 值 |",
        "|---|---:|",
        f"| 生成过的组（prompt）数 | {groups['seen']} |",
        f"| kept（有梯度）组数 | {groups['kept']} |",
        f"| **实测 kept 率** | **{groups['kept_rate']:.1%}** |",
        f"| accepted（真正进 batch）组数 | {groups['accepted']} |",
        "",
        "组被丢弃的原因分布（`kept` 行即上面已计的保留数，列出仅为配平）：",
        "",
        "| 原因 | 组数 |",
        "|---|---:|",
    ]
    for reason, count in sorted(
        groups["drop_reason_counts"].items(), key=lambda item: -item[1]
    ):
        lines.append(f"| `{reason}` | {count} |")

    lines += [
        "",
        heading("每步用掉几个 generation batch"),
        "",
        f"- 合计 generation batch：{batches['total']}，平均 **{batches['mean_per_step']:.2f}** 个/步，最多 {batches['max_per_step']} 个",
        f"- 每个 batch 平均 {batches['mean_groups_per_batch']:.2f} 个组，组内 kept 率 {batches['mean_kept_rate_per_batch']:.1%}",
        "",
        "| 每步 batch 数 | 出现步数 |",
        "|---:|---:|",
    ]
    for count, steps_count in batches["per_step_counts"].items():
        lines.append(f"| {count} | {steps_count} |")

    if max_gen_batches:
        # 补丁里 `dynamic_num_gen_batches` 在跳过时被 reset，且**跳过只在计数达到
        # `max_num_gen_batches` 时才触发** ⇒ 每个重试轮次必然跑满上限。所以这是精确值，不是上界。
        trajectories_per_batch = batches["mean_groups_per_batch"] * 4  # 每组 = rollout n = 4
        wasted_trajectories = wasted_batches * trajectories_per_batch
        lines += [
            "",
            f"- 配置 `max_num_gen_batches = {max_gen_batches}`（每个 batch = `train_batch_size` 条 prompt）",
            f"- **跳过只在达到上限时触发**，故每次重试必然跑满：浪费的 generation batch = "
            f"{steps['skipped_rounds']} × {max_gen_batches} = **{wasted_batches}** 个"
            f"（≈ {wasted_trajectories:.0f} 条轨迹），占全部 {batches['total']} 个 batch 的 "
            f"{wasted_batches / max(batches['total'], 1):.0%}",
        ]

    metrics = summary.get("optimizer_metrics") or {}
    picked = metrics.get("picked") or {}
    if picked:
        lines += [
            "",
            heading("训练动力学（optimizer_step 的指标首/末/极值）"),
            "",
            "| 指标 | 首步 | 首值 | 末步 | 末值 | 最小 | 最大 | 均值 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key in sorted(picked):
            row = picked[key]
            lines.append(
                f"| `{key}` | {row['first_step']} | {row['first']:.4g} | "
                f"{row['last_step']} | {row['last']:.4g} | {row['min']:.4g} | "
                f"{row['max']:.4g} | {row['mean']:.4g} |"
            )
        lines += [
            "",
            "> **怎么读**：`actor/ppo_kl` 与 `actor/pg_clipfrac` 是「策略动了多少」的直接度量。"
            "若两者都贴着 0（KL ≲ 0.01、clipfrac ≲ 1e-3），说明**优化信号太弱**，"
            "而不是数据不够 —— 此时该动的是 `lr` / 步数 / `n`，不是数据集。",
        ]

    if probe:
        lines += [
            "",
            heading("与全池探针的交叉核对"),
            "",
            "| 项 | 探针（离线，n=4, T=0.7） | 训练（动态采样实测） |",
            "|---|---:|---:|",
            f"| reward_varying 率 | {probe['probe_reward_varying_rate']:.1%} | — |",
            f"| eligible（有梯度）率 | **{probe['probe_eligible_rate']:.1%}** | **{probe['training_kept_rate']:.1%}** |",
            f"| 题数 | {probe['probe_tasks']} | — |",
            "",
            "> 两者同量级 ⇒ 探针可信；若训练侧明显更低，以训练侧数字为准。",
        ]

    lines += ["", heading("判读"), ""]

    if steps["steps_skipped_without_any_update"] > 0:
        lines.append(
            f"- ⚠️ 有 **{steps['steps_skipped_without_any_update']}** 步完全没有更新"
            "（罕见，通常只会在连续跳过逼近上限时出现）。"
        )
    else:
        lines.append(
            f"- ✅ **每一步都产出了梯度**（{total_steps}/{total_steps}）。"
            f"{steps['skipped_rounds']} 次重试只浪费 rollout，**没有损失梯度步**。"
        )
    if max_gen_batches:
        lines += [
            f"- 重试浪费 **{wasted_batches / max(batches['total'], 1):.0%}** 的 generation batch。"
            "把 `max_num_gen_batches` 从 3 提到 8 能省掉这部分 rollout 成本，"
            "**但它不会增加梯度步数** ⇒ **治不了「没提升」**。",
        ]
    if picked.get("actor/ppo_kl"):
        kl = picked["actor/ppo_kl"]
        clip = picked.get("actor/pg_clipfrac")
        weak = kl["mean"] < 0.01 and (clip is None or clip["mean"] < 1e-3)
        if weak:
            lines += [
                f"- 🎯 **`actor/ppo_kl` 均值 {kl['mean']:.4g}"
                + (f"、`pg_clipfrac` 均值 {clip['mean']:.4g}" if clip else "")
                + " ⇒ 策略几乎没动。**这才是「没提升」的直接原因**，"
                f"而数据池（实测 kept 率 {groups['kept_rate']:.1%}）是健康的。",
                "- 结论：**不该改数据集，也不该改采样上限**；该动的是学习信号强度"
                "（`lr` ↑ / 步数 ↑ / `n` ↑）。",
            ]
        else:
            lines.append(
                f"- `actor/ppo_kl` 均值 {kl['mean']:.4g}，策略确实在动；"
                "此时再看 Final-200 是否提升，以及是否被验证集噪声掩盖。"
            )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    records = read_records(args.diagnostics)
    summary = analyse(records)
    probe = (
        compare_with_probe(summary, args.probe_labels) if args.probe_labels else None
    )
    if probe:
        summary["probe"] = probe
    if args.max_gen_batches:
        summary["configured_max_gen_batches"] = args.max_gen_batches

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render(summary, probe, args.max_gen_batches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
