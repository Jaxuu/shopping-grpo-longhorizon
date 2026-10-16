#!/usr/bin/env python3
"""探针标签的「信号尺度」分析：把「有差异」细分成「差异有多大」。

背景
----
`dynamic_sampling` 的准入判据是 ``utility_max - utility_min > tolerance``（默认 ``1e-8``）。
这是一个**二值**判据 —— 它把 ``spread = 1e-7`` 与 ``spread = 1.0`` 同等对待。

但在 ``norm_adv_by_std_in_grpo = false``（Dr.GRPO 风格，本仓库与作者的设置）下，
GRPO 的优势是 ``A_i = r_i - mean(r)``，**不除以组内 std** ⇒ 优势幅度正比于 spread。
所以 ``spread ~ 0`` 的"有效组"会占掉每步 4 个名额之一，却贡献 ~0 的梯度。

本脚本回答：探针保留的那批题里，spread 的**分布**是什么形状？

- 若大部分 spread 都很大（如「1 成功 + 3 失败」⇒ spread ≈ 1）
  ⇒ 信号尺度没问题，每步走不动是 ``lr`` 太小 ⇒ **提 lr 是对的**。
- 若相当一部分 spread 很小（⇒ std 很小）
  ⇒ 优势被 spread 压住，提 lr 只是把一个 ~0 的数乘 4
  ⇒ **换杠杆才治本**（打开 std 归一化，或提高 tolerance）。

输入是 ``scripts/probe_group_diversity.py`` 写出的 ``difficulty_labels.jsonl``
（每行含 ``task_id`` / ``pass_count`` / ``pass_rate`` / ``reward_varying`` /
``utility_min`` / ``utility_max`` / ``utility_std`` / ``eligible``）。

用法::

    .venv/bin/python scripts/analyze_probe_signal_scale.py \
        outputs/probe/m6r-full/difficulty_labels.jsonl

    # 机器可读
    .venv/bin/python scripts/analyze_probe_signal_scale.py <labels> --json

    # 落盘报告
    .venv/bin/python scripts/analyze_probe_signal_scale.py <labels> \
        --out outputs/probe/m6r-full/signal_scale.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

# 与 `group_diversity.DEFAULT_TOLERANCE` 和 veRL 补丁的 `reward_tolerance` 一致。
DEFAULT_TOLERANCE = 1.0e-8

# 「近似退化」判据：std 低于此值的"有效组"在 Dr.GRPO 风格下贡献 ~0 优势。
# 取 0.05 是因为终局奖励的最小非零档位间隔是 0.55（gold 1.0 vs valid_alt 0.55）
# 的十分之一量级，足以区分"真的有差异"与"数值抖动"。
DEFAULT_NEAR_DEGENERATE_STD = 0.05

# 训练侧实测的「可用组」接受率（标定锚点）。反解自第一轮正式训练：
# 50 步共 82 次尝试 ⇒ 1.64 次/步 ⇒ 单次成功率 50/82 = 61.0% ⇒ p ≈ 0.338；
# 用 `fill_probability` 交叉验证：预测 80.6 次 vs 实测 82 次，误差 1.7%。
# ⚠️ 探针的 `reward_varying` 率（本仓库实测 42.8%）**不是**这个量 —— 它把
# sampling-invalid 的组也算作"能收"，会高估接受率、低估重试成本。
TRAINING_OBSERVED_ACCEPT_RATE = 0.338

# tolerance 扫描：问"如果把准入阈值从 1e-8 提高到这里，池子还剩多少、还能不能收满"。
TOLERANCE_SWEEP = (1.0e-8, 1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1, 2.0e-1, 3.0e-1)

# 静态 pass_rate 带：只作为「换训练集」的备选口径列出，本仓库默认不做（见 runbook §8.5）。
PASS_BANDS = ((0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (0.0, 0.5), (0.5, 1.0))


def load_labels(path: Path) -> list[dict[str, Any]]:
    """读 ``difficulty_labels.jsonl``；跳过空行。"""
    labels: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno} 不是 JSON 对象")
            labels.append(row)
    if not labels:
        raise ValueError(f"{path} 里没有任何标签行")
    return labels


def spread(row: dict[str, Any]) -> float:
    """组内终局奖励的极差。缺失字段时回退到 0（等价于退化组）。"""
    low = row.get("utility_min")
    high = row.get("utility_max")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return float(high) - float(low)
    std = row.get("utility_std")
    if isinstance(std, (int, float)):
        return 2.0 * float(std)  # 粗略回退：极差 ≈ 2×pstdev（n=4 时偏保守）
    return 0.0


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """线性插值分位数；``q`` 取 ``[0, 1]``。空序列返回 ``nan``。"""
    if not sorted_values:
        return float("nan")
    if q <= 0.0:
        return float(sorted_values[0])
    if q >= 1.0:
        return float(sorted_values[-1])
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(sorted_values[upper]) * weight


def fill_probability(
    accept_rate: float,
    *,
    prompts_per_batch: int = 4,
    max_batches: int = 3,
    target: int = 4,
) -> dict[str, float]:
    """在 ``max_batches`` 个 batch 内收满 ``target`` 个非退化组的概率。

    veRL 补丁每批抽 ``prompts_per_batch`` 个 prompt，每个以 ``accept_rate`` 的概率
    被判为「有差异」并被收下；累计收满 ``target`` 才做一次 optimizer update，
    否则重试（连续 ``max_consecutive_skipped_updates`` 次才中止）。
    因此"能否成功"等价于 ``Binomial(max_batches * prompts_per_batch, p) >= target``。
    """
    if not 0.0 <= accept_rate <= 1.0:
        raise ValueError("accept_rate 必须在 [0, 1]")
    if prompts_per_batch <= 0 or max_batches <= 0 or target <= 0:
        raise ValueError("prompts_per_batch / max_batches / target 必须为正")
    total_draws = prompts_per_batch * max_batches
    if target > total_draws:
        return {"success_probability": 0.0, "expected_batches": float(max_batches)}
    # P(X < target)，X ~ Binomial(total_draws, accept_rate)
    cdf_below = sum(
        math.comb(total_draws, k)
        * accept_rate**k
        * (1.0 - accept_rate) ** (total_draws - k)
        for k in range(min(target, total_draws))
    )
    success = max(0.0, 1.0 - cdf_below)
    # E[batches] = Σ_{b=1..max} P(前 b-1 批之后仍未收满)
    expected = 0.0
    for used_batches in range(1, max_batches + 1):
        draws_before = prompts_per_batch * (used_batches - 1)
        if draws_before < target:
            not_yet = 1.0  # 抽的次数都不够 target 次，必然没收满
        else:
            not_yet = sum(
                math.comb(draws_before, k)
                * accept_rate**k
                * (1.0 - accept_rate) ** (draws_before - k)
                for k in range(target)
            )
        expected += not_yet
        if not_yet == 0.0:
            break
    return {"success_probability": success, "expected_batches": expected}


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def analyze(
    labels: Sequence[dict[str, Any]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    near_degenerate_std: float = DEFAULT_NEAR_DEGENERATE_STD,
    sweep: Sequence[float] = TOLERANCE_SWEEP,
    pass_bands: Sequence[tuple[float, float]] = PASS_BANDS,
) -> dict[str, Any]:
    """核心分析：保留题的 spread / std 分布 + tolerance 扫描 + pass_rate 交叉。"""
    if not labels:
        raise ValueError("labels 为空")

    total = len(labels)
    varying = [row for row in labels if row.get("reward_varying")]
    kept = [row for row in labels if row.get("eligible")]

    spreads = sorted(spread(row) for row in varying)
    stds = sorted(float(row.get("utility_std") or 0.0) for row in varying)

    quantiles = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    spread_quantiles = {f"p{int(q * 100):02d}": quantile(spreads, q) for q in quantiles}
    std_quantiles = {f"p{int(q * 100):02d}": quantile(stds, q) for q in quantiles}

    near_degenerate = [row for row in varying if float(row.get("utility_std") or 0.0) < near_degenerate_std]
    meaningful = [row for row in varying if float(row.get("utility_std") or 0.0) >= near_degenerate_std]

    # ── tolerance 扫描：把准入阈值提高会怎样 ────────────────────────────
    # ⚠️ 分母必须是 `eligible`（无 sampling-invalid **且** spread > tolerance），
    # 不是 `reward_varying`：后者把 sampling-invalid 的组也算作"能收"，会高估接受率、
    # 低估重试成本（2026-10-17 实测：varying 42.8% / eligible 36.6% / 训练侧 33.8%）。
    sweep_rows = []
    for threshold in sweep:
        surviving = [row for row in kept if spread(row) > threshold]
        accept_rate = _rate(len(surviving), total)
        fill = fill_probability(accept_rate)
        sweep_rows.append(
            {
                "tolerance": float(threshold),
                "surviving_tasks": len(surviving),
                "accept_rate": accept_rate,
                "success_probability": fill["success_probability"],
                "expected_batches": fill["expected_batches"],
                "attempts_per_step": (
                    1.0 / fill["success_probability"] if fill["success_probability"] > 0 else float("inf")
                ),
            }
        )

    # ── pass_rate 交叉：按"组内成功条数"看 spread 的均值 ────────────────
    by_pass_count: dict[str, dict[str, float]] = {}
    buckets: dict[int, list[float]] = {}
    for row in labels:
        count = row.get("pass_count")
        if isinstance(count, int):
            buckets.setdefault(count, []).append(spread(row))
    for count in sorted(buckets):
        values = sorted(buckets[count])
        by_pass_count[str(count)] = {
            "tasks": len(values),
            "mean_spread": sum(values) / len(values),
            "median_spread": quantile(values, 0.5),
            "varying_rate": _rate(
                sum(1 for row in labels if row.get("pass_count") == count and row.get("reward_varying")),
                len(values),
            ),
        }

    # ── 静态 pass_rate 带：只作"换训练集"的备选口径列出 ─────────────────
    band_rows = []
    for low, high in pass_bands:
        in_band = [
            row
            for row in labels
            if isinstance(row.get("pass_rate"), (int, float)) and low <= float(row["pass_rate"]) <= high
        ]
        band_rows.append(
            {
                "band": [float(low), float(high)],
                "tasks": len(in_band),
                "varying_rate": _rate(sum(1 for row in in_band if row.get("reward_varying")), len(in_band)),
                "eligible_tasks": sum(1 for row in in_band if row.get("eligible")),
            }
        )

    return {
        "totals": {
            "tasks": total,
            "varying": len(varying),
            "kept": len(kept),
            "varying_rate": _rate(len(varying), total),
            "kept_rate": _rate(len(kept), total),
        },
        "tolerance": float(tolerance),
        "near_degenerate_std": float(near_degenerate_std),
        "spread_quantiles": spread_quantiles,
        "std_quantiles": std_quantiles,
        "spread_mean": (sum(spreads) / len(spreads)) if spreads else float("nan"),
        "std_mean": (sum(stds) / len(stds)) if stds else float("nan"),
        "near_degenerate": {
            "tasks": len(near_degenerate),
            "rate_of_varying": _rate(len(near_degenerate), len(varying)),
            "rate_of_all": _rate(len(near_degenerate), total),
            "meaningful_tasks": len(meaningful),
            "meaningful_rate_of_varying": _rate(len(meaningful), len(varying)),
        },
        "tolerance_sweep": sweep_rows,
        "by_pass_count": by_pass_count,
        "pass_bands": band_rows,
        "current_fill": fill_probability(_rate(len(kept), total)),
        "fill_calibration": {
            "eligible_rate": _rate(len(kept), total),
            "eligible_fill": fill_probability(_rate(len(kept), total)),
            "varying_rate": _rate(len(varying), total),
            "varying_fill": fill_probability(_rate(len(varying), total)),
            "training_observed_rate": TRAINING_OBSERVED_ACCEPT_RATE,
            "training_observed_fill": fill_probability(TRAINING_OBSERVED_ACCEPT_RATE),
        },
    }


def verdict(summary: dict[str, Any]) -> tuple[str, list[str]]:
    """把统计量翻成一句判读 + 若干依据行。"""
    near = summary["near_degenerate"]
    rate = near["rate_of_varying"]
    lines: list[str] = []
    lines.append(
        f"保留的「有差异」题共 {summary['totals']['varying']} 道，"
        f"spread 中位数 {summary['spread_quantiles']['p50']:.4f}、"
        f"均值 {summary['spread_mean']:.4f}。"
    )
    lines.append(
        f"其中 std < {summary['near_degenerate_std']:.2f} 的「近似退化组」"
        f"{near['tasks']} 道，占有效组的 **{rate:.1%}**。"
    )

    if rate >= 0.5:
        head = "换杠杆"
        lines.append(
            "⇒ 半数以上的「有效组」其实几乎没有优势幅度。在 "
            "`norm_adv_by_std_in_grpo=false` 下优势 ∝ spread ⇒ "
            "提 lr 只是把 ~0 的数乘 4，**换杠杆更治本**（打开 std 归一化，或提高 tolerance）。"
        )
    elif rate >= 0.25:
        head = "两者都值得做，但先提 lr"
        lines.append(
            "⇒ 有相当一部分有效组优势幅度很小。提 lr 能整体放大信号，"
            "但**打开 std 归一化**（`norm_adv_by_std_in_grpo=true`）能只放大被压住的那部分，"
            "两者不冲突；按 P8 先单变量提 lr。"
        )
    else:
        head = "提 lr 是对的"
        lines.append(
            "⇒ 绝大多数有效组的 spread 都很大，信号尺度不是瓶颈 ⇒ "
            "**提 lr（×4）是对症的**，不需要动优势归一化。"
        )

    risky = [
        row
        for row in summary["tolerance_sweep"]
        if row["tolerance"] > summary["tolerance"] and row["success_probability"] < 0.99
    ]
    if risky:
        first = risky[0]
        lines.append(
            f"⚠️ 提高 tolerance 的代价：阈值 {first['tolerance']:g} 时池子只剩 "
            f"{first['surviving_tasks']} 道（接受率 {first['accept_rate']:.1%}），"
            f"3 批内收满 4 组的概率降到 **{first['success_probability']:.1%}**"
            f"（期望用 {first['expected_batches']:.2f} 批）⇒ 会明显增加重试轮次。"
        )
    return head, lines


def render(summary: dict[str, Any]) -> str:
    """渲染 markdown 报告。"""
    totals = summary["totals"]
    out: list[str] = []
    out.append("# 探针信号尺度分析")
    out.append("")
    out.append(
        f"> 输入：`difficulty_labels.jsonl`。判据：`spread = utility_max - utility_min`；"
        f"准入 tolerance = `{summary['tolerance']:g}`。"
    )
    out.append("")
    out.append("## 1. 总量")
    out.append("")
    out.append("| 项 | 值 |")
    out.append("|---|---|")
    out.append(f"| 题目总数 | {totals['tasks']} |")
    out.append(f"| `reward_varying`（有差异） | {totals['varying']}（{totals['varying_rate']:.1%}） |")
    out.append(f"| `eligible`（保留） | {totals['kept']}（{totals['kept_rate']:.1%}） |")
    out.append("")

    out.append("## 2. 有效组的 spread / std 分布（关键）")
    out.append("")
    out.append("| 分位 | spread | std |")
    out.append("|---|---|---|")
    for key in ("p00", "p10", "p25", "p50", "p75", "p90", "p100"):
        out.append(
            f"| {key[1:]}% | {summary['spread_quantiles'][key]:.4f} | "
            f"{summary['std_quantiles'][key]:.4f} |"
        )
    out.append(f"| **均值** | **{summary['spread_mean']:.4f}** | **{summary['std_mean']:.4f}** |")
    out.append("")
    near = summary["near_degenerate"]
    out.append(
        f"**近似退化组**（std < {summary['near_degenerate_std']:.2f}）："
        f"{near['tasks']} 道，占有效组 **{near['rate_of_varying']:.1%}**、占全池 {near['rate_of_all']:.1%}。"
    )
    out.append("")

    out.append("## 3. 判读")
    out.append("")
    head, lines = verdict(summary)
    out.append(f"**结论：{head}**")
    out.append("")
    for line in lines:
        out.append(f"- {line}")
    out.append("")

    out.append("## 4. tolerance 扫描（提高准入阈值的代价）")
    out.append("")
    out.append(
        "> 口径：**存活题数** = `eligible`（无 sampling-invalid 且 spread > tolerance）中 "
        "spread 仍大于该阈值的题数；**接受率** = 存活题数 / 全池。"
        "⚠️ 分母**不能**用 `reward_varying` —— 它含 sampling-invalid 的组，会高估接受率。"
    )
    out.append("")
    out.append("| tolerance | 存活题数 | 接受率 | 3 批内收满 4 组的概率 | 期望用批数 | 每步期望尝试数 |")
    out.append("|---|---|---|---|---|---|")
    for row in summary["tolerance_sweep"]:
        mark = " ←当前" if row["tolerance"] == summary["tolerance"] else ""
        out.append(
            f"| {row['tolerance']:g}{mark} | {row['surviving_tasks']} | "
            f"{row['accept_rate']:.1%} | {row['success_probability']:.1%} | "
            f"{row['expected_batches']:.2f} | {row['attempts_per_step']:.2f} |"
        )
    out.append("")
    current = summary["current_fill"]
    out.append(
        f"当前口径（tolerance={summary['tolerance']:g}）下，3 批内收满 4 组的概率 = "
        f"**{current['success_probability']:.1%}**，期望用 **{current['expected_batches']:.2f}** 批、"
        f"每步期望尝试 **{1.0 / current['success_probability']:.2f}** 次。"
    )
    out.append("")
    calibration = summary.get("fill_calibration")
    if calibration:
        out.append(
            f"> **标定（2026-10-17）**：探针 `eligible` 率 {calibration['eligible_rate']:.1%} ⇒ "
            f"每步期望尝试 {1.0 / calibration['eligible_fill']['success_probability']:.2f} 次；"
            f"**训练侧实测** {calibration['training_observed_rate']:.1%} ⇒ "
            f"{1.0 / calibration['training_observed_fill']['success_probability']:.2f} 次"
            "（与第一轮 82 次尝试 / 50 步 = 1.64 次/步 一致）；"
            f"若误用 `reward_varying` 率 {calibration['varying_rate']:.1%} 会得到 "
            f"{1.0 / calibration['varying_fill']['success_probability']:.2f} 次，**那是高估**。"
        )
        out.append("")
    out.append(
        "> 读法：提高 tolerance 会**缩小**接受率 ⇒ 收满 4 组的概率下降、每步尝试数上升。"
        "例如接受率 33.8% → 25% 时，每步期望尝试数 1.61 → 2.85，"
        "**rollout 成本近乎翻倍**。所以「提高 tolerance 以剔除近似退化组」不是免费的；"
        "如果目标只是放大被压住的优势，**打开 `norm_adv_by_std_in_grpo`** 不缩池子、更便宜。"
    )
    out.append("")

    out.append("## 5. 按组内成功条数看 spread")
    out.append("")
    out.append("| pass_count | 题数 | spread 均值 | spread 中位 | varying 率 |")
    out.append("|---|---|---|---|---|")
    for key, row in summary["by_pass_count"].items():
        out.append(
            f"| {key} | {row['tasks']} | {row['mean_spread']:.4f} | "
            f"{row['median_spread']:.4f} | {row['varying_rate']:.1%} |"
        )
    out.append("")
    out.append(
        "> 读法：`pass_count = 4`（全 gold）的组奖励恒为 1.0 ⇒ **必然**是零方差组。"
        "`pass_count = 0` 的组**通常**也是零方差，但若四条轨迹的失败类型不同"
        "（`repeat_loop` −0.65 / `max_steps` −0.50 / `early_abstain` −0.35）或 partial 分不同，"
        "仍可能是 varying 组 —— 这正是「全失败 28.7%」里有一小部分其实有梯度的原因。"
        "spread 最大的通常是 1 成功 + 3 失败的组。"
    )
    out.append("")

    out.append("## 6. 静态 pass_rate 带（只作「换训练集」的备选口径）")
    out.append("")
    out.append("| 带 | 题数 | varying 率 | eligible 题数 |")
    out.append("|---|---|---|---|")
    for row in summary["pass_bands"]:
        low, high = row["band"]
        out.append(
            f"| [{low:g}, {high:g}] | {row['tasks']} | {row['varying_rate']:.1%} | "
            f"{row['eligible_tasks']} |"
        )
    out.append("")
    out.append(
        "> ⚠️ 本节**不是**建议改训练集。`dynamic_sampling` 每步已经只把有差异的组喂给优化器，"
        "静态过滤只省 rollout 成本（约 7–11%）、不改梯度分布。列在这里只是为了让"
        "「换池子」这个选项有数据可比。"
    )
    out.append("")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path, help="difficulty_labels.jsonl 路径")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"动态采样的准入 tolerance（默认 {DEFAULT_TOLERANCE:g}）",
    )
    parser.add_argument(
        "--near-degenerate-std",
        type=float,
        default=DEFAULT_NEAR_DEGENERATE_STD,
        help=f"「近似退化」的 std 判据（默认 {DEFAULT_NEAR_DEGENERATE_STD}）",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非 markdown")
    parser.add_argument("--out", type=Path, default=None, help="把 markdown 报告写到该路径")
    args = parser.parse_args(argv)

    try:
        labels = load_labels(args.labels)
    except (OSError, ValueError) as exc:
        print(f"读取失败：{exc}", file=sys.stderr)
        return 2

    summary = analyze(
        labels,
        tolerance=args.tolerance,
        near_degenerate_std=args.near_degenerate_std,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2) if args.json else render(summary)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render(summary), encoding="utf-8")
        print(f"\n报告已写入 {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
