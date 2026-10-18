#!/usr/bin/env python3
"""`ratio` 噪声审计：`pg_clipfrac` 到底在测「策略动了」还是「引擎噪声」？

背景
----
本仓库开了 `algorithm.rollout_correction.bypass_mode: true`（作者基线），
PPO 的 `ratio` 因此是

    ratio = exp(FSDP_new_logp − vLLM_rollout_logp)

两个推理引擎的数值差异会直接进入 `ratio`。于是 `actor/ppo_kl ≈ 0.003` 这个数
**分不清**是「策略真动了」还是「跨引擎底噪」，而 `actor/pg_clipfrac` 只在
`|log ratio| > log(1+clip_high)` 时计数，天然只统计尾部。

本脚本做三件事
--------------
1. **无假设口径**：`clipfrac` 本身就是「被真正更新超过阈值的 token 占比」。
   乘上每步 token 数 ⇒ 每步有多少 token 获得了「有意义的更新」。这个数不需要任何模型。
2. **两种口径的噪声检验**：把 `ppo_kl` 分别解释为
   - k3（``0.5·E[d²]``，veRL 常见的 `((r-1) - log r)` 口径）⇒ ``σ = sqrt(2·kl)``；
   - k1（``E|d|``）⇒ ``σ = kl·sqrt(π/2)``。
   在 `d ~ N(0, σ²)`（纯噪声）下预测 ``clipfrac = 2Φ(−c/σ)``，与实测比。
   两个预测都不贴 ⇒ 分布是**重尾**的：bulk 是噪声、尾部是真信号。
3. **跨轮对比**：同一配方不同 lr 的两轮放在一起，看 `ppo_kl`（不动）与
   `clipfrac`（动了）是否互相矛盾 —— 这是判断 `bypass_mode` 该不该关的直接证据。

用法::

    .venv/bin/python scripts/audit_ratio_noise.py \
        outputs/models/grpo-m6r/training_diagnostics.jsonl \
        outputs/models/grpo-m6/training_diagnostics.jsonl

    # 只看汇总
    .venv/bin/python scripts/audit_ratio_noise.py <jsonl> --no-per-step
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

# 与配方一致的 clip 上界（CLI override 0.28）。
DEFAULT_CLIP_HIGH = 0.28

# 一致性判据：预测/实测 落在 [1/2, 2] 内视为「该口径说得通」。
CONSISTENCY_BAND = (0.5, 2.0)


def norm_cdf(value: float) -> float:
    """标准正态 CDF（只用 stdlib）。"""
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def load_steps(path: Path) -> list[dict[str, float]]:
    """抽出每个 `optimizer_step` 的 kl / clipfrac / token 数。"""
    steps: list[dict[str, float]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
            if not isinstance(record, dict) or record.get("event") != "optimizer_step":
                continue
            metrics = record.get("metrics")
            if not isinstance(metrics, dict):
                continue
            step = record.get("global_step")
            kl = metrics.get("actor/ppo_kl")
            clipfrac = metrics.get("actor/pg_clipfrac")
            if not isinstance(kl, (int, float)) or not isinstance(clipfrac, (int, float)):
                continue
            tokens = metrics.get("perf/total_num_tokens") or metrics.get("global_seqlen/mean")
            steps.append(
                {
                    "step": float(step) if isinstance(step, int) else float(len(steps) + 1),
                    "kl": float(kl),
                    "clipfrac": float(clipfrac),
                    "tokens": float(tokens) if isinstance(tokens, (int, float)) else 0.0,
                }
            )
    if not steps:
        raise ValueError(f"{path} 里没有带 ppo_kl / pg_clipfrac 的 optimizer_step 记录")
    return steps


def predicted_clipfrac(sigma: float, clip_high: float) -> float:
    """``d ~ N(0, σ²)`` 时 ``P(|d| > log(1+clip_high))``。"""
    if sigma <= 0.0:
        return 0.0
    threshold = math.log1p(clip_high)
    return 2.0 * norm_cdf(-threshold / sigma)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarize(steps: Sequence[dict[str, float]], *, clip_high: float = DEFAULT_CLIP_HIGH) -> dict[str, Any]:
    """把一串 step 压成审计指标。"""
    kl = _mean([step["kl"] for step in steps])
    clipfrac = _mean([step["clipfrac"] for step in steps])
    tokens = _mean([step["tokens"] for step in steps])

    sigma_k3 = math.sqrt(max(0.0, 2.0 * kl))
    sigma_k1 = kl * math.sqrt(math.pi / 2.0)
    pred_k3 = predicted_clipfrac(sigma_k3, clip_high)
    pred_k1 = predicted_clipfrac(sigma_k1, clip_high)

    def ratio(predicted: float) -> float:
        return (predicted / clipfrac) if clipfrac > 0 else float("inf")

    ratio_k3 = ratio(pred_k3)
    ratio_k1 = ratio(pred_k1)
    low, high = CONSISTENCY_BAND
    if low <= ratio_k3 <= high and low <= ratio_k1 <= high:
        verdict = "两口径都能解释（噪声与信号无法区分）"
    elif low <= ratio_k3 <= high:
        verdict = "k3 口径自洽 ⇒ ppo_kl 主要是引擎噪声，clipfrac 是它的尾部"
    elif low <= ratio_k1 <= high:
        verdict = "k1 口径自洽 ⇒ 尾部是真信号，ppo_kl 的 bulk 是噪声"
    elif ratio_k3 > high and ratio_k1 < low:
        verdict = "两口径都不自洽 ⇒ 重尾分布：bulk 是噪声、尾部是真实更新"
    elif ratio_k3 > high:
        verdict = "k3 高估 4 倍以上 ⇒ ppo_kl 不是纯方差口径"
    elif ratio_k1 < low:
        verdict = "k1 严重低估 ⇒ 尾部不可能是纯噪声"
    else:
        verdict = "口径不明确"

    return {
        "steps": len(steps),
        "kl_mean": kl,
        "clipfrac_mean": clipfrac,
        "tokens_per_step": tokens,
        "threshold_log_ratio": math.log1p(clip_high),
        "sigma_k3": sigma_k3,
        "sigma_k1": sigma_k1,
        "predicted_clipfrac_k3": pred_k3,
        "predicted_clipfrac_k1": pred_k1,
        "ratio_k3": ratio_k3,
        "ratio_k1": ratio_k1,
        "tail_tokens_per_step": clipfrac * tokens,
        "verdict": verdict,
    }


def _fmt(value: float, digits: int = 4) -> str:
    """小数用 4 位有效数字，大数取整（token 数不需要小数）。"""
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.{digits}g}"


def _fmt_ratio(value: float) -> str:
    """比值：接近 0 显示 ≈0，其余保留 2 位有效数字。"""
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "∞"
    if value < 0.01:
        return "≈0"
    return f"{value:.2g}"


def render(
    summaries: Sequence[tuple[str, dict[str, Any]]],
    steps_by_run: dict[str, list[dict[str, float]]],
    *,
    clip_high: float,
    per_step: bool,
) -> str:
    """渲染 markdown 报告。"""
    out: list[str] = []
    out.append("# ratio 噪声审计（bypass_mode 下 ppo_kl 的可解释性）")
    out.append("")
    out.append(
        f"> `clip_high = {clip_high:g}` ⇒ 触发裁剪的门槛是 `|log ratio| > "
        f"{math.log1p(clip_high):.4f}`。"
    )
    out.append("")

    out.append("## 1. 汇总")
    out.append("")
    out.append(
        "| 运行 | 步数 | ppo_kl 均值 | clipfrac 均值 | 每步 token | 尾部 token/步 | "
        "k3 预测 | k3 比值 | k1 预测 | k1 比值 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, summary in summaries:
        out.append(
            f"| `{name}` | {summary['steps']} | {_fmt(summary['kl_mean'])} | "
            f"{_fmt(summary['clipfrac_mean'])} | {summary['tokens_per_step']:.0f} | "
            f"**{summary['tail_tokens_per_step']:.0f}** | "
            f"{_fmt(summary['predicted_clipfrac_k3'])} | {_fmt_ratio(summary['ratio_k3'])} | "
            f"{_fmt(summary['predicted_clipfrac_k1'])} | {_fmt_ratio(summary['ratio_k1'])} |"
        )
    out.append("")
    out.append(
        "> **比值** = 预测 / 实测。落在 0.5–2 之间说明该口径能解释实测的尾部质量。"
    )
    out.append("")

    out.append("## 2. 判读")
    out.append("")
    for name, summary in summaries:
        out.append(f"**`{name}`**：{summary['verdict']}")
        out.append("")
        out.append(
            f"- 每步约 **{summary['tail_tokens_per_step']:.0f}** 个 token 被更新超过 "
            f"{math.log1p(clip_high) * 100:.0f}% 的 log-ratio 门槛"
            f"（占每步 {summary['tokens_per_step']:.0f} token 的 "
            f"{summary['clipfrac_mean']:.3%}）。"
        )
        out.append(
            f"- 若 ppo_kl 是 k3 口径 ⇒ 隐含 RMS |d| = {summary['sigma_k3']:.4f}，"
            f"纯噪声模型会预测 clipfrac = {summary['predicted_clipfrac_k3']:.2%}"
            f"（实测 {summary['clipfrac_mean']:.2%}）。"
        )
        out.append(
            f"- 若 ppo_kl 是 k1 口径 ⇒ 隐含 σ = {summary['sigma_k1']:.4f}，"
            f"纯噪声预测 clipfrac ≈ {summary['predicted_clipfrac_k1']:.2%}。"
        )
        out.append("")

    if len(summaries) >= 2:
        out.append("### 跨轮对比（同配方、不同 lr）")
        out.append("")
        first_name, first = summaries[0]
        out.append("| 对比项 | " + " | ".join(name for name, _ in summaries) + " |")
        out.append("|---" * (len(summaries) + 1) + "|")
        for key, label in (
            ("kl_mean", "ppo_kl"),
            ("clipfrac_mean", "clipfrac"),
            ("tail_tokens_per_step", "尾部 token/步"),
        ):
            values = " | ".join(_fmt(summary[key]) for _, summary in summaries)
            out.append(f"| {label} | {values} |")
        out.append("")
        if all(math.isfinite(summary["kl_mean"]) for _, summary in summaries):
            kl_spread = max(summary["kl_mean"] for _, summary in summaries) - min(
                summary["kl_mean"] for _, summary in summaries
            )
            clip_spread = max(summary["clipfrac_mean"] for _, summary in summaries) - min(
                summary["clipfrac_mean"] for _, summary in summaries
            )
            out.append(
                f"> `ppo_kl` 的跨轮极差 {kl_spread:.4f}，`clipfrac` 的跨轮极差 {clip_spread:.4f}"
                f"（相对变化 {clip_spread / max(first['clipfrac_mean'], 1e-12):.0%}）。"
                "**同一批数据、同一策略起点，只是 lr 变了**：若 `ppo_kl` 几乎不动而 "
                "`clipfrac` 明显变化，说明两者测的不是同一个量 —— 这正是"
                "`bypass_mode=true` 下 `ratio` 被跨引擎差异污染的直接证据。"
            )
            out.append("")

    if per_step:
        out.append("## 3. 逐步明细")
        out.append("")
        for name, steps in steps_by_run.items():
            out.append(f"### `{name}`")
            out.append("")
            out.append("| step | ppo_kl | clipfrac | 尾部 token |")
            out.append("|---|---|---|---|")
            for step in steps:
                out.append(
                    f"| {int(step['step'])} | {_fmt(step['kl'])} | {_fmt(step['clipfrac'])} | "
                    f"{step['clipfrac'] * step['tokens']:.0f} |"
                )
            out.append("")
    return "\n".join(out)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("diagnostics", nargs="+", type=Path, help="一个或多个 training_diagnostics.jsonl")
    parser.add_argument(
        "--clip-high",
        type=float,
        default=DEFAULT_CLIP_HIGH,
        help=f"PPO 的 clip 上界（默认 {DEFAULT_CLIP_HIGH}）",
    )
    parser.add_argument("--no-per-step", action="store_true", help="不输出逐步明细")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非 markdown")
    parser.add_argument("--out", type=Path, default=None, help="把 markdown 报告写到该路径")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summaries: list[tuple[str, dict[str, Any]]] = []
    steps_by_run: dict[str, list[dict[str, float]]] = {}
    for path in args.diagnostics:
        try:
            steps = load_steps(path)
        except (OSError, ValueError) as exc:
            print(f"读取失败：{exc}", file=sys.stderr)
            return 2
        name = path.parent.name if path.name == "training_diagnostics.jsonl" else path.stem
        steps_by_run[name] = steps
        summaries.append((name, summarize(steps, clip_high=args.clip_high)))

    if args.json:
        print(
            json.dumps(
                {name: summary for name, summary in summaries},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(render(summaries, steps_by_run, clip_high=args.clip_high, per_step=not args.no_per_step))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            render(summaries, steps_by_run, clip_high=args.clip_high, per_step=not args.no_per_step),
            encoding="utf-8",
        )
        print(f"\n报告已写入 {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
