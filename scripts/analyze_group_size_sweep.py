#!/usr/bin/env python3
"""组大小 `n` 的信号吞吐估算：把 `n` 从 4 放大到 8/16 能换回多少「有信号组」。

背景
----
GRPO 的梯度只在**组内奖励不全相同**时存在（`dynamic_sampling` 的准入判据）。
探针（n=4）实测：全池 906 题只有 36.6% 的组有差异 —— 其余是全成功或全失败，
在优化器眼里贡献 0。放大 `n` 能把「4 条恰好全同」的边界题转成有信号组，
但每条轨迹都是真金白银的 rollout，所以要先算清楚：**多花的 rollout 换回多少可用组**。

两种估计（同时输出，避免单一假设）
----------------------------------
1. ``bootstrap``（**保守下界**）：对每道题从它已观测的 4 个奖励里有放回重采 `n` 个，
   统计「不全相同」的比例。4 条全同的题在 bootstrap 下永远全同 ⇒ 得到的是下界。
2. ``beta_binomial``（**乐观上界**）：把每题看成成功概率 `p` 的伯努利过程，
   `p` 取 Jeffreys 后验 ``Beta(k+0.5, 4-k+0.5)``（`k` = 成功条数），
   用 ``E[p^n] = (a)_n / (a+b)_n`` 精确算 ``P(不全相同) = 1 - E[p^n] - E[(1-p)^n]``。
   它假设「全同」的题其真实 `p` 落在 (0,1) 内部 —— 对**结构性卡死**的题
   （例如永远 `repeat_loop`、真实 p≈0）会高估。

代价口径
--------
- ``rollouts_per_step``：每步期望 rollout 条数（含 dynamic_sampling 重试）；
- ``groups_per_1000_rollouts``：每 1000 条 rollout 换来多少个**可训练组**。

⚠️ 每步固定只训练 `train_batch_size` 个组（本仓库 = 4），放大 `n` **不会**让单步梯度
含更多组。它买到的是三件事：① 可用题池变大（降低「无信号熔断」风险）；
② 重试变少；③ 组内 baseline 估计更准（优势方差下降）。判断值不值，看这两列。

用法::

    # 主用法：直接读探针轨迹（有每题 4 条的真实奖励）
    .venv/bin/python scripts/analyze_group_size_sweep.py \
        outputs/probe/m6r-full/trajectories.jsonl

    # 落盘报告
    .venv/bin/python scripts/analyze_group_size_sweep.py \
        outputs/probe/m6r-full/trajectories.jsonl \
        --out outputs/probe/m6r-full/group_size_sweep.md

    # 机器可读
    .venv/bin/python scripts/analyze_group_size_sweep.py <trajectories> --json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

try:  # 以模块方式导入（pytest / python -m）
    from scripts.analyze_probe_signal_scale import fill_probability
except ModuleNotFoundError:  # 直接 `python scripts/xxx.py` 时 sys.path[0] 是 scripts/
    from analyze_probe_signal_scale import fill_probability  # type: ignore[no-redef]

# 组大小扫描范围：4 = 当前配方，8/16 = 候选。
DEFAULT_GROUP_SIZES = (4, 8, 16)

# 「成功」判据：gold = 1.0、valid_alt = 0.55、partial_alt 封顶 0.25 ⇒ 0.5 能干净分开。
DEFAULT_SUCCESS_THRESHOLD = 0.5

# 与训练配方一致：每批 4 个 prompt、最多 3 批、收满 4 组才更新。
DEFAULT_PROMPTS_PER_BATCH = 4
DEFAULT_MAX_BATCHES = 3
DEFAULT_TARGET = 4

# 探针每题的采样条数（n=4 的观测上限）。
DEFAULT_OBSERVED_N = 4

# bootstrap 的蒙特卡洛次数（固定种子 ⇒ 可复现）。
DEFAULT_TRIALS = 200
DEFAULT_SEED = 20261018


def load_groups(
    path: Path,
    *,
    observed_n: int = DEFAULT_OBSERVED_N,
    keep_invalid: bool = False,
) -> dict[int, list[float]]:
    """读探针 `trajectories.jsonl`，返回 ``{task_id: [final_reward, ...]}``。

    ``keep_invalid=False``（默认）时剔除 sampling-invalid 轨迹
    （`error` / `release_error` / `reward_valid is False` / `reward_unverifiable`），
    口径与 `difficulty_labels.jsonl` 的 ``eligible`` 一致。
    """
    if observed_n < 2:
        raise ValueError("observed_n 必须至少为 2")
    grouped: dict[int, list[tuple[int, float]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
            if not isinstance(row, dict) or "task_id" not in row:
                continue
            if not keep_invalid and _is_sampling_invalid(row):
                continue
            reward = row.get("final_reward")
            if not isinstance(reward, (int, float)) or isinstance(reward, bool):
                continue
            value = float(reward)
            if not math.isfinite(value):
                continue
            attempt = row.get("attempt_index", 0)
            attempt_index = int(attempt) if isinstance(attempt, int) else 0
            grouped.setdefault(int(row["task_id"]), []).append((attempt_index, value))

    groups: dict[int, list[float]] = {}
    for task_id, items in grouped.items():
        items.sort(key=lambda item: item[0])
        rewards = [value for _, value in items[:observed_n]]
        if len(rewards) >= 2:
            groups[task_id] = rewards
    if not groups:
        raise ValueError(f"{path} 里没有可用的轨迹（每道题至少需要 2 条）")
    return groups


def _is_sampling_invalid(row: dict[str, Any]) -> bool:
    """与 `group_diversity._sampling_invalid_reasons` 同口径的轻量判定。"""
    if row.get("error") or row.get("release_error"):
        return True
    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail")
    if isinstance(detail, dict) and detail.get("reward_valid") is False:
        return True
    reward_type = ""
    if isinstance(detail, dict):
        reward_type = str(detail.get("reward_type") or detail.get("termination_reason") or "")
    if reward_type in {"reward_unverifiable", "infrastructure_invalid"}:
        return True
    return False


def observed_mixed_rate(groups: dict[int, list[float]], *, tolerance: float = 1.0e-8) -> float:
    """n=4 实测的「有差异组」比例（与探针口径一致）。"""
    if not groups:
        return float("nan")
    mixed = sum(1 for rewards in groups.values() if max(rewards) - min(rewards) > tolerance)
    return mixed / len(groups)


def bootstrap_mixed_rate(
    groups: dict[int, list[float]],
    n: int,
    *,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    tolerance: float = 1.0e-8,
) -> float:
    """有放回重采 `n` 条，统计「不全相同」的比例（保守下界）。"""
    if n < 2:
        raise ValueError("n 必须至少为 2")
    if trials <= 0:
        raise ValueError("trials 必须为正")
    rng = random.Random(seed)
    hits = 0
    total = 0
    for rewards in groups.values():
        # 每道题做 `trials` 次独立重采，全部计入分母（不能「一旦命中就 break」，
        # 那会把分母变成「题数」而不是「题数 × 次数」，等于统计每题的命中概率）。
        for _ in range(trials):
            sample = [rewards[rng.randrange(len(rewards))] for _ in range(n)]
            total += 1
            if max(sample) - min(sample) > tolerance:
                hits += 1
    return hits / total if total else float("nan")


def _rising_ratio(a: float, total: float, n: int) -> float:
    """``(a)_n / (a+b)_n``（Pólya 罐的 n 次幂期望 ``E[p^n]``，Beta(a,b) 下）。"""
    value = 1.0
    for i in range(n):
        value *= (a + i) / (total + i)
    return value


def beta_binomial_mixed_rate(
    groups: dict[int, list[float]],
    n: int,
    *,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    prior: float = 0.5,
) -> float:
    """Jeffreys 后验下的 ``P(n 条不全相同)``（乐观上界，只看成功/失败二分）。"""
    if n < 2:
        raise ValueError("n 必须至少为 2")
    total = 0.0
    for rewards in groups.values():
        successes = sum(1 for value in rewards if value >= success_threshold)
        a = successes + prior
        b = len(rewards) - successes + prior
        all_success = _rising_ratio(a, a + b, n)
        all_fail = _rising_ratio(b, a + b, n)
        total += 1.0 - all_success - all_fail
    return total / len(groups)


def _fill(accept_rate: float, *, prompts_per_batch: int, max_batches: int, target: int) -> dict[str, float]:
    return fill_probability(
        accept_rate,
        prompts_per_batch=prompts_per_batch,
        max_batches=max_batches,
        target=target,
    )


def analyze(
    groups: dict[int, list[float]],
    *,
    group_sizes: Sequence[int] = DEFAULT_GROUP_SIZES,
    prompts_per_batch: int = DEFAULT_PROMPTS_PER_BATCH,
    max_batches: int = DEFAULT_MAX_BATCHES,
    target: int = DEFAULT_TARGET,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    observed_n: int = DEFAULT_OBSERVED_N,
) -> dict[str, Any]:
    """核心分析：各 `n` 下的可用组比例 + 每步 rollout 成本 + 吞吐。"""
    if not groups:
        raise ValueError("groups 为空")
    base_rate = observed_mixed_rate(groups)
    rows = []
    for n in group_sizes:
        if n == observed_n:
            # n 等于探针的实际观测条数 ⇒ 直接用实测比例（bootstrap 在 n=4 会低估：
            # 有差异的组重采 4 条仍可能全同）。这是唯一无需假设的点。
            lower = upper = base_rate
        else:
            lower = bootstrap_mixed_rate(groups, n, trials=trials, seed=seed)
            upper = beta_binomial_mixed_rate(groups, n, success_threshold=success_threshold)
        accept = (lower + upper) / 2.0
        fill = _fill(accept, prompts_per_batch=prompts_per_batch, max_batches=max_batches, target=target)
        rollouts_per_step = fill["expected_batches"] * prompts_per_batch * n
        rows.append(
            {
                "n": int(n),
                "mixed_rate_bootstrap": lower,
                "mixed_rate_beta_binomial": upper,
                "mixed_rate_mid": accept,
                "fill_probability": fill["success_probability"],
                "expected_batches": fill["expected_batches"],
                "rollouts_per_step": rollouts_per_step,
                "groups_per_1000_rollouts": (target / rollouts_per_step * 1000.0)
                if rollouts_per_step > 0
                else 0.0,
                "usable_pool": accept * len(groups),
                "gen_cost_multiplier": n / DEFAULT_OBSERVED_N,
            }
        )
    return {
        "tasks": len(groups),
        "observed_mixed_rate": base_rate,
        "observed_n": observed_n,
        "success_threshold": success_threshold,
        "prompts_per_batch": prompts_per_batch,
        "max_batches": max_batches,
        "target_groups_per_step": target,
        "trials": trials,
        "rows": rows,
    }


def render(summary: dict[str, Any]) -> str:
    """渲染 markdown 报告。"""
    rows = summary["rows"]
    base = rows[0] if rows else None
    out: list[str] = []
    out.append("# 组大小 `n` 的信号吞吐估算")
    out.append("")
    out.append(
        f"> 输入：探针轨迹，{summary['tasks']} 道题（每题观测 4 条）。"
        f"「有信号」= 组内终局奖励不全相同（tolerance 1e-8，与动态采样同口径）。"
    )
    out.append("")
    out.append("## 1. 可用组比例")
    out.append("")
    out.append("| n | bootstrap（保守下界） | beta-binomial（乐观上界） | 取值 | 可用题池 | 相对 gen 成本 |")
    out.append("|---|---|---|---|---|---|")
    for row in rows:
        mark = " ←当前" if row["n"] == DEFAULT_OBSERVED_N else ""
        out.append(
            f"| {row['n']}{mark} | {row['mixed_rate_bootstrap']:.1%} | "
            f"{row['mixed_rate_beta_binomial']:.1%} | {row['mixed_rate_mid']:.1%} | "
            f"{row['usable_pool']:.0f}/{summary['tasks']} | ×{row['gen_cost_multiplier']:.1f} |"
        )
    out.append("")
    if base is not None:
        out.append(
            f"实测锚点：n={base['n']} 的有差异组比例 = **{summary['observed_mixed_rate']:.1%}**"
            f"（与探针 `reward_varying` 同口径）。"
        )
        out.append("")
    out.append(
        "> 两个估计的差就是「不确定性预算」：bootstrap 认为 4 条全同的题**永远**全同，"
        "beta-binomial 认为它们的真实成功率在 (0,1) 内部。真实值落在这两者之间；"
        "**结构性卡死**的题（永远 repeat_loop / 永远买错）只会贡献下界，"
        "要救它们必须靠过程奖励，不是靠加 `n`。"
    )
    out.append("")

    out.append("## 2. 每步成本与吞吐（关键取舍）")
    out.append("")
    out.append(
        "| n | 3 批内收满 4 组的概率 | 期望用批数 | 每步 rollout 条数 | 每 1000 条 rollout 换到的可训练组 |")
    out.append("|---|---|---|---|---|")
    for row in rows:
        out.append(
            f"| {row['n']} | {row['fill_probability']:.1%} | {row['expected_batches']:.2f} | "
            f"{row['rollouts_per_step']:.1f} | **{row['groups_per_1000_rollouts']:.1f}** |"
        )
    out.append("")
    out.append(
        f"> 口径：每批 {summary['prompts_per_batch']} 个 prompt、最多 {summary['max_batches']} 批、"
        f"收满 {summary['target_groups_per_step']} 组才更新（与训练配方一致）。"
    )
    out.append("")

    out.append("## 3. 判读")
    out.append("")
    if base is not None:
        best = max(rows, key=lambda row: row["groups_per_1000_rollouts"])
        out.append(
            f"- **每步训练组数不变**：无论 `n` 多大，每步仍只训练 {summary['target_groups_per_step']} 个组。"
            "放大 `n` 不会让单步梯度含更多独立组。"
        )
        out.append(
            f"- **纯吞吐口径下最优是 n={best['n']}**"
            f"（{best['groups_per_1000_rollouts']:.1f} 组 / 1000 rollout）。"
            "若 n=4 最优，说明「加 n」在 rollout 效率上是**亏的**，它的价值只能来自"
            "「可用题池变大」与「优势估计更准」这两点。"
        )
        out.append(
            f"- 可用题池从 {rows[0]['usable_pool']:.0f} 道涨到 "
            f"{rows[-1]['usable_pool']:.0f} 道（n={rows[-1]['n']}，上界口径）——"
            "这才是加 `n` 的真实收益：**降低无信号熔断风险**，而不是让每步学得更多。"
        )
        out.append(
            "- 因此若决定加 `n`，应同时考虑把 `train_batch_size` 一起抬（换更多组/步），"
            "否则只是把同样的 4 个组算得更准。"
        )
    out.append("")
    return "\n".join(out)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trajectories", type=Path, help="探针 trajectories.jsonl")
    parser.add_argument(
        "--group-sizes",
        default=",".join(str(n) for n in DEFAULT_GROUP_SIZES),
        help="逗号分隔的 n 列表（默认 4,8,16）",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=DEFAULT_SUCCESS_THRESHOLD,
        help=f"beta-binomial 的「成功」阈值（默认 {DEFAULT_SUCCESS_THRESHOLD}）",
    )
    parser.add_argument("--prompts-per-batch", type=int, default=DEFAULT_PROMPTS_PER_BATCH)
    parser.add_argument("--max-batches", type=int, default=DEFAULT_MAX_BATCHES)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="bootstrap 次数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--keep-invalid", action="store_true", help="保留 sampling-invalid 轨迹")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非 markdown")
    parser.add_argument("--out", type=Path, default=None, help="把 markdown 报告写到该路径")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sizes = tuple(int(part) for part in str(args.group_sizes).split(",") if part.strip())
    except ValueError:
        print("--group-sizes 需要逗号分隔的整数", file=sys.stderr)
        return 2
    if not sizes or any(n < 2 for n in sizes):
        print("--group-sizes 里每个 n 都必须 ≥ 2", file=sys.stderr)
        return 2
    try:
        groups = load_groups(args.trajectories, keep_invalid=args.keep_invalid)
    except (OSError, ValueError) as exc:
        print(f"读取失败：{exc}", file=sys.stderr)
        return 2

    summary = analyze(
        groups,
        group_sizes=sizes,
        prompts_per_batch=args.prompts_per_batch,
        max_batches=args.max_batches,
        target=args.target,
        success_threshold=args.success_threshold,
        trials=args.trials,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else render(summary))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render(summary), encoding="utf-8")
        print(f"\n报告已写入 {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
