"""GRPO 组内信号诊断：纯函数，不依赖 torch / veRL / 环境服务。

用途：在**不训练**的前提下，用一批同题多次采样的轨迹回答三个问题
  1. 终局奖励在组内是否真的有差异（零方差组 = 零学习信号，会被动态采样丢弃）
  2. 组内"行为"是否多样（打开的商品、搜索词、工具序列、最终购买）
  3. 把 n 从 4 提到 8，能救回多少组

判定口径与 `dynamic_sampling.select_reward_varying_groups` 严格对齐：
  - 组内存在 sampling_invalid（基础设施无效 / reward 不可核验 / 超长）→ 丢弃
  - 组内 terminal_utility 的 max-min <= tolerance → 丢弃（constant_reward）
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

DEFAULT_TOLERANCE = 1.0e-8


def _tool_names(trajectory: Mapping) -> list[str]:
    return [str(step.get("tool_name")) for step in trajectory.get("steps") or []]


def _param_values(trajectory: Mapping, tool_name: str, key: str) -> list[str]:
    values = []
    for step in trajectory.get("steps") or []:
        if step.get("tool_name") != tool_name:
            continue
        parameters = step.get("parameters") or {}
        value = parameters.get(key)
        if value is not None:
            values.append(str(value))
    return values


def _purchase_asin(trajectory: Mapping) -> str | None:
    """从终局结果里取实际购买的商品；未购买返回 None。"""
    terminal = trajectory.get("terminal_result") or {}
    purchase = terminal.get("purchase")
    if isinstance(purchase, Mapping):
        asin = purchase.get("asin")
        if asin:
            return str(asin)
    for step in reversed(trajectory.get("steps") or []):
        if step.get("tool_name") == "buy_now":
            asin = (step.get("parameters") or {}).get("asin")
            if asin:
                return str(asin)
    return None


def _reward_detail(trajectory: Mapping) -> Mapping:
    """终局 reward 明细。

    真实轨迹（`evaluation/rollout.py` 写出的 `terminal_result`）把 reward 字段放在
    **嵌套的** `terminal_result["reward_detail"]` 下（见 `evaluation/trajectory.py::_terminal_view`
    与 `evaluation/metrics.py`）；部分离线夹具把它直接摊平在 `terminal_result` 上。
    两种都支持，优先取嵌套形式。
    """
    terminal = trajectory.get("terminal_result") or {}
    detail = terminal.get("reward_detail")
    if isinstance(detail, Mapping) and detail:
        return detail
    return terminal


def _reward_type(trajectory: Mapping) -> str:
    detail = _reward_detail(trajectory)
    return str(
        detail.get("reward_type")
        or detail.get("termination_reason")
        or trajectory.get("status")
        or "unknown"
    )


def _sampling_invalid_reasons(trajectory: Mapping) -> list[str]:
    reasons = []
    if trajectory.get("error"):
        reasons.append("error")
    if trajectory.get("release_error"):
        reasons.append("release_error")
    detail = _reward_detail(trajectory)
    if detail.get("reward_valid") is False:
        reasons.append("reward_unverifiable")
    reward_type = _reward_type(trajectory)
    if reward_type == "reward_unverifiable":
        reasons.append("reward_unverifiable")
    if reward_type == "infrastructure_invalid":
        reasons.append("infrastructure_invalid")
    return sorted(set(reasons))


def behavior_fingerprint(trajectory: Mapping) -> dict[str, object]:
    """把一条轨迹压成可比对的"行为指纹"。"""
    return {
        "open_set": tuple(sorted(set(_param_values(trajectory, "open_product", "asin")))),
        "query_set": tuple(
            sorted(set(_param_values(trajectory, "search_products", "query")))
        ),
        "purchase": _purchase_asin(trajectory),
        "tool_sequence": tuple(_tool_names(trajectory)),
    }


def group_trajectories(
    trajectories: Iterable[Mapping],
) -> dict[int, list[Mapping]]:
    grouped: dict[int, list[Mapping]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[int(trajectory["task_id"])].append(trajectory)
    return dict(grouped)


def summarize_group(
    items: Sequence[Mapping],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    n: int | None = None,
) -> dict[str, object]:
    """对一道题的全部 attempt 求组内统计。``n`` 用于模拟"只采 n 条"。"""
    if n is not None:
        items = list(items)[: int(n)]
    if not items:
        raise ValueError("summarize_group requires at least one trajectory")

    utilities = [float(item.get("final_reward", 0.0)) for item in items]
    if any(not math.isfinite(value) for value in utilities):
        raise ValueError("final_reward must be finite")

    fingerprints = [behavior_fingerprint(item) for item in items]
    open_sets = {fingerprint["open_set"] for fingerprint in fingerprints}
    query_sets = {fingerprint["query_set"] for fingerprint in fingerprints}
    purchases = {fingerprint["purchase"] for fingerprint in fingerprints}
    tool_sequences = {fingerprint["tool_sequence"] for fingerprint in fingerprints}

    reasons = sorted(
        {reason for item in items for reason in _sampling_invalid_reasons(item)}
    )
    utility_min = min(utilities)
    utility_max = max(utilities)
    reward_varying = utility_max - utility_min > tolerance

    if reasons:
        drop_reason: str | None = "sampling_invalid"
    elif not reward_varying:
        drop_reason = "constant_reward"
    else:
        drop_reason = None

    return {
        "task_id": int(items[0]["task_id"]),
        "n": len(items),
        "terminal_utilities": utilities,
        "utility_min": utility_min,
        "utility_max": utility_max,
        "utility_std": statistics.pstdev(utilities) if len(utilities) > 1 else 0.0,
        "reward_varying": reward_varying,
        "reward_types": [_reward_type(item) for item in items],
        "purchase_success": [
            _reward_type(item) in {"gold_purchase", "valid_alternative_purchase"}
            for item in items
        ],
        "sampling_invalid_reasons": reasons,
        "drop_reason": drop_reason,
        "kept": drop_reason is None,
        "distinct_open_sets": len(open_sets),
        "distinct_query_sets": len(query_sets),
        "distinct_purchases": len(purchases),
        "distinct_tool_sequences": len(tool_sequences),
        "mean_steps": statistics.mean(len(item.get("steps") or []) for item in items),
        "guard_blocked": sum(len(item.get("blocked_tool_calls") or []) for item in items),
        "truncations": sum(
            len(item.get("tool_call_truncations") or []) for item in items
        ),
        "compactions": sum(len(item.get("context_compactions") or []) for item in items),
    }


def aggregate_groups(
    group_rows: Sequence[Mapping], *, tolerance: float = DEFAULT_TOLERANCE
) -> dict[str, object]:
    """把组级统计汇总成探针报告。"""
    if not group_rows:
        raise ValueError("aggregate_groups requires at least one group")

    total = len(group_rows)
    constant = sum(not row["reward_varying"] for row in group_rows)
    invalid = sum(bool(row["sampling_invalid_reasons"]) for row in group_rows)
    kept = sum(bool(row["kept"]) for row in group_rows)

    def rate(count: int) -> float:
        return count / total

    def mean_of(key: str) -> float:
        return statistics.mean(float(row[key]) for row in group_rows)

    def rate_ge2(key: str) -> float:
        return sum(int(row[key]) >= 2 for row in group_rows) / total

    reward_type_counts: dict[str, int] = defaultdict(int)
    for row in group_rows:
        for reward_type in row["reward_types"]:
            reward_type_counts[reward_type] += 1

    utilities = [value for row in group_rows for value in row["terminal_utilities"]]

    return {
        "num_groups": total,
        "num_trajectories": sum(int(row["n"]) for row in group_rows),
        "tolerance": tolerance,
        "signal": {
            "constant_reward_group_rate": rate(constant),
            "sampling_invalid_group_rate": rate(invalid),
            "kept_group_rate": rate(kept),
            "dropped_group_rate": rate(total - kept),
            "all_purchase_success_group_rate": rate(
                sum(all(row["purchase_success"]) for row in group_rows)
            ),
            "no_purchase_success_group_rate": rate(
                sum(not any(row["purchase_success"]) for row in group_rows)
            ),
        },
        "diversity": {
            "mean_distinct_open_sets": mean_of("distinct_open_sets"),
            "rate_groups_with_ge2_open_sets": rate_ge2("distinct_open_sets"),
            "mean_distinct_query_sets": mean_of("distinct_query_sets"),
            "rate_groups_with_ge2_query_sets": rate_ge2("distinct_query_sets"),
            "mean_distinct_purchases": mean_of("distinct_purchases"),
            "rate_groups_with_ge2_purchases": rate_ge2("distinct_purchases"),
            "mean_distinct_tool_sequences": mean_of("distinct_tool_sequences"),
        },
        "behaviour": {
            "mean_steps_per_trajectory": mean_of("mean_steps"),
            "guard_blocked_per_group": mean_of("guard_blocked"),
            "truncations_per_group": mean_of("truncations"),
            "compactions_per_group": mean_of("compactions"),
        },
        "reward": {
            "mean_terminal_utility": statistics.mean(utilities),
            "reward_type_counts": dict(sorted(reward_type_counts.items())),
        },
    }


def compare_n(
    grouped: Mapping[int, Sequence[Mapping]],
    n_values: Sequence[int],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, dict[str, object]]:
    """对每个 n 值模拟"只采 n 条"，用于回答"n 4→8 值不值"。

    注意：用同一批最多 ``max(n_values)`` 条轨迹的前 n 条来模拟，因此各 n 之间
    是同题同起点的配对比较。
    """
    result: dict[str, dict[str, object]] = {}
    for n in n_values:
        n = int(n)
        usable = [items for items in grouped.values() if len(items) >= n]
        if not usable:
            continue
        rows = [summarize_group(items, tolerance=tolerance, n=n) for items in usable]
        report = aggregate_groups(rows, tolerance=tolerance)
        report["num_groups_usable"] = len(usable)
        result[f"n={n}"] = report
    return result
