#!/usr/bin/env python3
"""Gate 3：v5 候选公式的**梯度份额**（决定「值不值得开跑」）。

为什么需要这一层
----------------
`configs/grpo.yaml` 的 `algorithm.norm_adv_by_std_in_grpo` 是 **false** ⇒ verl 的
GRPO advantage 是 `A_i = r_i - mean(group)`，**不除标准差**。因此：

* 带宽（span）直接决定梯度量级，`scale` 不是无量纲系数；
* 「v5 把 1 个取值变成 14 个」是**题与题之间**的分辨率，对 advantage 毫无贡献。

Gate 2 只回答「组内有没有差异」，不回答「差异够不够大」。本脚本量化后者：

    share = Σ_{回收组} Σ_i |A_i| · L_i  /  Σ_{全部 kept 组} Σ_i |A_i| · L_i

其中「回收组」= 组内 4 条全是 `wrong_purchase`（v4 下 reward 全等 ⇒ 被
`dynamic_sampling` 丢弃），在候选公式下变得有差异的组。

判据（预登记，见 `plans/m7-implementation.md §7 Gate 3`）：

| share | 裁定 |
|---|---|
| ≥ 0.10 | 通过 |
| 0.05 ~ 0.10 | 边际：可开跑，但只作「机制验证」 |
| < 0.05 | 不通过：结构性空臂，不值得花 2.7 h |

用法::

    .venv/bin/python scripts/analyze_v5_gradient_share.py \
        outputs/probe/m6r-full/trajectories.jsonl \
        --out outputs/probe/m6r-full/v5_gradient_share.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts", ROOT / "environments" / "ShopSimulator" / "shop_env"):
    if str(extra) not in sys.path:
        sys.path.append(str(extra))

from analyze_wrong_purchase_variance import (  # noqa: E402
    CANDIDATES,
    WRONG_TYPE,
    _candidate_value,
    _reward_detail,
    budget_proximity,
    extract_wrong_purchases,
)
from web_agent_site.engine.reward import DEFAULT_REWARDS  # noqa: E402

SHARE_PASS = 0.10
SHARE_MARGINAL = 0.05
TOLERANCE = 1.0e-8
V4_WRONG = float(DEFAULT_REWARDS["wrong_purchase"])
# 轨迹长度字段的候选键（探针/评测产物命名不一，缺失时退化为均匀权重）。
LENGTH_KEYS = ("response_tokens", "num_response_tokens", "completion_tokens", "tokens")


def _length(row: Mapping[str, Any]) -> float:
    """轨迹长度权重；找不到就用 1.0（等价于「按轨迹数」而非「按 token 数」）。"""
    for key in LENGTH_KEYS:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0:
            return float(value)
    usage = row.get("usage")
    if isinstance(usage, Mapping):
        value = usage.get("completion_tokens")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    steps = row.get("steps")
    if isinstance(steps, Sequence) and steps:
        return float(len(steps))
    return 1.0


def _v4_reward(detail: Mapping[str, Any]) -> float:
    try:
        return float(detail.get("terminal_utility", detail.get("reward")))
    except (TypeError, ValueError):
        return 0.0


def collect_groups(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """按 `task_id` 聚成组，保留每条的 (类型, ms, prox, 长度, v4 奖励)。"""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        detail = _reward_detail(row)
        if not detail:
            continue
        task_id = row.get("task_id", detail.get("task_id"))
        if task_id is None:
            continue
        budget = (detail.get("hard_gates") or {}).get("budget") or {}
        buckets[str(task_id)].append(
            {
                "reward_type": str(detail.get("reward_type", "unknown")),
                "match_score": float(detail.get("weighted_score") or 0.0),
                "proximity": budget_proximity(budget.get("required"), budget.get("actual")),
                "length": _length(row),
                "v4": _v4_reward(detail),
            }
        )
    return [
        {"task_id": task_id, "samples": samples}
        for task_id, samples in buckets.items()
        if len(samples) >= 2
    ]


def gradient_share(groups: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    """某候选公式的梯度份额。

    非 `wrong_purchase` 样本保留 v4 落盘奖励（v5 不碰它们），
    `wrong_purchase` 样本换成候选公式的值。
    """
    recovered_weight = 0.0
    kept_weight = 0.0
    recovered: list[dict[str, Any]] = []
    kept_groups = 0
    uniform_lengths = True
    for group in groups:
        samples = group["samples"]
        rewards = []
        weights = []
        for sample in samples:
            if sample["reward_type"] == WRONG_TYPE:
                reward = _candidate_value(
                    name, sample["match_score"], sample["proximity"]
                )
            else:
                reward = sample["v4"]
            rewards.append(reward)
            weights.append(sample["length"])
            if sample["length"] != 1.0:
                uniform_lengths = False
        spread = max(rewards) - min(rewards)
        if spread <= TOLERANCE:
            continue  # 该组仍会被 dynamic_sampling 丢弃，不贡献梯度
        kept_groups += 1
        mean = sum(rewards) / len(rewards)
        group_weight = sum(
            abs(reward - mean) * weight
            for reward, weight in zip(rewards, weights, strict=True)
        )
        kept_weight += group_weight
        if all(s["reward_type"] == WRONG_TYPE for s in samples):
            recovered_weight += group_weight
            recovered.append(
                {
                    "task_id": group["task_id"],
                    "spread": spread,
                    "group_weight": group_weight,
                }
            )
    share = recovered_weight / kept_weight if kept_weight else 0.0
    if share >= SHARE_PASS:
        verdict = "pass"
    elif share >= SHARE_MARGINAL:
        verdict = "marginal"
    else:
        verdict = "fail"
    return {
        "name": name,
        "spec": CANDIDATES[name],
        "groups": len(groups),
        "kept_groups": kept_groups,
        "recovered_groups": len(recovered),
        "share": share,
        "verdict": verdict,
        "median_recovered_spread": (
            statistics.median(r["spread"] for r in recovered) if recovered else None
        ),
        "length_basis": "uniform" if uniform_lengths else "trajectory_length",
        "recovered": sorted(recovered, key=lambda r: -r["group_weight"])[:20],
    }


def build_report(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = collect_groups(rows)
    results = [gradient_share(groups, name) for name in CANDIDATES]
    return {"source": str(path), "rows": len(rows), "groups": len(groups), "formulas": results}


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render(report: Mapping[str, Any]) -> str:
    lines = ["# v5 候选公式的梯度份额（Gate 3）", ""]
    lines.append(
        f"> 输入：`{report['source']}`，{report['rows']} 条轨迹 / {report['groups']} 个多采样组。"
    )
    lines.append(
        "> `norm_adv_by_std_in_grpo=false` ⇒ `A = r − mean(group)`，带宽直接决定梯度量级。"
    )
    lines.append("")
    lines.append("| 候选 | 轴 | scale | cap | 回收组 | kept 组 | 中位组内跨度 | 梯度份额 | 裁定 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for item in report["formulas"]:
        spec = item["spec"]
        spread = item["median_recovered_spread"]
        lines.append(
            f"| `{item['name']}` | `{spec['axis']}` | {spec['scale']} | {spec['cap']} | "
            f"{item['recovered_groups']} | {item['kept_groups']} | "
            f"{'—' if spread is None else f'{spread:.4f}'} | "
            f"**{_pct(item['share'])}** | **{item['verdict']}** |"
        )
    lines.append("")
    lines.append(
        f"判据（预登记）：≥{SHARE_PASS:.0%} 通过；{SHARE_MARGINAL:.0%}~{SHARE_PASS:.0%} "
        f"边际（只作机制验证）；<{SHARE_MARGINAL:.0%} 不通过（结构性空臂）。"
    )
    lines.append("")
    basis = {item["length_basis"] for item in report["formulas"]}
    lines.append(
        "> 长度权重口径：" + "、".join(sorted(basis)) + "。`uniform` 表示产物没有 token 数字段，"
        "份额按「轨迹数」而非「token 数」计；同题轨迹长度相近，偏差可忽略但需在报告中注明。"
    )
    lines.append("")
    lines.append("## 回收组明细（按梯度质量降序，前 20）")
    lines.append("")
    for item in report["formulas"]:
        if not item["recovered"]:
            lines.append(f"- `{item['name']}`：无回收组")
            continue
        top = item["recovered"][0]
        lines.append(
            f"- `{item['name']}`：{item['recovered_groups']} 组，"
            f"最大组 task_id={top['task_id']}（跨度 {top['spread']:.4f}）"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trajectories", type=Path, help="同题多采样的轨迹 jsonl")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--out", type=Path, default=None, help="写入该路径")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.trajectories.exists():
        print(f"找不到轨迹文件：{args.trajectories}", file=sys.stderr)
        return 2
    report = build_report(args.trajectories)
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, default=str)
        if args.json
        else render(report)
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
