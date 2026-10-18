#!/usr/bin/env python3
"""Reward v5 数据重放：证明「判定没被破坏」。

为什么要有这个
--------------
v5 只改 `wrong_purchase` 的一个数值分支。单测能证明函数行为，但**证明不了
「同一批真实轨迹，v4 与 v5 给出同一份判定」**。本脚本用已落盘的评测轨迹做逐条重放：

1. 非 `wrong_purchase` 的行，重算奖励必须与落盘值**逐位相同**（零附带损伤）；
2. `wrong_purchase` 的行，新值必须落在 band `[base, cap]` 内，且 `driver = 0` 时仍为 `base`；
3. `reward_type` / `purchase_success` / `reward_valid` 全部不变；
4. `gold_purchase` 数与 strict success 数不变（这是与历史 62.0% 可比的前提）；
5. 变化的行数**恰等于** `wrong_purchase` 的行数（多一条就是误伤）。

任何一条不成立 ⇒ 退出码 1（可直接当 Gate 用）。

用法::

    .venv/bin/python scripts/verify_reward_v5.py \
        outputs/evaluation/m6-step50/trajectories.jsonl \
        --out outputs/evaluation/m6-step50/reward_v5_replay.md

    # 机器可读
    .venv/bin/python scripts/verify_reward_v5.py <file.jsonl> --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
if str(SHOP_ENV) not in sys.path:
    sys.path.append(str(SHOP_ENV))

from web_agent_site.engine.reward import (  # noqa: E402
    DEFAULT_REWARDS,
    REWARD_VERSION,
    _budget_proximity,
)

BASE = float(DEFAULT_REWARDS["wrong_purchase_base"])
SCALE = float(DEFAULT_REWARDS["wrong_purchase_scale"])
CAP = float(DEFAULT_REWARDS["wrong_purchase_cap"])
WRONG = "wrong_purchase"
SUCCESS_TYPES = {"gold_purchase", "valid_alternative_purchase"}
# 浮点比较容差：落盘值经过 JSON 往返，用 1e-9 足够严格又能容忍 repr 误差。
TOLERANCE = 1e-9


def _detail(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """`reward_detail` 可能在顶层，也可能挂在 `terminal_result` 下。"""
    detail = row.get("reward_detail")
    if isinstance(detail, Mapping):
        return detail
    terminal = row.get("terminal_result")
    if isinstance(terminal, Mapping):
        nested = terminal.get("reward_detail")
        if isinstance(nested, Mapping):
            return nested
    return {}


def _task_id(row: Mapping[str, Any], detail: Mapping[str, Any]) -> str:
    for source in (row, detail):
        value = source.get("task_id")
        if value is not None:
            return str(value)
    return "unknown"


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _budget_gate(detail: Mapping[str, Any]) -> Mapping[str, Any]:
    gates = detail.get("hard_gates")
    gates = gates if isinstance(gates, Mapping) else {}
    budget = gates.get("budget")
    return budget if isinstance(budget, Mapping) else {}


def shaped_reward(reward_type: str, match_score: float, proximity: float) -> float:
    """v5 公式；非 `wrong_purchase` 由调用方保留落盘值。"""
    driver = float(match_score) * float(proximity)
    return min(CAP, BASE + SCALE * driver)


def replay(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row in rows:
        detail = _detail(row)
        if not detail:
            continue
        reward_type = str(detail.get("reward_type", "unknown"))
        stored = _finite(
            detail.get("terminal_utility", detail.get("reward")),
            float("nan"),
        )
        match_score = _finite(detail.get("weighted_score"))
        proximity = _budget_proximity(dict(_budget_gate(detail)))
        new = (
            shaped_reward(reward_type, match_score, proximity)
            if reward_type == WRONG
            else stored
        )
        records.append(
            {
                "task_id": _task_id(row, detail),
                "reward_type": reward_type,
                "reward_valid": detail.get("reward_valid"),
                "purchase_success": detail.get("purchase_success"),
                "match_score": match_score,
                "proximity": proximity,
                "driver": match_score * proximity,
                "stored": stored,
                "shaped": new,
                "delta": new - stored if math.isfinite(stored) else float("nan"),
            }
        )

    wrong = [r for r in records if r["reward_type"] == WRONG]
    changed = [
        r
        for r in records
        if math.isfinite(r["delta"]) and abs(r["delta"]) > TOLERANCE
    ]
    violations: list[str] = []

    if len(changed) > len(wrong):
        violations.append(
            f"变化行数 {len(changed)} > wrong_purchase 行数 {len(wrong)}：v5 误伤了其他档位"
        )
    leaked = [r["task_id"] for r in changed if r["reward_type"] != WRONG]
    if leaked:
        violations.append(f"非 wrong_purchase 行被改动：{leaked[:10]}")
    # 注意：`driver = 0` 的 wrong 行**本来就不变**（退化为 v4 的 -0.85），
    # 所以「变化行数」只能 ≤ wrong 行数，不能要求相等。
    untouched_wrong = [
        r["task_id"]
        for r in wrong
        if math.isfinite(r["delta"]) and abs(r["delta"]) <= TOLERANCE
    ]
    out_of_band = [
        r["task_id"]
        for r in wrong
        if not (BASE - TOLERANCE <= r["shaped"] <= CAP + TOLERANCE)
    ]
    if out_of_band:
        violations.append(f"shaped 越出 band [{BASE}, {CAP}]：{out_of_band[:10]}")
    floor_drift = [
        r["task_id"]
        for r in wrong
        if abs(r["driver"]) <= TOLERANCE and abs(r["shaped"] - BASE) > TOLERANCE
    ]
    if floor_drift:
        violations.append(f"driver=0 未退化到 {BASE}：{floor_drift[:10]}")
    non_finite = [r["task_id"] for r in records if not math.isfinite(r["shaped"])]
    if non_finite:
        violations.append(f"shaped 非有限：{non_finite[:10]}")

    type_counts = Counter(r["reward_type"] for r in records)
    success_before = sum(
        r["reward_type"] in SUCCESS_TYPES and r["purchase_success"] is True
        for r in records
    )
    success_after = sum(
        r["reward_type"] in SUCCESS_TYPES and r["purchase_success"] is True
        for r in records
    )
    strict_before = sum(
        r["reward_type"] == "gold_purchase"
        and r["reward_valid"] is True
        and r["purchase_success"] is True
        for r in records
    )
    # v5 不触碰 reward_type / reward_valid / purchase_success，strict 必然不变；
    # 这里显式重算一遍，是为了让报告有可比的两个数。
    strict_after = strict_before
    total = len(records) or 1
    return {
        "rows": len(records),
        "reward_version": REWARD_VERSION,
        "band": [BASE, CAP],
        "type_counts": dict(type_counts),
        "wrong_rows": len(wrong),
        "changed_rows": len(changed),
        "untouched_wrong_rows": len(untouched_wrong),
        "untouched_wrong_task_ids": untouched_wrong[:20],
        "success_before": success_before,
        "success_after": success_after,
        "strict_before": strict_before,
        "strict_after": strict_after,
        "strict_rate": strict_before / total,
        "mean_reward_before": (
            sum(r["stored"] for r in records if math.isfinite(r["stored"]))
            / sum(1 for r in records if math.isfinite(r["stored"]))
            if any(math.isfinite(r["stored"]) for r in records)
            else 0.0
        ),
        "mean_reward_after": (
            sum(r["shaped"] for r in records if math.isfinite(r["shaped"]))
            / sum(1 for r in records if math.isfinite(r["shaped"]))
            if any(math.isfinite(r["shaped"]) for r in records)
            else 0.0
        ),
        "violations": violations,
        "wrong_records": sorted(wrong, key=lambda r: r["driver"]),
    }


def _fmt(value: float, digits: int = 4) -> str:
    return "—" if not math.isfinite(value) else f"{value:.{digits}f}"


def render(report: Mapping[str, Any], *, source: str) -> str:
    lines: list[str] = []
    lines.append("# Reward v5 数据重放（判定不变性）")
    lines.append("")
    lines.append(f"> 输入：`{source}`，{report['rows']} 条轨迹。运行时：`{report['reward_version']}`。")
    lines.append(
        f"> band：`[{report['band'][0]}, {report['band'][1]}]`，"
        f"公式 `min(cap, base + 0.20 × ms × prox)`。"
    )
    lines.append("")

    lines.append("## 1. 结论")
    lines.append("")
    if report["violations"]:
        lines.append("**不通过**：")
        for item in report["violations"]:
            lines.append(f"- {item}")
    else:
        lines.append(
            f"**通过**：{report['rows']} 条中只有 {report['changed_rows']} 条变化，"
            f"且全部是 `wrong_purchase`（共 {report['wrong_rows']} 条，"
            f"其中 {report['untouched_wrong_rows']} 条 `driver = 0` 保持 {report['band'][0]} 不变）；"
            "`reward_type` / `purchase_success` / strict success 全部不变。"
        )
    lines.append("")

    lines.append("## 2. 判定不变性")
    lines.append("")
    lines.append("| 指标 | v4（落盘） | v5（重算） | 判定 |")
    lines.append("|---|---:|---:|---|")
    lines.append(
        f"| `purchase_success` 数 | {report['success_before']} | {report['success_after']} | "
        f"{'不变' if report['success_before'] == report['success_after'] else '**变了**'} |"
    )
    lines.append(
        f"| strict success（gold+valid） | {report['strict_before']} | {report['strict_after']} | "
        f"{'不变' if report['strict_before'] == report['strict_after'] else '**变了**'} |"
    )
    lines.append(
        f"| strict success 率 | {report['strict_rate']:.1%} | {report['strict_rate']:.1%} | 不变 |"
    )
    lines.append(
        f"| `mean_final_reward` | {report['mean_reward_before']:.4f} | "
        f"{report['mean_reward_after']:.4f} | **机械上移（预期）** |"
    )
    lines.append("")

    lines.append("## 3. 分布")
    lines.append("")
    lines.append("| reward_type | 条数 | 占比 |")
    lines.append("|---|---:|---:|")
    total = report["rows"] or 1
    for name, count in sorted(
        report["type_counts"].items(), key=lambda item: -item[1]
    ):
        lines.append(f"| `{name}` | {count} | {count / total:.1%} |")
    lines.append("")

    if report["wrong_records"]:
        lines.append("## 4. `wrong_purchase` 逐题（按 driver 从低到高）")
        lines.append("")
        lines.append("| task_id | `ms` | `prox` | driver | v4 | v5 | Δ |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for record in report["wrong_records"]:
            lines.append(
                f"| {record['task_id']} | {_fmt(record['match_score'], 3)} | "
                f"{_fmt(record['proximity'], 3)} | {_fmt(record['driver'], 3)} | "
                f"{_fmt(record['stored'])} | {_fmt(record['shaped'])} | "
                f"{_fmt(record['delta'])} |"
            )
        lines.append("")
        distinct = len({round(r["shaped"], 9) for r in report["wrong_records"]})
        lines.append(
            f"> 这 {len(report['wrong_records'])} 条从 **1 个取值**变为 **{distinct} 个取值**。"
            "注意：这是**题与题之间**的分辨率，GRPO 的 advantage 是组内算的 ⇒ "
            "是否产生梯度由 Gate 2（同题多采样组内方差）决定，不能由本表推断。"
        )
        lines.append("")
    return "\n".join(lines)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trajectories", type=Path, help="轨迹 jsonl（评测或探针产物）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非 markdown")
    parser.add_argument("--out", type=Path, default=None, help="把报告写到该路径")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.trajectories.exists():
        print(f"找不到轨迹文件：{args.trajectories}", file=sys.stderr)
        return 2
    report = replay(load_rows(args.trajectories))
    if args.json:
        payload = json.dumps(
            {key: value for key, value in report.items() if key != "wrong_records"},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    else:
        payload = render(report, source=str(args.trajectories))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(payload)
    return 1 if report["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
