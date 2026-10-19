"""M8 Step 1 采集后筛选：只留下「规格显式选全且买对」的轨迹。

为什么要有这个
--------------
`plans/m8-data-plan.md` §7.0.3 用服务器实测定案：401 条「买对 gold 却只拿 0.25」
的唯一扣分项是 `key_options`（选规格），`core_functions` 401 条全满分，
`brand`/`model` 的 0.0 多数只是「该题不要求」。所以 M8 要教的是**选对规格**。

因此筛选条件只有两条：

1. ``reward_type == gold_purchase``
   —— 这一条已隐含 ``key_options == 1.0`` 且 ``evidence_coverage == 1.0``，
   因为 ``all_satisfied`` 要求所有维度同时满分。
2. ``select_option`` 次数 ≥ ``key_options.required_count``
   —— 保证「选规格」在轨迹里是**显式、逐个轴**发生的，学生才有动作可学。
   只在结果上选对、过程里看不出来，对 SFT 没有教学价值。

明确**不用**的两条（都有实测反证）：

* ``open_product ≥ 2``：SFT 池 89.2% 的轨迹只开 1 个商品就下单，
  且 ``opens < 2`` 覆盖 99% 的题连 gold 也一样 ⇒ 加了等于扔掉正确数据。
* 「选规格后核对价格」：价格是 ``key_options`` 的下游结果，规格对了价格自然对，
  作为硬门槛只会误杀。

用法::

    .venv/bin/python scripts/filter_m8_verification.py \\
        outputs/sft-collection-m8/raw.jsonl \\
        --out data/collection/m8_verification.jsonl \\
        --report outputs/sft-collection-m8/m8_verification.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.collection.sft import build_sft_row  # noqa: E402

GOLD_TYPE = "gold_purchase"
OPTION_TOOL = "select_option"
OPTION_DIMENSION = "key_options"

# 拒收原因，按「越靠前越根本」的顺序判定，一条轨迹只记第一个命中的原因。
REASON_NO_DETAIL = "no_reward_detail"
REASON_NOT_GOLD = "not_gold_purchase"
REASON_NO_OPTION_REQUIREMENT = "no_option_requirement"
REASON_SELECTS_INCOMPLETE = "selects_less_than_required"
REASON_OVER_TASK_CAP = "over_task_cap"


def _reward_detail(trajectory: Mapping[str, Any]) -> Mapping[str, Any]:
    terminal = trajectory.get("terminal_result")
    if isinstance(terminal, Mapping):
        detail = terminal.get("reward_detail")
        if isinstance(detail, Mapping):
            return detail
    detail = trajectory.get("reward_detail")
    return detail if isinstance(detail, Mapping) else {}


def _steps(trajectory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, Mapping)]


def _tool_name(step: Mapping[str, Any]) -> str:
    return str(step.get("tool_name") or step.get("tool") or "")


def count_selects(trajectory: Mapping[str, Any]) -> int:
    """轨迹里显式选规格的次数（一个轴一次调用）。"""
    return sum(1 for step in _steps(trajectory) if _tool_name(step) == OPTION_TOOL)


def required_option_count(detail: Mapping[str, Any]) -> int:
    """目标要求的规格轴数量；读不到时返回 0（调用方据此拒收）。"""
    evidence = detail.get("evidence")
    if not isinstance(evidence, Mapping):
        return 0
    scoring = evidence.get("preference_scoring")
    if not isinstance(scoring, Mapping):
        return 0
    dimensions = scoring.get("dimensions")
    if not isinstance(dimensions, Mapping):
        return 0
    dimension = dimensions.get(OPTION_DIMENSION)
    if not isinstance(dimension, Mapping):
        return 0
    try:
        return int(dimension.get("required_count") or 0)
    except (TypeError, ValueError):
        return 0


def inspect(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """抽出筛选需要的全部字段，并给出第一条拒收原因（通过则为空）。"""
    detail = _reward_detail(trajectory)
    selects = count_selects(trajectory)
    required = required_option_count(detail)
    reason = ""
    if not detail:
        reason = REASON_NO_DETAIL
    elif str(detail.get("reward_type")) != GOLD_TYPE:
        reason = REASON_NOT_GOLD
    elif required <= 0:
        # 该题不要求规格 ⇒ 这条轨迹教不了「选规格」，即便买对了也不收。
        reason = REASON_NO_OPTION_REQUIREMENT
    elif selects < required:
        reason = REASON_SELECTS_INCOMPLETE
    return {
        "task_id": str(trajectory.get("task_id")),
        "reward_type": str(detail.get("reward_type", "")),
        "selects": selects,
        "required": required,
        "steps": len(_steps(trajectory)),
        "reason": reason,
    }


def select(
    trajectories: Iterable[Mapping[str, Any]],
    *,
    max_per_task: int = 2,
) -> dict[str, Any]:
    """筛选并限制每题条数，保证采集结果覆盖尽可能多的商品。"""
    kept: list[dict[str, Any]] = []
    rejected: Counter = Counter()
    per_task: Counter = Counter()
    seen = 0
    for trajectory in trajectories:
        if not isinstance(trajectory, Mapping):
            continue
        seen += 1
        record = inspect(trajectory)
        if record["reason"]:
            rejected[record["reason"]] += 1
            continue
        if max_per_task > 0 and per_task[record["task_id"]] >= max_per_task:
            rejected[REASON_OVER_TASK_CAP] += 1
            continue
        per_task[record["task_id"]] += 1
        kept.append({"record": record, "trajectory": trajectory})
    return {"seen": seen, "kept": kept, "rejected": dict(rejected)}


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _fmt(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(result: Mapping[str, Any], *, source: str, max_per_task: int) -> str:
    kept = result["kept"]
    seen = result["seen"]
    records = [item["record"] for item in kept]
    tasks = {record["task_id"] for record in records}
    lines = [
        "# M8 Step 1 筛选结果（规格显式选全且买对）",
        "",
        f"> 输入：`{source}`，{seen} 条原始轨迹。",
        f"> 通过：**{len(kept)}** 条，覆盖 **{len(tasks)}** 个 task"
        + (f"（每题最多 {max_per_task} 条）" if max_per_task > 0 else "") + "。",
        "",
        "## 1. 拒收原因",
        "",
        "| 原因 | 条数 | 含义 |",
        "|---|---:|---|",
    ]
    meanings = {
        REASON_NO_DETAIL: "没有 `reward_detail`，无法判定",
        REASON_NOT_GOLD: "不是 `gold_purchase`（没买对）",
        REASON_NO_OPTION_REQUIREMENT: "该题不要求规格，教不了「选规格」",
        REASON_SELECTS_INCOMPLETE: "选了规格，但次数少于要求轴数（过程看不全）",
        REASON_OVER_TASK_CAP: "超过每题条数上限（保证覆盖更多题）",
    }
    for reason, count in sorted(result["rejected"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{reason}` | {count} | {meanings.get(reason, '')} |")
    if not result["rejected"]:
        lines.append("| — | 0 | 无 |")
    lines.append("")

    lines.append("## 2. 通过样本的形态")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 平均工具步数 | {_fmt(_mean([r['steps'] for r in records]))} |")
    lines.append(f"| 平均 `select_option` 次数 | {_fmt(_mean([r['selects'] for r in records]))} |")
    lines.append(f"| 平均要求轴数 | {_fmt(_mean([r['required'] for r in records]))} |")
    lines.append(
        f"| 选规格次数 > 要求轴数（改过主意）的比例 | "
        f"{_share(sum(1 for r in records if r['selects'] > r['required']), len(records))} |"
    )
    lines.append("")

    lines.append("## 3. 闸门（`plans/m8-data-plan.md` Step 1）")
    lines.append("")
    steps_mean = _mean([r["steps"] for r in records])
    gate_count = len(kept) >= 80
    gate_steps = steps_mean is not None and steps_mean <= 14
    lines.append(f"- 可收条数 ≥ 80：**{len(kept)}** {'✅' if gate_count else '❌'}")
    lines.append(f"- 平均工具步数 ≤ 14：**{_fmt(steps_mean)}** {'✅' if gate_steps else '❌'}")
    lines.append("")
    lines.append(
        "两条全过才进 Step 2（造数据 + 只训 SFT）；任一不过 ⇒ 停，转 Step 4 替代路线。"
    )
    return "\n".join(lines) + "\n"


def _share(count: int, total: int) -> str:
    return f"{count}/{total}（{count / total:.1%}）" if total else f"{count}/0"


def load_trajectories(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path, help="采集产出的原始轨迹 jsonl")
    parser.add_argument("--out", type=Path, required=True, help="筛选后的 SFT 行 jsonl")
    parser.add_argument("--report", type=Path, default=None, help="markdown 报告路径")
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=2,
        help="每题最多保留几条（0 表示不限制）；默认 2，保证覆盖更多商品",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.trajectories.exists():
        print(f"找不到输入文件：{args.trajectories}", file=sys.stderr)
        return 2
    result = select(
        load_trajectories(args.trajectories), max_per_task=args.max_per_task
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for item in result["kept"]:
            handle.write(
                json.dumps(build_sft_row(item["trajectory"]), ensure_ascii=False) + "\n"
            )
    print(f"已写入 {args.out}（{len(result['kept'])} 条）")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            render_report(result, source=str(args.trajectories), max_per_task=args.max_per_task),
            encoding="utf-8",
        )
        print(f"已写入 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
