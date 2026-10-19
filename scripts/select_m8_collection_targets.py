#!/usr/bin/env python3
"""M8 Step 1：从探针里挑「预算卡在边缘」的题，作为核验型采集的目标。

为什么这样挑
------------
M8-B 实测（`selection_vs_retrieval.md`）与硬门分布（`wp_variance.md` §2）给出三条事实：

1. 买错里 **80.6% 涉及超预算**，纯品类错只占 19.4%；
2. **26.4% 的买错，买的就是正确商品**，唯一的问题是没核对「选完规格后的实际价格」；
3. 现有 SFT 池 **89.2% 的轨迹只打开 1 个商品就下单** ⇒ 模型没被示范过「核验」。

所以采集要挑的不是「难题」，而是**「成功与失败只差一次价格核验」的题**——这类题
教「下单前核对实际价格」的性价比最高。

分档（优先级递减，互斥，取最高档）
----------------------------------
* **A 档**：组内出现过 `bought_gold_failed_gate`（买对了商品、只是超支）。
  最直接：正确商品已经找得到，只缺最后一道核价。
* **B 档**：组内出现过 `saw_gold_bought_other` 且替代品 `weighted_score ≥ 0.70`
  （打开过正确商品，却买了几乎一样的另一个）。
* **C 档**：有失败且失败里涉及超预算，但没有 A/B 的证据（多为「没找到」）。
* **D 档**：其余可诊断的题（默认不输出）。

入选前提：**至少 1 次成功 + 至少 1 次失败**（否则不是「边缘」，要么全对要么全错）。

用法::

    .venv/bin/python scripts/select_m8_collection_targets.py \
        outputs/probe/m6r-full/trajectories.jsonl \
        --out data/collection/m8_verification_targets.jsonl \
        --report outputs/probe/m6r-full/m8_targets.md \
        --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import analyze_selection_vs_retrieval as base  # noqa: E402

DEFAULT_HELD_OUT = ROOT / "data" / "evaluation" / "tasks.jsonl"
# B 档「几乎一样」的门槛：与 `analyze_selection_vs_retrieval.CLOSE_MATCH` 保持一致。
CLOSE_MATCH = base.CLOSE_MATCH
TIERS = ("A", "B", "C", "D")


def load_task_ids(path: Path) -> set[int]:
    """读 `{"task_id": N}` 形式的 jsonl；文件不存在时返回空集。"""
    ids: set[int] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, Mapping):
                continue
            try:
                ids.add(int(payload["task_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    return ids


def classify_task(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """给一道题分档。返回 ``None`` 表示不入池。"""
    successes = [r for r in records if base.is_success(r["reward_type"])]
    failures = [r for r in records if not base.is_success(r["reward_type"])]
    if not successes or not failures:
        return None

    evidence = [
        r for r in records
        if r["purchased_asin"]
        and (r["reward_type"] == base.GOLD_TYPE or r["target_asin_match"])
    ]
    if not evidence:
        return None
    gold_asins = {r["purchased_asin"] for r in evidence}
    if len(gold_asins) > 1:
        return None  # 口径不一致，不进池，避免污染采集目标
    gold_asin = next(iter(gold_asins))

    buckets: Counter = Counter()
    budget_failed = 0
    close_substitute = 0
    for record in failures:
        saw = gold_asin in record["opened"]
        if record["purchased_asin"] == gold_asin:
            buckets["bought_gold_failed_gate"] += 1
        elif saw and record["purchased_asin"]:
            buckets["saw_gold_bought_other"] += 1
            if record["match_score"] >= CLOSE_MATCH:
                close_substitute += 1
        elif saw:
            buckets["saw_gold_no_purchase"] += 1
        else:
            buckets["never_saw_gold"] += 1
        if record["budget_failed"]:
            budget_failed += 1

    if buckets["bought_gold_failed_gate"]:
        tier = "A"
    elif close_substitute:
        tier = "B"
    elif budget_failed:
        tier = "C"
    else:
        tier = "D"

    successes = len(records) - len(failures)
    return {
        "task_id": int(records[0]["task_id"]) if str(records[0]["task_id"]).isdigit() else records[0]["task_id"],
        "tier": tier,
        "attempts": len(records),
        "failures": len(failures),
        # 同一题是否至少成功过一次：决定「重新采样能不能收到正样本」的命中率。
        # 163 题里 83 题赢过（50.9%），这 83 题才是高产出采集目标。
        "successes": successes,
        "winnable": successes > 0,
        "budget_failed": budget_failed,
        # M8 真正的靶点：买对了商品但规格选错（`key_options` 未满），不是超支。
        "spec_failed": buckets["bought_gold_failed_gate"],
        "bought_gold_failed_gate": buckets["bought_gold_failed_gate"],
        "saw_gold_bought_other": buckets["saw_gold_bought_other"],
        "close_substitute": close_substitute,
        "never_saw_gold": buckets["never_saw_gold"],
        "budget_share": budget_failed / len(failures) if failures else 0.0,
    }


def _sort_key(item: Mapping[str, Any]) -> tuple:
    """档位优先；同档内先要有「这题赢过」，再看规格错得多不多。

    M8 已定案：401 条「买对 gold 却只拿 partial」的唯一扣分项是 `key_options`，
    所以排序不再看超预算，而是看「规格错」的集中度与重新采样的命中率。
    """
    return (
        TIERS.index(str(item["tier"])),
        0 if item.get("winnable") else 1,
        -int(item.get("spec_failed", 0)),
        -int(item["failures"]),
        str(item["task_id"]),
    )


def select(
    rows: Sequence[Mapping[str, Any]],
    *,
    held_out: set[int],
    min_group_size: int = 2,
    include_d: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        record = base.extract(row)
        groups.setdefault(record["task_id"], []).append(record)

    tiers: Counter = Counter()
    selected: list[dict[str, Any]] = []
    excluded_held_out = 0
    skipped_no_mix = 0
    for records in groups.values():
        if len(records) < min_group_size:
            continue
        result = classify_task(records)
        if result is None:
            skipped_no_mix += 1
            continue
        try:
            task_id = int(result["task_id"])
        except (TypeError, ValueError):
            continue
        if task_id in held_out:
            excluded_held_out += 1
            continue
        tiers[str(result["tier"])] += 1
        if result["tier"] == "D" and not include_d:
            continue
        selected.append(result)

    selected.sort(key=_sort_key)
    if limit is not None:
        selected = selected[:limit]
    return {
        "tasks_total": len(groups),
        "skipped_no_mix": skipped_no_mix,
        "excluded_held_out": excluded_held_out,
        "tiers": dict(tiers),
        "selected": selected,
    }


def render_report(result: Mapping[str, Any], *, source: str, limit: int | None) -> str:
    selected = result["selected"]
    lines: list[str] = []
    lines.append("# M8 Step 1 采集目标（规格选错题）")
    lines.append("")
    lines.append(f"> 输入：`{source}`；held-out 已剔除 **{result['excluded_held_out']}** 题。")
    lines.append(
        f"> 跳过：无「成功+失败」混合或推不出正确商品的题 **{result['skipped_no_mix']}** 个。"
    )
    lines.append(f"> 输出：**{len(selected)}** 题" + (f"（上限 {limit}）" if limit else "") + "。")
    lines.append("")

    lines.append("## 1. 分档")
    lines.append("")
    lines.append("| 档 | 含义 | 题数 |")
    lines.append("|---|---|---:|")
    lines.append(
        f"| **A** | **买对了商品，但规格选错/没选全**（M8 唯一靶点） | "
        f"{result['tiers'].get('A', 0)} |"
    )
    lines.append(
        f"| **B** | 打开过正确商品，却买了 ≥0.70 的近似品（无 A 证据时） | "
        f"{result['tiers'].get('B', 0)} |"
    )
    lines.append(
        f"| **C** | 有超预算失败，但无 A/B 证据（多为没找到） | "
        f"{result['tiers'].get('C', 0)} |"
    )
    lines.append(f"| D | 其余可诊断题（默认不输出） | {result['tiers'].get('D', 0)} |")
    lines.append("")

    if selected:
        lines.append("## 2. 入选明细（前 30）")
        lines.append("")
        lines.append(
            "| task_id | 档 | 赢过 | 失败数 | 规格错 | 超预算 | 近似替代 | 没看到 |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for item in selected[:30]:
            lines.append(
                f"| {item['task_id']} | **{item['tier']}** | "
                f"{'是' if item.get('winnable') else '否'} | {item['failures']} | "
                f"{item.get('spec_failed', 0)} | {item['budget_failed']} | "
                f"{item['close_substitute']} | {item['never_saw_gold']} |"
            )
        lines.append("")

        winnable = sum(1 for item in selected if item.get("winnable"))
        spec = sum(int(item.get("spec_failed", 0)) for item in selected)
        lines.append("## 3. 质量检查")
        lines.append("")
        lines.append(
            f"- 入选题里「至少赢过一次」的：**{winnable}/{len(selected)}"
            f"（{winnable / len(selected):.1%}）**"
            "——这批重新采样能收到正样本，命中率最高"
        )
        lines.append(
            f"- 入选题里「买对商品但规格错」的失败次数合计：**{spec}**"
            "（M8 的唯一扣分项，见 `plans/m8-data-plan.md` §7.0.3）"
        )
        lines.append(
            f"- A 档占比：**{result['tiers'].get('A', 0) / max(sum(result['tiers'].values()), 1):.1%}**"
        )
        lines.append("")

    lines.append("## 4. 下一步命令")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python scripts/collect_sft_data.py \\")
    lines.append("  --tasks data/collection/m8_verification_targets.jsonl \\")
    lines.append("  --output-dir outputs/sft-collection-m8 \\")
    lines.append("  --attempts-per-task 8 --temperature 0.7 --top-p 0.95 \\")
    lines.append("  --workers 4")
    lines.append("```")
    lines.append("")
    lines.append(
        "采集完再用闸门筛选（`open_product ≥ 2` 且 `gold_purchase` 且 `search_products ≥ 2`），"
        "见 `plans/m8-data-plan.md` §3 Step 1。"
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
    parser.add_argument("trajectories", type=Path, help="探针轨迹 jsonl（同题多采样）")
    parser.add_argument(
        "--out", type=Path, default=Path("data/collection/m8_verification_targets.jsonl"),
        help="输出的 task_id jsonl（可直接传给 collect_sft_data.py --tasks）",
    )
    parser.add_argument("--report", type=Path, default=None, help="markdown 报告路径")
    parser.add_argument(
        "--held-out", type=Path, default=DEFAULT_HELD_OUT,
        help="评测题 id 文件，绝不进采集目标",
    )
    parser.add_argument("--limit", type=int, default=100, help="最多输出多少题")
    parser.add_argument(
        "--min-group-size", type=int, default=2, help="至少几条同题轨迹才算一组"
    )
    parser.add_argument("--include-d", action="store_true", help="连 D 档一起输出")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.trajectories.exists():
        print(f"找不到轨迹文件：{args.trajectories}", file=sys.stderr)
        return 2
    rows = load_rows(args.trajectories)
    result = select(
        rows,
        held_out=load_task_ids(args.held_out),
        min_group_size=args.min_group_size,
        include_d=args.include_d,
        limit=args.limit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for item in result["selected"]:
            handle.write(
                json.dumps(
                    {
                        "task_id": item["task_id"],
                        "tier": item["tier"],
                        "failures": item["failures"],
                        "budget_failed": item["budget_failed"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"已写入 {args.out}（{len(result['selected'])} 题）")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            render_report(result, source=str(args.trajectories), limit=args.limit) + "\n",
            encoding="utf-8",
        )
        print(f"已写入 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
