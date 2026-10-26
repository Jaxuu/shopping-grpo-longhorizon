"""量「优化上限」到底在哪：模型到底有没有看到过 gold。

为什么需要它：之前说的 143/200（oracle@top-2）是**裁判挑票数前 2 名**的上限，
而那个裁判被定义成「知道答案」。知道答案的东西不是算法，所以 143 不是能追的目标。
真正能追的目标是另一个数：

**这道题的正确商品，有没有出现在模型自己翻过的搜索结果页里。**

- 没出现 ⇒ 无论选择多完美都不可能对（这是「信息缺失」，只能改检索或改模型）；
- 出现了而没买对 ⇒ 是「选择失败」，**这才是优化能拿到分的地方**。

所以本脚本算三个数：
1. 冠军的「可见上限」：它的轨迹里 gold 出现过几次；
2. 九模型并集的「可见上限」：换一个检索更强的模型能否把 gold 带进来；
3. 在 gold 可见的题里，冠军买对了多少（= 选择精度）。

**用 gold 只做统计，不做选择。** 本脚本不产出任何可用于评测时决策的规则。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 结果行：`1|831919364430|5.0 to…8.0|品牌|品类|属性|标题`
_ROW = re.compile(r"^(\d+)\|([0-9A-Za-z]{6,})\|")
BUCKETS = ((1, 3), (4, 10), (11, 20), (21, 60))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ens = _load("eval_ensemble", ROOT / "scripts" / "eval_ensemble.py")


def search_rows(observation: str) -> list[tuple[int, str]]:
    """从一条 observation 里取搜索结果行的 (名次, asin)。不是搜索页就返回空。"""
    if "page_type: search_results" not in (observation or ""):
        return []
    rows = []
    for line in observation.splitlines():
        match = _ROW.match(line.strip())
        if match:
            rows.append((int(match.group(1)), match.group(2)))
    return rows


def visible_asins(record: Mapping) -> dict[str, int]:
    """这个模型在这道题里翻到过的所有商品：asin → 最好名次。"""
    best: dict[str, int] = {}
    for step in record.get("steps") or []:
        for rank, asin in search_rows(step.get("observation") or ""):
            if asin not in best or rank < best[asin]:
                best[asin] = rank
    return best


def gold_asin(record: Mapping) -> str | None:
    goal = (record.get("terminal_result") or {}).get("goal") or {}
    asin = goal.get("asin")
    return str(asin) if asin else None


def bucket_of(rank: int) -> str:
    for low, high in BUCKETS:
        if low <= rank <= high:
            return f"{low}-{high}"
    return ">60"


def summarize(
    per_model: Mapping[str, Mapping[int, Mapping]],
    *,
    champion: str,
) -> dict:
    """per_model[标签][task_id] = 该题的 record（缺题用 {}）。"""
    tasks = sorted({task_id for rows in per_model.values() for task_id in rows})
    champ = per_model.get(champion, {})

    solvable = 0
    champ_visible = 0
    union_visible = 0
    champ_success = 0
    champ_success_and_visible = 0
    champ_visible_but_failed = 0
    ranks: Counter = Counter()
    bucket_stats: dict[str, dict[str, int]] = {}
    invisible_but_someone_visible = 0
    for task_id in tasks:
        record = champ.get(task_id) or {}
        gold = gold_asin(record) or next(
            (
                gold_asin(rows.get(task_id) or {})
                for rows in per_model.values()
                if gold_asin(rows.get(task_id) or {})
            ),
            None,
        )
        if not gold:
            continue
        solvable += 1
        champ_seen = visible_asins(record)
        if gold in champ_seen:
            champ_visible += 1
            label = bucket_of(champ_seen[gold])
            ranks[label] += 1
            bucket = bucket_stats.setdefault(label, {"total": 0, "success": 0})
            bucket["total"] += 1
            if ens.is_success(record, strict=True):
                champ_success_and_visible += 1
                bucket["success"] += 1
            else:
                champ_visible_but_failed += 1
        else:
            someone = any(
                gold in visible_asins(per_model[label].get(task_id) or {})
                for label in per_model
                if label != champion
            )
            invisible_but_someone_visible += 1 if someone else 0
        if gold in champ_seen or any(
            gold in visible_asins(per_model[label].get(task_id) or {})
            for label in per_model
            if label != champion
        ):
            union_visible += 1
        if ens.is_success(record, strict=True):
            champ_success += 1

    return {
        "n": len(tasks),
        "solvable": solvable,
        "champion": champion,
        "champion_success": champ_success,
        "champion_visible": champ_visible,
        "champion_visible_but_failed": champ_visible_but_failed,
        "champion_success_and_visible": champ_success_and_visible,
        "union_visible": union_visible,
        "champion_invisible_but_others_visible": invisible_but_someone_visible,
        "ranks": dict(sorted(ranks.items())),
        "bucket_stats": bucket_stats,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path, default=ens.DEFAULT_REFERENCE)
    parser.add_argument("--champion", type=str, default=None)
    parser.add_argument("--min-tasks", type=int, default=100)
    parser.add_argument("--min-overlap", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    reference_set = set(ens.probe.task_ids_of(args.reference))
    runs = ens.resolve_runs(args.runs, min_tasks=args.min_tasks)
    kept = {}
    for label, rows in runs.items():
        ids = {int(row["task_id"]) for row in rows if row.get("task_id") is not None}
        if len(ids & reference_set) / max(1, len(reference_set)) >= args.min_overlap:
            kept[label] = {
                int(row["task_id"]): row for row in rows if row.get("task_id") is not None
            }
    if not kept:
        raise SystemExit("没有可用的评测目录。")

    champion = args.champion
    if champion is None:
        champion = max(
            kept,
            key=lambda label: sum(
                1 for row in kept[label].values() if ens.is_success(row, strict=True)
            ),
        )
    if champion not in kept:
        raise SystemExit(f"冠军目录 {champion!r} 不在可用目录里：{sorted(kept)}")

    stats = summarize(kept, champion=champion)
    solvable = stats["solvable"]
    visible = stats["champion_visible"]
    failed_visible = stats["champion_visible_but_failed"]

    print(f"# 可见上限：{len(kept)} 个目录，{stats['n']} 题，冠军 `{champion}`")
    print()
    print("| 项 | 题数 | 占可解题 |")
    print("|---|---:|---:|")
    print(f"| 可解（有 gold） | {solvable} | — |")
    print(
        f"| 冠军做对 | {stats['champion_success']} | {stats['champion_success'] / solvable:.1%} |"
    )
    print(f"| **gold 出现在冠军翻过的搜索页里** | {visible} | {visible / solvable:.1%} |")
    print(f"| 其中买对了 | {stats['champion_success_and_visible']} | — |")
    print(f"| 其中**看到却没买对** | {failed_visible} | — |")
    print(f"| gold 只在别的模型那里可见 | {stats['champion_invisible_but_others_visible']} | — |")
    print(
        f"| 九模型并集可见 | {stats['union_visible']} | {stats['union_visible'] / solvable:.1%} |"
    )
    print()

    print("## gold 在冠军翻到的页里排第几名（以及那一档买对了多少）")
    print()
    print("| 名次 | 题数 | 买对 | 看到却没买对 |")
    print("|---|---:|---:|---:|")
    for label, _ in [(f"{low}-{high}", None) for low, high in BUCKETS] + [(">60", None)]:
        total = stats["ranks"].get(label, 0)
        ok = stats["bucket_stats"].get(label, {}).get("success", 0)
        print(f"| {label} | {total} | {ok} | {total - ok} |")
    print()

    print("## 判读")
    print()
    if visible:
        print(
            f"冠军的选择精度（gold 可见时买对的比例）："
            f"**{stats['champion_success_and_visible']}/{visible} = "
            f"{stats['champion_success_and_visible'] / visible:.1%}**"
        )
    print(
        f"- 若选择完美：冠军上限 **{visible}/{stats['n']} = {visible / stats['n']:.1%}**"
        f"（只需把已经看到的买对，不改检索、不改训练）"
    )
    print(
        f"- 若检索也完美：并集上限 **{stats['union_visible']}/{stats['n']} = "
        f"{stats['union_visible'] / stats['n']:.1%}**"
    )
    target = round(stats["n"] * 0.70)
    print(f"- 目标 {target}/{stats['n']}（70%）")
    top3 = stats["bucket_stats"].get("1-3", {})
    if top3:
        print(
            f"- gold 在前 3 名的共 {top3['total']} 题，已买对 {top3['success']}，"
            f"差 {top3['total'] - top3['success']} 题；这一档全买对就是 "
            f"{top3['total']}/{stats['n']} = {top3['total'] / stats['n']:.1%}"
        )
    if visible >= target:
        need = target - stats["champion_success"]
        print(
            f"- **结论：上限够。** 缺口 {need} 题全部落在「看到却没买对」的 "
            f"{failed_visible} 题里，只需修好 {need}/{failed_visible} = "
            f"{need / failed_visible:.0%} 就能到 70%。"
        )
    else:
        print(
            f"- **结论：上限不够。** 即使把看到的全买对也只有 {visible} 题，差 {target - visible} 题。"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
