"""把「看到 gold 却没买对」的题按失败原因拆开。

为什么必须拆：服务器实测冠军翻到过 gold 的有 180/200，其中 53 题没买对；
只要修好 13 题（25%）就到 140。但「没买对」有两种，修法完全不同：

- **买错商品**（买的 asin ≠ gold）⇒ 让 agent 打开前几名比较一次，有机会修；
- **买对商品、规格/属性不符**（asin 对、选项错）⇒ 多打开几个商品**完全没用**，
  得改选规格这一步。

不先拆就上干预，等于又一次「先给结论再找证据」。本脚本只统计，不产出决策规则。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ens = _load("eval_ensemble", ROOT / "scripts" / "eval_ensemble.py")
vc = _load("analyze_visible_ceiling", ROOT / "scripts" / "analyze_visible_ceiling.py")

TOP_K = 3


def purchased_asin(record: Mapping) -> str | None:
    purchase = (record.get("terminal_result") or {}).get("purchase") or {}
    asin = purchase.get("asin")
    return str(asin) if asin else None


def asin_match(record: Mapping) -> bool | None:
    """优先用奖励明细里的 `target_asin_match`；没有就自己比。"""
    detail = ((record.get("terminal_result") or {}).get("reward_detail")) or {}
    flag = detail.get("target_asin_match")
    if isinstance(flag, bool):
        return flag
    gold = vc.gold_asin(record)
    bought = purchased_asin(record)
    if gold is None or bought is None:
        return None
    return gold == bought


def classify(record: Mapping) -> str:
    """把一题归到一种失败原因。"""
    if not purchased_asin(record):
        return "没买"
    match = asin_match(record)
    if match is True:
        return "买对商品·规格或属性不符"
    if match is False:
        return "买错商品"
    return "无法判定"


def collect(per_model: Mapping[str, Mapping[int, Mapping]], *, champion: str) -> dict:
    champ = per_model.get(champion, {})
    kinds: Counter = Counter()
    kinds_top3: Counter = Counter()
    bought_ahead: Counter = Counter()
    rows = []
    for task_id, record in sorted(champ.items()):
        gold = vc.gold_asin(record)
        if not gold:
            continue
        seen = vc.visible_asins(record)
        if gold not in seen or ens.is_success(record, strict=True):
            continue
        kind = classify(record)
        gold_rank = seen[gold]
        bought = purchased_asin(record)
        bought_rank = seen.get(bought) if bought else None
        kinds[kind] += 1
        if gold_rank <= TOP_K:
            kinds_top3[kind] += 1
        if kind == "买错商品" and bought_rank is not None:
            if bought_rank < gold_rank:
                bought_ahead["买了更靠前的"] += 1
            elif bought_rank > gold_rank:
                bought_ahead["买了更靠后的"] += 1
            else:
                bought_ahead["名次相同"] += 1
        rows.append(
            {
                "task_id": task_id,
                "kind": kind,
                "gold_rank": gold_rank,
                "bought_asin": bought,
                "bought_rank": bought_rank,
                "reward_type": (
                    ((record.get("terminal_result") or {}).get("reward_detail")) or {}
                ).get("reward_type"),
            }
        )
    return {
        "n_failed": sum(kinds.values()),
        "kinds": dict(kinds),
        "kinds_top3": dict(kinds_top3),
        "bought_rank_vs_gold": dict(bought_ahead),
        "tasks": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path, default=ens.DEFAULT_REFERENCE)
    parser.add_argument("--champion", type=str, required=True)
    parser.add_argument("--min-tasks", type=int, default=100)
    parser.add_argument("--min-overlap", type=float, default=0.8)
    parser.add_argument("--target", type=int, default=140)
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
    if args.champion not in kept:
        raise SystemExit(f"冠军目录 {args.champion!r} 不在可用目录里：{sorted(kept)}")

    stats = collect(kept, champion=args.champion)
    total = len(kept[args.champion])
    success = sum(1 for row in kept[args.champion].values() if ens.is_success(row, strict=True))

    print(f"# 选择失败归因：冠军 `{args.champion}`，{total} 题，做对 {success}")
    print()
    print(f"看到 gold 却没买对：**{stats['n_failed']} 题**")
    print()
    print(f"| 失败原因 | 题数 | 其中 gold 在前 {TOP_K} 名 | 能靠「多开几个商品」修吗 |")
    print("|---|---:|---:|---|")
    for kind in ("买错商品", "买对商品·规格或属性不符", "没买", "无法判定"):
        count = stats["kinds"].get(kind, 0)
        if not count:
            continue
        top3 = stats["kinds_top3"].get(kind, 0)
        fixable = "能" if kind == "买错商品" else "不能"
        print(f"| {kind} | {count} | {top3} | {fixable} |")
    print()

    if stats["bought_rank_vs_gold"]:
        print("## 买错商品时，买的东西比 gold 更靠前还是更靠后")
        print()
        print("| 情形 | 题数 |")
        print("|---|---:|")
        for label, count in sorted(stats["bought_rank_vs_gold"].items()):
            print(f"| {label} | {count} |")
        print()

    wrong_top3 = stats["kinds_top3"].get("买错商品", 0)
    spec = stats["kinds"].get("买对商品·规格或属性不符", 0)
    need = args.target - success

    print("## 判读")
    print()
    print(f"- 缺口：{args.target} − {success} = **{need} 题**")
    print(
        f"- 「打开前 {TOP_K} 名再比一次」能覆盖的上限：**{wrong_top3} 题**（买错商品且 gold 在前 {TOP_K} 名）"
    )
    if wrong_top3 >= need:
        print(
            f"- **机制成立**：{wrong_top3} ≥ {need}，只要这套动作修好其中 "
            f"{need}/{wrong_top3} = {need / wrong_top3:.0%}，就到 {args.target}。"
        )
    else:
        print(
            f"- **机制不够**：{wrong_top3} < {need}，光靠「多开几个商品」到不了 {args.target}，"
            f"还得同时修规格选择（那一档有 {spec} 题）。"
        )
    print(f"- 规格类失败 {spec} 题**不会**因为多开商品而改善，需要单独干预。")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
