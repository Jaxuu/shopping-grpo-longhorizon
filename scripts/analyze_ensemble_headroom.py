#!/usr/bin/env python3
"""多数票离 70% 还差的那些题，到底有没有救。

背景
----
Final-200 上实测（2026-10-26）：10 个评测目录，最好的单模型 63.5%（127），
多数票 64.5%（129），并集 74.0%（148）。目标 70%（140）⇒ **还差 11 题**。

多数票只比单模型高 1 个点，说明「错答案是分散的」这个前提**不成立**——9 个模型
都是同一个底座的 SFT 后代，错在同一件商品上，票数互相加固。这个脚本不猜，
只做三件事：

1. 把多数票判错的题列出来，看**差几票**。差 1 票和差 8 票是完全不同的处境。
2. 算一批**只用票数**的聚合规则各自能到多少，其中最重要的是「只看票数前 k 名、
   由裁判挑」的上界（oracle@top-k）：
   - 上界 ≥ 140 ⇒ 缺的是**选择规则**，还能在聚合这一层想办法；
   - 上界 < 140 ⇒ 缺的是**新信息**，改规则没用，必须让模型自己多对几道。
3. 报「冠军模型在这些错题上对不对」，判断「给冠军加权」值不值。

这三条全部是纯 CPU，读已有轨迹，不碰模型。

用法::

    .venv/bin/python scripts/analyze_ensemble_headroom.py outputs/evaluation/*/
    .venv/bin/python scripts/analyze_ensemble_headroom.py outputs/evaluation/*/ \\
        --champion m10r-final200 --out outputs/analysis/headroom.json
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

TARGET_RATE = 0.70
DEFAULT_CHAMPION = "m10r-final200"


def ranked(votes: Mapping[str, int]) -> list[tuple[str, int]]:
    """票数降序；票数相同按身份字典序，保证可复现（不靠字典插入顺序）。"""
    return sorted(votes.items(), key=lambda item: (-item[1], item[0]))


def top_k(votes: Mapping[str, int], k: int) -> list[str]:
    return [key for key, _ in ranked(votes)[:k]]


def oracle_top_k(
    votes_by_task: Mapping[int, Mapping[str, int]], good: Mapping[int, set[str]], k: int
) -> int:
    """只看票数前 k 名、由裁判挑最好的那个 —— 任何「只读票数」的规则的硬上界。"""
    hit = 0
    for task_id, votes in votes_by_task.items():
        allowed = set(top_k(votes, k))
        if allowed & (good.get(task_id) or set()):
            hit += 1
    return hit


def champion_weighted(
    votes_by_task: Mapping[int, Mapping[str, int]],
    champion_choice: Mapping[int, str],
    good: Mapping[int, set[str]],
    *,
    weight: int = 2,
) -> int:
    """给冠军的票加权后再取多数。`weight=1` 就等价于普通多数票。"""
    hit = 0
    for task_id, votes in votes_by_task.items():
        boosted = Counter({key: count * 1 for key, count in votes.items()})
        pick = champion_choice.get(task_id)
        if pick:
            boosted[pick] += weight - 1
        winner, tied = ens.plurality(boosted)
        if not tied and winner is not None and winner in (good.get(task_id) or set()):
            hit += 1
    return hit


def plurality_hit(
    votes_by_task: Mapping[int, Mapping[str, int]],
    good: Mapping[int, set[str]],
    *,
    tie_break: Mapping[int, str] | None = None,
) -> int:
    stats = ens.vote_summary(votes_by_task, good, tie_break=tie_break)
    return stats["hit_after_break"] if tie_break else stats["hit"]


def diagnose(
    votes_by_task: Mapping[int, Mapping[str, int]],
    good: Mapping[int, set[str]],
    champion_choice: Mapping[int, str],
) -> dict:
    """把「多数票判错的题」拆开：差几票、冠军对不对、前 2 名里有没有正确答案。"""
    margins: Counter = Counter()
    ties = 0
    champion_right = champion_wrong = champion_silent = 0
    in_top2 = 0
    in_top3 = 0
    wrong_tasks: list[dict] = []
    right_tasks: list[dict] = []
    for task_id, votes in votes_by_task.items():
        keys = good.get(task_id) or set()
        winner, tied = ens.plurality(votes)
        ok = (not tied) and winner is not None and winner in keys
        if ok:
            right_tasks.append(
                {"task_id": task_id, "votes": dict(ranked(votes)), "good_keys": sorted(keys)}
            )
            continue
        if tied:
            # 平票没有「赢家票数」可比，单独计一行，不混进差票直方图。
            ties += 1
            margin = None
            winner_votes = max(votes.values(), default=0)
        else:
            winner_votes = votes.get(winner, 0) if winner is not None else 0
            # 正确身份拿到的最多票（可能一个都没人买到 ⇒ 0）
            best_correct = max((count for key, count in votes.items() if key in keys), default=0)
            margin = winner_votes - best_correct
            margins[margin] += 1
        pick = champion_choice.get(task_id)
        if pick is None:
            champion_silent += 1
        elif pick in keys:
            champion_right += 1
        else:
            champion_wrong += 1
        top2 = set(top_k(votes, 2))
        top3 = set(top_k(votes, 3))
        if top2 & keys:
            in_top2 += 1
        if top3 & keys:
            in_top3 += 1
        wrong_tasks.append(
            {
                "task_id": task_id,
                "margin": margin,
                "tied": tied,
                "winner": winner,
                "winner_votes": winner_votes,
                "votes": dict(ranked(votes)),
                "good_keys": sorted(keys),
                "correct_in_top2": bool(top2 & keys),
                "correct_in_top3": bool(top3 & keys),
                "champion_correct": bool(pick is not None and pick in keys),
            }
        )
    return {
        "margins": dict(sorted(margins.items())),
        "ties": ties,
        "champion_right": champion_right,
        "champion_wrong": champion_wrong,
        "champion_silent": champion_silent,
        "correct_in_top2": in_top2,
        "correct_in_top3": in_top3,
        "wrong_tasks": sorted(
            wrong_tasks, key=lambda item: (item["margin"] is None, item["margin"] or 0)
        ),
        "right_tasks": right_tasks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path, default=ens.DEFAULT_REFERENCE)
    parser.add_argument("--champion", default=DEFAULT_CHAMPION, help="用来加权/兜底的目录名")
    parser.add_argument("--min-tasks", type=int, default=100)
    parser.add_argument("--min-overlap", type=float, default=0.8)
    parser.add_argument(
        "--loose",
        action="store_true",
        help="把 purchase_success 也算做对（valid_alternative 会被算成 gold，仅作对照）",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    strict = not args.loose

    reference_ids = ens.probe.task_ids_of(args.reference)
    reference_set = set(reference_ids)
    runs = ens.resolve_runs(args.runs, min_tasks=args.min_tasks)
    for label in list(runs):
        ids = {int(row["task_id"]) for row in runs[label] if row.get("task_id") is not None}
        if len(ids & reference_set) / max(1, len(reference_set)) < args.min_overlap:
            del runs[label]
    if not runs:
        raise SystemExit("没有可用的评测目录。")
    if args.champion not in runs:
        raise SystemExit(f"找不到冠军目录 `{args.champion}`；现有：{', '.join(sorted(runs))}")

    indexed = {
        label: {int(row["task_id"]): row for row in rows if row.get("task_id") is not None}
        for label, rows in runs.items()
    }
    tasks = list(reference_ids)
    votes_by_task: dict[int, Counter] = {}
    good: dict[int, set[str]] = {}
    champion_choice: dict[int, str] = {}
    for task_id in tasks:
        votes: Counter = Counter()
        keys: set[str] = set()
        for rows in indexed.values():
            record = rows.get(task_id)
            if record is None:
                continue
            key = ens.purchased(record)
            if key is None:
                continue
            votes[key] += 1
            if ens.is_success(record, strict=strict):
                keys.add(key)
        votes_by_task[task_id] = votes
        good[task_id] = keys
        pick = ens.purchased(indexed[args.champion].get(task_id) or {})
        if pick:
            champion_choice[task_id] = pick

    n = len(tasks)
    target = round(n * TARGET_RATE)
    champion_ok = sum(1 for task_id in tasks if champion_choice.get(task_id) in good[task_id])
    plain = plurality_hit(votes_by_task, good)
    tie_broken = plurality_hit(votes_by_task, good, tie_break=champion_choice)

    print(f"# 聚合上界分析：{len(runs)} 个目录，{n} 题，目标 {target}（{TARGET_RATE:.0%}）")
    print()
    print("## 各规则能到多少")
    print()
    print("| 规则 | 做对 | 准确率 | 到 70% |")
    print("|---|---:|---:|---|")
    rows = [
        (f"冠军单模型 `{args.champion}`", champion_ok, "基线"),
        ("多数票（平票算失败）", plain, "**当前最好**"),
        ("多数票（平票用冠军兜底）", tie_broken, "固定规则"),
        (
            "冠军票 ×2 后取多数",
            champion_weighted(votes_by_task, champion_choice, good, weight=2),
            "试加权",
        ),
        (
            "冠军票 ×3 后取多数",
            champion_weighted(votes_by_task, champion_choice, good, weight=3),
            "试加权",
        ),
        (
            "**上界：只看票数前 2 名，裁判挑**",
            oracle_top_k(votes_by_task, good, 2),
            "任何读票数规则的天花板",
        ),
        ("**上界：只看票数前 3 名，裁判挑**", oracle_top_k(votes_by_task, good, 3), "同上，更松"),
        ("上界：全部候选，裁判挑（=并集）", sum(1 for keys in good.values() if keys), "绝对天花板"),
    ]
    for name, ok, note in rows:
        mark = "✅" if ok >= target else f"差 {target - ok}"
        print(f"| {name} | {ok} | {ok / n:.1%} | {mark} |")
    print()

    diag = diagnose(votes_by_task, good, champion_choice)
    print("## 多数票判错的题：差几票")
    print()
    print("| 赢家票数 − 正确票数 | 题数 |")
    print("|---|---:|")
    for margin, count in diag["margins"].items():
        print(f"| {margin:+d} | {count} |")
    if diag["ties"]:
        print(f"| 平票（无赢家） | {diag['ties']} |")
    print()
    print(
        f"其中：正确答案仍在票数前 2 名的 **{diag['correct_in_top2']}** 题，"
        f"前 3 名 **{diag['correct_in_top3']}** 题。"
    )
    print()
    print(
        f"这些错题里冠军自己：做对 **{diag['champion_right']}** / 做错 "
        f"{diag['champion_wrong']} / 没买 {diag['champion_silent']}。"
    )
    print()

    # 「每题有几个不同的正确身份」：>1 说明这题有多个可接受答案，那么
    # 「票数前 2 名里有一个正确」更容易达成，oracle@top-2 也就更乐观。
    multi = Counter(len(keys) for keys in good.values() if keys)
    print("## 有几种「正确答案」（只统计至少被买对一次的题）")
    print()
    print("| 不同的正确身份数 | 题数 |")
    print("|---|---:|")
    for count in sorted(multi):
        print(f"| {count} | {multi[count]} |")
    print()
    print(f"（并集 {sum(multi.values())} 题；只有唯一正确答案的 {multi.get(1, 0)} 题）")
    print()

    best_rule = max(
        plain,
        tie_broken,
        oracle_top_k(votes_by_task, good, 2),
        champion_weighted(votes_by_task, champion_choice, good, weight=2),
    )
    print("## 判读")
    print()
    if tie_broken >= target:
        print(f"**已够**：平票兜底规则 {tie_broken}/{n} 达到目标。")
    elif oracle_top_k(votes_by_task, good, 2) >= target:
        print(
            "**缺的是选择规则，不是模型**：只看票数前 2 名、由裁判挑就能到 "
            f"{oracle_top_k(votes_by_task, good, 2)}/{n} ≥ {target}。"
        )
        print("⇒ 值得在聚合层做改进（更好的排序/校验信号），不用重训。")
    else:
        print(
            f"**改聚合规则没用**：连「票数前 2 名里挑」的上界都只有 "
            f"{oracle_top_k(votes_by_task, good, 2)}/{n}，低于目标 {target}。"
        )
        print("⇒ 正确答案压根没进前 2 名，必须让模型自己多对几道（新信息），不是调规则。")
    print()
    print(f"（现有最好规则 {best_rule}/{n} = {best_rule / n:.1%}）")

    if args.out:
        payload = {
            "n": n,
            "target": target,
            "champion": args.champion,
            "strict": strict,
            "rules": {name: ok for name, ok, _ in rows},
            "diagnosis": diag,
            "good_key_counts": {str(k): v for k, v in sorted(multi.items())},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print()
        print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
