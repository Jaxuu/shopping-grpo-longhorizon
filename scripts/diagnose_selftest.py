#!/usr/bin/env python3
"""把「训练集自测 59.7% < 考卷 63.5%」拆开：是池子更难，还是真没学会。

为什么不能直接照抄分叉
----------------------
预登记的分叉是「训练集 ≥90% ⇒ 泛化问题 / ≤70% ⇒ 没学会」。实测 59.7% 落在
「没学会」里，但那个阈值隐含一个前提：**训练池与考卷难度同构**。这个前提没验过。

而训练池恰恰不是随机样本：M10 的选题规则明确挑硬题（gold 名次 21–60 要翻页、
名次 2–5 且第 1 名是竞品、第 1 名超预算、规格轴候选 ≥3）。考卷 Final-200 是
策展过的。两边难度不同构时，直接比总分会把「池子难」误判成「没学会」。

两个都不改判据的拆法
--------------------
1. **按 gold 检索名次分桶**：同一桶内比做对率（同难度才可比）；
2. **直接标准化**：把考卷每个桶的做对率按自测的桶权重重新加权，回答
   「如果考卷也这么难，它该考多少分」。反过来也做一次。

结论只可能是三种之一，脚本会自己判：
* 对齐后自测 ≈ 考卷（差 <5pt）⇒ 池子难度解释不了，**真没学会**；
* 对齐后自测明显高于考卷（≥5pt）⇒ **池子更难**，总分不可比，该按桶看；
* 失败以 `repeat_loop` / `max_steps` 为主 ⇒ 处方是**治循环**，不是加量。

用法::

    .venv/bin/python scripts/diagnose_selftest.py \\
        --selftest-run outputs/evaluation/selftest-300 \\
        --exam-run outputs/evaluation/m10r-v2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
if str(SHOP_ENV) not in sys.path:
    sys.path.append(str(SHOP_ENV))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load("build_probe_sets", ROOT / "scripts" / "build_probe_sets.py")

BUCKETS = ("1-5", "6-20", "21-60", ">60/搜不到")
BEHAVIOUR_TYPES = ("repeat_loop", "max_steps", "wrong_purchase", "partial_alternative_purchase")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_run(run_dir: Path) -> dict[int, dict]:
    """task_id -> {success, reward_type}。"""
    table: dict[int, dict] = {}
    for row in read_jsonl(Path(run_dir) / "trajectories.jsonl"):
        task_id = row.get("task_id")
        if task_id is None:
            continue
        detail = ((row.get("terminal_result") or {}).get("reward_detail")) or {}
        table[int(task_id)] = {
            "success": detail.get("purchase_success") is True
            or detail.get("reward_type") == "gold_purchase",
            "reward_type": detail.get("reward_type", "unknown"),
        }
    return table


def bucket_of(rank: int | None) -> str:
    if rank is None:
        return ">60/搜不到"
    if rank <= 5:
        return "1-5"
    if rank <= 20:
        return "6-20"
    if rank <= 60:
        return "21-60"
    return ">60/搜不到"


def bucket_stats(run: Mapping[int, dict], ranks: Mapping[int, str]) -> dict[str, dict]:
    stats: dict[str, dict] = {name: {"n": 0, "ok": 0} for name in BUCKETS}
    for task_id, record in run.items():
        name = ranks.get(task_id, ">60/搜不到")
        stats[name]["n"] += 1
        stats[name]["ok"] += 1 if record["success"] else 0
    for entry in stats.values():
        entry["rate"] = entry["ok"] / entry["n"] if entry["n"] else None
    return stats


def standardize(source: Mapping[str, dict], weights_from: Mapping[str, dict]) -> float | None:
    """用 weights_from 的桶权重，给 source 的每桶做对率加权（直接标准化）。"""
    total = sum(entry["n"] for entry in weights_from.values())
    if not total:
        return None
    weighted = 0.0
    used = 0
    for name, weight in weights_from.items():
        if not weight["n"]:
            continue
        rate = source.get(name, {}).get("rate")
        if rate is None:
            continue
        weighted += rate * weight["n"]
        used += weight["n"]
    return weighted / used if used else None


def fmt_rate(ok: int, n: int) -> str:
    return f"{ok}/{n} = {ok / n:.1%}" if n else "—"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest-run", type=Path, required=True)
    parser.add_argument("--exam-run", type=Path, required=True)
    parser.add_argument("--products", type=Path, default=probe.audit.DEFAULT_PRODUCTS)
    parser.add_argument("--index", type=Path, default=probe.DEFAULT_INDEX)
    args = parser.parse_args(argv)

    goals = probe.audit.build_goals(probe.audit.load_products(args.products))
    searcher = probe.load_searcher(args.index)

    selftest = load_run(args.selftest_run)
    exam = load_run(args.exam_run)
    if not selftest or not exam:
        raise SystemExit("轨迹为空：先确认 --selftest-run / --exam-run 指对了目录")

    rank_cache: dict[int, str] = {}

    def ranks_of(run: Mapping[int, dict]) -> dict[int, str]:
        for task_id in run:
            if task_id not in rank_cache:
                rank_cache[task_id] = bucket_of(probe.gold_rank(searcher, goals[task_id]))
        return {task_id: rank_cache[task_id] for task_id in run}

    selftest_ranks = ranks_of(selftest)
    exam_ranks = ranks_of(exam)

    st_stats = bucket_stats(selftest, selftest_ranks)
    ex_stats = bucket_stats(exam, exam_ranks)

    st_ok = sum(1 for r in selftest.values() if r["success"])
    ex_ok = sum(1 for r in exam.values() if r["success"])

    print("# 训练集自测 vs 考卷：难度对齐后再比")
    print()
    print(f"- 训练集自测 {fmt_rate(st_ok, len(selftest))}（{args.selftest_run}）")
    print(f"- Final-200  {fmt_rate(ex_ok, len(exam))}（{args.exam_run}）")
    print(f"- 原始差值 **{st_ok / len(selftest) - ex_ok / len(exam):+.1%}**")
    print()

    print("## 1. 两边的难度构成（gold 检索名次）")
    print()
    print("| 名次桶 | 自测题数 | 自测做对率 | 考卷题数 | 考卷做对率 |")
    print("|---|---:|---:|---:|---:|")
    for name in BUCKETS:
        a, b = st_stats[name], ex_stats[name]
        print(
            f"| {name} | {a['n']} | {fmt_rate(a['ok'], a['n'])} | "
            f"{b['n']} | {fmt_rate(b['ok'], b['n'])} |"
        )
    print()

    st_on_exam = standardize(st_stats, ex_stats)
    ex_on_selftest = standardize(ex_stats, st_stats)
    print("## 2. 直接标准化（不改判据，只把难度对齐）")
    print()
    if ex_on_selftest is not None:
        print(f"- 把**考卷**的每桶做对率按**自测的难度构成**加权：**{ex_on_selftest:.1%}**")
    if st_on_exam is not None:
        print(f"- 把**自测**的每桶做对率按**考卷的难度构成**加权：**{st_on_exam:.1%}**")
    print()

    print("## 3. 自测失败的那些题长什么样")
    print()
    failures = Counter(
        record["reward_type"] for record in selftest.values() if not record["success"]
    )
    print("| reward_type | 道数 |")
    print("|---|---:|")
    for name, count in failures.most_common():
        print(f"| `{name}` | {count} |")
    print()

    print("## 4. 判定")
    print()
    aligned = ex_on_selftest if ex_on_selftest is not None else ex_ok / len(exam)
    gap = st_ok / len(selftest) - aligned
    behaviour = sum(failures[name] for name in ("repeat_loop", "max_steps"))
    n_fail = sum(failures.values())
    if gap >= 0.05:
        print(
            f"**池子更难**：按同一难度构成对齐后，自测 {st_ok / len(selftest):.1%} 对考卷 {aligned:.1%}，"
            f"自测反而高 {gap:.1%}。⇒ 两个总分不可比，「没学会」这个结论**不成立**，"
            "该按桶看，M2 不能简单加量。"
        )
    elif behaviour and n_fail and behaviour / n_fail >= 0.4:
        print(
            f"**行为问题**：{n_fail} 道失败里 {behaviour} 道是 `repeat_loop`/`max_steps`"
            f"（{behaviour / n_fail:.0%}）。⇒ 处方是**治循环/省步数**，不是加量。"
        )
    else:
        print(
            f"**真没学会**：对齐后自测 {st_ok / len(selftest):.1%} 对考卷 {aligned:.1%}，"
            f"差 {gap:+.1%}，池子难度解释不了。⇒ 按预登记走「加量/加 epoch」。"
        )
    print()
    print("> 提醒：这里只改「怎么比」，不改阈值。分叉规则仍是 ≥90%/≤70%，一个数没动。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
