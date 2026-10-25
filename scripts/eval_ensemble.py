#!/usr/bin/env python3
"""把已有的多个评测轨迹做**投票集成**，看能不能把 63.5% 抬到 70%。

为什么这条不算作弊
------------------
投票（self-consistency / ensemble）是推理侧的聚合，不改模型、不看 gold、不调参：
每个模型独立跑完，取它们买到的**（商品 + 规格）身份**里出现次数最多的那个。规则只有
两个「多数票、平票算失败」和「平票用最好的单模型兜底」，都在 dev 上定死，再在
Final-200 上跑一次。

为什么它可能有效
----------------
错答案是**分散**的（每个模型错在不同的 ASIN 上），正确答案只有一个。所以即使
每个模型只有 63.5%，只要「做对的模型买的是同一个 ASIN」而「做错的各错各的」，
多数票就能把 gold 顶出来。这就是 10 个模型并集 148/200（74%）而单模型 63.5%
的原因——问题是并集需要一个裁判，而多数票不需要。

天花板
------
任何集成的上限都是**并集**（148/200 = 74%）。所以 70%（140 题）要求
「148 题里 95% 的多数票判对」。这个数能不能达到，只能用现有轨迹量。

用法::

    .venv/bin/python scripts/eval_ensemble.py outputs/evaluation/*/
    .venv/bin/python scripts/eval_ensemble.py outputs/evaluation/*/ --min-tasks 100
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

DEFAULT_REFERENCE = ROOT / "data" / "evaluation" / "tasks.jsonl"
TARGET_RATE = 0.70


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load("build_probe_sets", ROOT / "scripts" / "build_probe_sets.py")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_runs(paths: Sequence[Path], *, min_tasks: int) -> dict[str, list[dict]]:
    """把目录或 trajectories.jsonl 展开成 {标签: 轨迹行}；题数不足的跳过（冒烟目录）。"""
    runs: dict[str, list[dict]] = {}
    for path in paths:
        path = Path(path)
        file = path / "trajectories.jsonl" if path.is_dir() else path
        if not file.is_file():
            continue
        rows = read_jsonl(file)
        if len(rows) < min_tasks:
            continue
        runs[file.parent.name] = rows
    return runs


def purchased(record: Mapping) -> str | None:
    """完整购买身份：ASIN + 规格。**不能只看 ASIN**——买对商品选错规格是 partial，不是成功。"""
    purchase = (record.get("terminal_result") or {}).get("purchase") or {}
    asin = purchase.get("asin")
    if not asin:
        return None
    options = purchase.get("options") or {}
    if not isinstance(options, Mapping):
        return str(asin)
    suffix = "&".join(f"{key}={value}" for key, value in sorted(options.items()))
    return f"{asin}|{suffix}" if suffix else str(asin)


def is_success(record: Mapping, *, strict: bool = True) -> bool:
    """是否算做对。

    默认 **strict**：只认 `reward_type == "gold_purchase"`，与项目约定「以后统一用
    strict/gold 口径」一致。`purchase_success` 在 `valid_alternative_purchase` 上
    也是 True（历史上有过 179 vs 178 的差，见 2026-10-26 记录），用它会把「买到了
    另一个也算对的东西」算成 gold，口径就飘了。`strict=False` 时才退回旧行为。
    """
    detail = ((record.get("terminal_result") or {}).get("reward_detail")) or {}
    if detail.get("reward_type") == "gold_purchase":
        return True
    return (not strict) and detail.get("purchase_success") is True


def plurality(votes: Mapping[str, int]) -> tuple[str | None, bool]:
    """出现次数最多的购买身份；平票返回 (None, True)。"""
    if not votes:
        return None, False
    ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, True
    return ranked[0][0], False


def vote_summary(
    votes_by_task: Mapping[int, Mapping[str, int]],
    good_keys_by_task: Mapping[int, set[str]],
    *,
    tie_break: Mapping[int, str] | None = None,
) -> dict[str, int]:
    """多数票命中统计。

    「命中」的判据是**这个购买身份曾被某个模型成功买过**（`good_keys`），不是
    「ASIN 等于 gold」。后者会把「买对商品、选错规格」算成对——本地冒烟时它
    让多数票算出 71.5%，比并集 68% 还高，物理上不可能，就是踩了这个坑。
    用 `good_keys` 判还保证命中 ⊆ 并集，结果不会自相矛盾。
    """
    hit = tie = hit_after_break = 0
    for task_id, votes in votes_by_task.items():
        good = good_keys_by_task.get(task_id) or set()
        winner, tied = plurality(votes)
        if tied:
            tie += 1
            fallback = (tie_break or {}).get(task_id)
            if fallback is not None and fallback in good:
                hit_after_break += 1
            continue
        if winner is not None and winner in good:
            hit += 1
            hit_after_break += 1
    return {"hit": hit, "tie": tie, "hit_after_break": hit_after_break, "n": len(votes_by_task)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="评测目录（或 trajectories.jsonl）")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--min-tasks", type=int, default=100, help="少于这么多题的目录当冒烟跳过")
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.8,
        help="与参照集 task_id 的重合率低于此值的目录直接剔除（防止把 dev-300 的分数混进 Final-200）",
    )
    parser.add_argument("--out", type=Path, default=None, help="把结果写成 json")
    parser.add_argument(
        "--loose",
        action="store_true",
        help="把 purchase_success 也算做对（valid_alternative 会被算成 gold，仅作对照）",
    )
    args = parser.parse_args(argv)
    strict = not args.loose

    reference_ids = probe.task_ids_of(args.reference)
    reference_set = set(reference_ids)

    runs = resolve_runs(args.runs, min_tasks=args.min_tasks)
    if not runs:
        raise SystemExit("没有可用的评测目录（题数都少于 --min-tasks？）")

    # 口径保护：四池零重叠，所以跑在 dev-300 / 自测卷上的目录对 Final-200 一票都投不出来。
    # 混进来只会让「模型准确率」那张表出现一堆不相干的数字，必须剔除而不是静默带过。
    dropped: list[tuple[str, float]] = []
    for label in list(runs):
        ids = {int(row["task_id"]) for row in runs[label] if row.get("task_id") is not None}
        overlap = len(ids & reference_set) / max(1, len(reference_set))
        if overlap < args.min_overlap:
            dropped.append((label, overlap))
            del runs[label]
    if not runs:
        raise SystemExit("剔除口径不符的目录后没有剩余目录。")
    if dropped:
        print("# 已剔除（与参照集重合率不足）")
        for label, overlap in dropped:
            print(f"- `{label}`：重合 {overlap:.0%}")
        print()

    print(
        f"# 投票集成：{len(runs)} 个评测目录，参照集 {len(reference_ids)} 题"
        f"（口径 {'strict/gold' if strict else 'loose/purchase_success'}）"
    )
    print()

    per_model: dict[str, dict[int, bool]] = {}
    loose_only: dict[str, int] = {}
    print("| 模型 | 题数 | 做对 | 准确率 | 宽松口径多算 |")
    print("|---|---:|---:|---:|---:|")
    for label, rows in sorted(runs.items()):
        table = {
            int(row["task_id"]): is_success(row, strict=strict)
            for row in rows
            if row.get("task_id") is not None
        }
        per_model[label] = table
        loose_only[label] = sum(
            1
            for row in rows
            if row.get("task_id") is not None
            and not is_success(row, strict=strict)
            and is_success(row, strict=False)
        )
        ok = sum(1 for value in table.values() if value)
        print(f"| `{label}` | {len(table)} | {ok} | {ok / len(table):.1%} | {loose_only[label]} |")
    print()

    tasks = [task_id for task_id in reference_ids]
    # 先把每个模型的行按 task_id 建索引，避免下面 O(题数 × 模型数 × 行数) 的线性查找。
    indexed = {
        label: {int(row["task_id"]): row for row in rows if row.get("task_id") is not None}
        for label, rows in runs.items()
    }
    votes_by_task: dict[int, Counter] = {}
    good_keys_by_task: dict[int, set[str]] = {}
    union_hit = 0
    for task_id in tasks:
        votes: Counter = Counter()
        good_keys: set[str] = set()
        for rows in indexed.values():
            record = rows.get(task_id)
            if record is None:
                continue
            key = purchased(record)
            if key is None:
                continue
            votes[key] += 1
            if is_success(record, strict=strict):
                good_keys.add(key)
        votes_by_task[task_id] = votes
        good_keys_by_task[task_id] = good_keys
        union_hit += 1 if good_keys else 0

    best_label = max(per_model, key=lambda label: sum(per_model[label].values()))
    best_ok = sum(1 for value in per_model[best_label].values() if value)

    plain = vote_summary(votes_by_task, good_keys_by_task)
    # 平票兜底：用最好的单模型的选择。
    # 注意这个「最好」是**在这批题上挑出来的**，属于在评测集上做选择，所以下面
    # 只把它当乐观上界报，主判据一律用 plain（平票算失败）。
    best_choice: dict[int, str] = {}
    for task_id in tasks:
        record = indexed[best_label].get(task_id)
        key = purchased(record) if record is not None else None
        if key:
            best_choice[task_id] = key
    broken = vote_summary(votes_by_task, good_keys_by_task, tie_break=best_choice)

    print("## 汇总")
    print()
    print("| 方案 | 做对 | 准确率 | 说明 |")
    print("|---|---:|---:|---|")
    print(f"| 最好的单模型 `{best_label}` | {best_ok} | {best_ok / len(tasks):.1%} | 现在的基线 |")
    print(
        f"| **多数票（平票算失败）** | {plain['hit']} | {plain['hit'] / len(tasks):.1%} | "
        f"平票 {plain['tie']} 题 —— **主判据** |"
    )
    print(
        f"| 多数票（平票用最好单模型兜底） | {broken['hit_after_break']} | "
        f"{broken['hit_after_break'] / len(tasks):.1%} | 乐观上界，兜底模型是在同一批题上挑的 |"
    )
    print(
        f"| 并集（谁能做对算谁，需要裁判） | {union_hit} | {union_hit / len(tasks):.1%} | 天花板 |"
    )
    print()

    target = round(len(tasks) * TARGET_RATE)
    print("## 离 70% 有多远")
    print()
    print(f"目标 {target}/{len(tasks)}（{TARGET_RATE:.0%}）")
    gap = target - plain["hit"]
    if gap <= 0:
        print(f"**已达线**：多数票 {plain['hit']}/{len(tasks)}，超过目标 {abs(gap)} 题。")
    else:
        print(
            f"**还差 {gap} 题**。并集 {union_hit} 是硬上限，离目标还有 {union_hit - target} 题余量。"
        )
        if union_hit < target:
            print()
            print("> **并集都不到目标** ⇒ 再加票数也没用，这条路的上限就在这里。")
        else:
            print()
            print(
                "> 补齐的办法：给冠军模型加温度采样（同一题跑 N 次），把票数从 "
                f"{len(runs)} 张加到 {len(runs)}+N 张。这是推理成本，不是训练。"
            )
    print()

    correct_counts: Counter = Counter()
    for task_id in tasks:
        correct_counts[sum(1 for rows in per_model.values() if rows.get(task_id))] += 1
    print("## 每题有几个模型做对")
    print()
    print("| 做对的模型数 | 题数 |")
    print("|---|---:|")
    for count in sorted(correct_counts):
        print(f"| {count} | {correct_counts[count]} |")
    print()
    print(
        f"（0 个：{correct_counts.get(0, 0)} 题，谁都做不对；"
        f"≥1 个：{sum(v for k, v in correct_counts.items() if k)} 题 = 并集）"
    )

    if args.out:
        payload = {
            "strict": strict,
            "loose_only": loose_only,
            "dropped": {label: round(overlap, 4) for label, overlap in dropped},
            "models": {
                label: {"n": len(table), "ok": sum(1 for v in table.values() if v)}
                for label, table in per_model.items()
            },
            "best_single": {"label": best_label, "ok": best_ok, "n": len(tasks)},
            "plurality": plain,
            "plurality_with_tiebreak": broken,
            "union": {"ok": union_hit, "n": len(tasks)},
            "target": {"ok": target, "rate": TARGET_RATE},
            "correct_count_histogram": {str(k): v for k, v in sorted(correct_counts.items())},
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
