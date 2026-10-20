#!/usr/bin/env python3
"""M10-R：把「答案不在第 1 名」的增量并进 SFT 池，并按名次重新配比。

为什么要按名次配比
------------------
2026-10-20 实测（`plans/m10-step2-closing.md`）：

* 现有 SFT 池 1,252 行里，**买到的商品有 68.0% 就是搜索第 1 名**；
* 评测 200 题按 gold 名次切开：**第 1 名对 → 86.8%，不对 → 34.0%**；
* ⇒ 模型学到的是「开第一个」，不是「核对后挑对」。

`scripts/filter_m10_rank2.py` 已经筛出 383 条「gold 名次 ≥2 且老师买对」的轨迹
（`outputs/m10-rank2/kept.jsonl`）。本脚本把它并进池子，并给出配比报告。

两个旋钮
--------
1. **并入增量**（默认全并）：把「答案不在第 1 名」的样本从 401 条加到 784 条，
   占比 32.0% → 47.9%。
2. **降采样第 1 名**（`--target-share`，默认不启用）：如果想推到 60%，
   就按公式 ``r1 = rank_ge2 × (1-share) / share`` 随机丢掉一部分第 1 名样本。
   丢的是正确样本，只是让它们不再占多数；用固定 seed，可复现。

**默认不降采样**的理由：评测集里 53% 的题 gold 就在第 1 名，训练集 47.9% 与它几乎
相等；推到 60% 会低于部署分布，可能反过来伤那 106 道简单题。先跑「不降采样」这一臂。

产出的三件东西
--------------
* ``<out-dir>/all.jsonl`` —— 合并后的 SFT 池（4 键 schema）；
* ``<out-dir>/difficulty_labels.jsonl`` —— **只保留池内已有行**的难度标签，
  新题留给 ``label_sft_difficulty_offline.py`` 补（它只标缺的 id）；
* ``<out-dir>/build_report.json`` —— 配比、名次分布、被丢弃的行、契约校验结果。

用法::

    .venv/Scripts/python.exe scripts/build_m10r_pool.py \\
        --increment outputs/m10-rank2/kept.jsonl --out-dir data/sft_m10r

    ... --target-share 0.60      # 强干预臂：把第 1 名降到 40%
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_sft_curriculum import _validate_row  # noqa: E402

SFT = ROOT / "data" / "sft_pure_v4"
SEARCH_ROW = re.compile(r"^(?P<rank>\d+)\|(?P<asin>[^|\n]+)\|", re.MULTILINE)
SEARCH_PAGE_MARKER = "page_type: search_results"
#: 只有 `next_page` 算「翻页示范」。
#: ⚠️ `prev_page` 名字里也有 page，但实测 2,032 次几乎全是「从详情页/信息子页返回列表」
#: （`before_page_type=product_detail|information_subpage`），不是翻搜索结果页。
#: 把它算进来会把翻页示范从 14 条虚报成 1,089 条。
PAGING_TOOLS = frozenset({"next_page"})
SFT_ROW_KEYS = {"trajectory_id", "task_id", "messages", "tools"}
REPORT_SCHEMA = "shopping-m10r-pool-v1"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def portable(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def sha256_text(path: Path) -> str:
    """文本哈希按 LF 归一（仓库约定，避免 Windows autocrlf 破坏 SHA）。"""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def tool_calls(messages: list[dict]) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []
    for message in messages or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError):
                arguments = {}
            calls.append(
                (
                    str(function.get("name") or ""),
                    arguments if isinstance(arguments, dict) else {},
                )
            )
    return calls


def seen_ranks(messages: list[dict]) -> dict[str, int]:
    """老师**自己看到过**的每个 ASIN 的最好名次（跨所有搜索页取 min）。"""
    ranks: dict[str, int] = {}
    for message in messages or []:
        if message.get("role") != "tool":
            continue
        text = str(message.get("content") or "")
        if SEARCH_PAGE_MARKER not in text:
            continue
        for match in SEARCH_ROW.finditer(text):
            asin = match.group("asin")
            rank = int(match.group("rank"))
            ranks[asin] = min(ranks.get(asin, 10**9), rank)
    return ranks


def purchased_asin(messages: list[dict]) -> str | None:
    """`buy_now` 不带参数，只能买当前打开的商品 ⇒ 买的是它前面最后一个 open_product。"""
    calls = tool_calls(messages)
    buy_index = max((i for i, (name, _) in enumerate(calls) if name == "buy_now"), default=-1)
    for index in range(buy_index - 1, -1, -1):
        name, arguments = calls[index]
        if name == "open_product" and arguments.get("asin"):
            return str(arguments["asin"])
    return None


def opened_asins(messages: list[dict]) -> list[str]:
    return [
        str(arguments["asin"])
        for name, arguments in tool_calls(messages)
        if name == "open_product" and arguments.get("asin")
    ]


def describe_row(row: dict) -> dict:
    """一条 SFT 行的名次画像：买到谁、名次多少、开过几个商品、有没有翻页。"""
    messages = row.get("messages") or []
    calls = tool_calls(messages)
    bought = purchased_asin(messages)
    ranks = seen_ranks(messages)
    opened = opened_asins(messages)
    return {
        "task_id": int(row["task_id"]),
        "bought": bought,
        "rank": ranks.get(bought) if bought else None,
        "distinct_opens": len(set(opened)),
        "steps": len(calls),
        # A 桶（gold 在第 21 名以后）只能靠翻页救；池子里有多少条翻页示范必须单列，
        # 否则会被「名次≥2」这个大数字盖住，误以为已经覆盖。
        "paging": any(name in PAGING_TOOLS for name, _ in calls),
    }


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "未出现"
    if rank == 1:
        return "1"
    if rank <= 3:
        return "2-3"
    if rank <= 20:
        return "4-20"
    return ">20"


def is_rank1(profile: dict) -> bool:
    return profile["rank"] == 1


def plan_mix(rank_ge2: int, rank1: int, target_share: float | None) -> dict:
    """算出第 1 名样本要留几条。``target_share=None`` 表示一条不丢。"""
    if target_share is None:
        return {"rank_ge2": rank_ge2, "rank1_kept": rank1, "rank1_dropped": 0,
                "achieved_share": rank_ge2 / (rank_ge2 + rank1) if rank_ge2 + rank1 else 0.0}
    if not 0.0 < target_share < 1.0:
        raise ValueError("target_share 必须在 (0, 1) 之间")
    allowed = int(round(rank_ge2 * (1.0 - target_share) / target_share))
    allowed = max(0, min(allowed, rank1))
    total = rank_ge2 + allowed
    return {
        "rank_ge2": rank_ge2,
        "rank1_kept": allowed,
        "rank1_dropped": rank1 - allowed,
        "achieved_share": (rank_ge2 / total) if total else 0.0,
    }


def downsample_rank1(
    profiles: list[dict], keep: int, seed: int
) -> set[int]:
    """返回要**丢掉**的第 1 名行的下标（固定 seed，可复现）。"""
    rank1_indexes = [i for i, p in enumerate(profiles) if is_rank1(p)]
    if keep >= len(rank1_indexes):
        return set()
    shuffled = list(rank1_indexes)
    random.Random(seed).shuffle(shuffled)
    return set(shuffled[keep:])


def check_pool_contract(rows: list[dict], label: str) -> list[str]:
    problems: list[str] = []
    for row in rows:
        if set(row) != SFT_ROW_KEYS:
            problems.append(f"{label} task {row.get('task_id')}: keys {sorted(row)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pool", type=Path, default=SFT / "all.jsonl")
    parser.add_argument("--increment", type=Path, default=ROOT / "outputs/m10-rank2/kept.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "sft_m10r")
    parser.add_argument(
        "--labels",
        type=Path,
        default=SFT / "difficulty_labels.jsonl",
        help="池子现有的难度标签；只保留池内还存在的行，新题留给离线标注脚本",
    )
    parser.add_argument("--evaluation", type=Path, default=ROOT / "data/evaluation/tasks.jsonl")
    parser.add_argument(
        "--target-share",
        type=float,
        default=None,
        help="「答案不在第 1 名」的目标占比；不给则保留全部第 1 名样本（推荐）",
    )
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args(argv)

    pool_rows = read_jsonl(args.pool)
    increment_rows = read_jsonl(args.increment)
    evaluation_ids = {int(row["task_id"]) for row in read_jsonl(args.evaluation)}

    problems = check_pool_contract(pool_rows, "pool") + check_pool_contract(
        increment_rows, "increment"
    )
    if problems:
        raise SystemExit("schema 不符合 4 键契约：\n  " + "\n  ".join(problems[:10]))

    # 契约校验：增量行必须能过课程清单那一关，否则训练期会炸
    admitted: list[dict] = []
    rejected: list[dict] = []
    for row in increment_rows:
        try:
            _validate_row(row)
        except Exception as exc:  # noqa: BLE001 - 记录下来，不吞
            rejected.append({"task_id": int(row["task_id"]), "error": str(exc)})
            continue
        admitted.append(row)

    pool_ids = {int(row["task_id"]) for row in pool_rows}
    admitted_ids = [int(row["task_id"]) for row in admitted]
    if len(set(admitted_ids)) != len(admitted_ids):
        raise SystemExit("增量里 task_id 有重复")
    twin = sorted(set(admitted_ids) & pool_ids)
    if twin:
        raise SystemExit(f"增量与池子 task_id 重叠：{twin[:10]}")
    leak = sorted((set(admitted_ids) | pool_ids) & evaluation_ids)
    if leak:
        raise SystemExit(f"训练池与评测集重叠：{leak[:10]}")

    tools_signature = {
        hashlib.sha256(
            json.dumps(row["tools"], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        for row in pool_rows + admitted
    }
    if len(tools_signature) != 1:
        raise SystemExit(f"出现 {len(tools_signature)} 种 tools schema，期望 1 种")

    profiles = [describe_row(row) for row in pool_rows]
    increment_profiles = [describe_row(row) for row in admitted]
    if any(p["rank"] is not None and p["rank"] < 2 for p in increment_profiles):
        bad = [p["task_id"] for p in increment_profiles if p["rank"] is not None and p["rank"] < 2]
        raise SystemExit(f"增量里混进了第 1 名样本：{bad[:10]}")

    rank_ge2_base = sum(1 for p in profiles if p["rank"] is not None and p["rank"] >= 2)
    rank1_base = sum(1 for p in profiles if is_rank1(p))
    mix = plan_mix(rank_ge2_base + len(increment_profiles), rank1_base, args.target_share)

    dropped = downsample_rank1(profiles, mix["rank1_kept"], args.seed)
    kept_pool = [row for index, row in enumerate(pool_rows) if index not in dropped]
    merged = sorted(kept_pool + admitted, key=lambda row: int(row["task_id"]))
    merged_ids = {int(row["task_id"]) for row in merged}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool_out = args.out_dir / "all.jsonl"
    labels_out = args.out_dir / "difficulty_labels.jsonl"
    write_jsonl(pool_out, merged)

    existing_labels = read_jsonl(args.labels) if args.labels.is_file() else []
    kept_labels = [row for row in existing_labels if int(row["task_id"]) in merged_ids]
    write_jsonl(labels_out, kept_labels)
    labelled_ids = {int(row["task_id"]) for row in kept_labels}
    pending_labels = sorted(merged_ids - labelled_ids)

    final_profiles = [describe_row(row) for row in merged]
    buckets = collections.Counter(rank_bucket(p["rank"]) for p in final_profiles)
    opens = collections.Counter(p["distinct_opens"] for p in final_profiles)
    steps = [p["steps"] for p in final_profiles]

    report = {
        "schema_version": REPORT_SCHEMA,
        "inputs": {
            "pool": portable(args.pool),
            "increment": portable(args.increment),
            "labels": portable(args.labels),
            "evaluation": portable(args.evaluation),
        },
        "outputs": {"pool": portable(pool_out), "labels": portable(labels_out)},
        "counts": {
            "pool_rows": len(pool_rows),
            "increment_rows": len(increment_rows),
            "increment_rejected_by_contract": len(rejected),
            "increment_admitted": len(admitted),
            "rank1_dropped": len(dropped),
            "merged_rows": len(merged),
            "labels_carried_over": len(kept_labels),
            "labels_pending": len(pending_labels),
        },
        "mix": mix,
        "rank_distribution": dict(sorted(buckets.items())),
        "rank_ge2_rows": sum(1 for p in final_profiles if p["rank"] and p["rank"] >= 2),
        "comparison_rows": sum(
            1 for p in final_profiles if p["distinct_opens"] >= 2 and p["rank"] and p["rank"] >= 2
        ),
        "paging_rows": sum(1 for p in final_profiles if p["paging"]),
        "paging_rows_in_increment": sum(1 for p in increment_profiles if p["paging"]),
        "beyond_first_page_rows": sum(
            1 for p in final_profiles if p["rank"] is not None and p["rank"] > 20
        ),
        "distinct_opens_distribution": {str(k): v for k, v in sorted(opens.items())},
        "mean_steps": sum(steps) / len(steps) if steps else 0.0,
        "increment_rejected": rejected,
        "sha256": {
            "pool": sha256_text(pool_out),
            "labels": sha256_text(labels_out),
        },
    }
    (args.out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print(
        f"池子 {len(pool_rows)} 行 + 增量 {len(increment_rows)} 条"
        f"（契约拒绝 {len(rejected)} 条）→ 合并 {len(merged)} 行"
    )
    if dropped:
        print(f"  按目标占比 {args.target_share:.0%} 丢掉第 1 名样本 {len(dropped)} 行")
    print(
        f"  「答案不在第 1 名」占比：{mix['achieved_share']:.1%}"
        f"（{mix['rank_ge2']} / {mix['rank_ge2'] + mix['rank1_kept']}）"
    )
    print(f"  其中「开≥2 且名次≥2」的对照样本：{report['comparison_rows']} 条")
    print(
        f"  翻页示范：{report['paging_rows']} 条"
        f"（新增贡献 {report['paging_rows_in_increment']} 条）；"
        f"gold 在第 21 名以后：{report['beyond_first_page_rows']} 条"
    )
    print("  名次分布：" + "，".join(f"{k} {v}" for k, v in report["rank_distribution"].items()))
    print(f"  平均步数 {report['mean_steps']:.1f}")
    print(f"  待标注难度标签：{len(pending_labels)} 条")
    print(f"写入 {portable(pool_out)}；报告 {portable(args.out_dir / 'build_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
