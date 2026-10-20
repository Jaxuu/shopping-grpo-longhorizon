#!/usr/bin/env python3
"""M10 Step 1：从**未使用的商品**里挑采集目标，保证老师「必须」做出目标动作。

为什么不用 prompt 逼
--------------------
老师的动作分布由**题目难度**决定，不由指令决定。池里那 11 条翻页轨迹全是硬题
（如 task 2223 搜 10 次、24 步）。所以选题要靠「离线可判的检索名次」，
而不是在 prompt 里写「请多翻页」。

四类目标（互斥，优先级 S1 > S2 > S3 > S4）
------------------------------------------
* **S1 翻页型**：gold 在 BM25 150 召回内，但名次 **21–60**（翻 1–2 页可达）。
  对应评测失败 A 桶（gold 没进结果页）。
* **S1b 换词型**：gold 名次 **>60 或不在召回内**。只能靠改查询，采集价值较低（老师可能直接放弃），
  单独统计、默认不入选。
* **S2 候选筛选**：gold 名次 **2–5**，且第 1 名是同 category 的另一个 ASIN。
  对应评测失败 B 桶（进了结果页却没打开）。
* **S3 超预算排除**：gold 名次 ≤20，但**第 1 名价格超预算**而 gold 在预算内。
  对应预算硬门 20 条失败。
* **S4 规格难例**：某规格轴候选 ≥3（近似变体易挑错）。对应 D 桶 18 条。

task_id 的口径
--------------
`task_id` 就是语料（`fine_items_eval_train_all.json.gz`）的**行号**，已用 197/197 条评测轨迹
交叉校验。语料每个商品恰好 1 条 instruction。

用法::

    .venv/Scripts/python.exe scripts/select_m10_collection_targets.py --limit 500
    .venv/Scripts/python.exe scripts/select_m10_collection_targets.py --all
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
CORPUS = SHOP_ENV / "data" / "fine_items_eval_train_all.json.gz"
DEFAULT_INDEX = ROOT / ".workbuddy-ai" / "cache" / "products_local.sqlite3"

USED_SOURCES = (
    ROOT / "data" / "evaluation" / "tasks.jsonl",
    ROOT / "data" / "sft_pure_v4" / "all.jsonl",
    ROOT / "data" / "grpo" / "train.jsonl",
    ROOT / "data" / "grpo" / "validation.jsonl",
)

# 观察只渲染前 20 名；`next_page` 一次翻到 40，两次到 60。
PAGE_SIZE = 20
FLIP_MAX_RANK = 60
RECALL_DEPTH = 150


def load_used_ids() -> set[int]:
    used: set[int] = set()
    for path in USED_SOURCES:
        if not path.is_file():
            print(f"  [warn] 缺少 {path}", file=sys.stderr)
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    used.add(int(payload["task_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
    return used


def load_searcher(index_path: Path):
    """只在这里碰 sys.path：顶层污染会破坏 `import scripts.<name>` 的解析。"""
    shop_env = str(SHOP_ENV)
    if shop_env not in sys.path:
        sys.path.append(shop_env)
    from web_agent_site.engine.search import MultiFieldBM25Searcher, build_index

    if not index_path.is_file():
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(CORPUS, "rt", encoding="utf-8") as handle:
            products = json.load(handle)
        build_index(iter(products), index_path, product_data_sha256="local-audit")
    return MultiFieldBM25Searcher(index_path)


def _clean_attributes(instruction: dict) -> list[str]:
    """丢掉脏属性（空、单字符、纯符号），否则会造出 `场效应管 -` 这种查询。"""
    out: list[str] = []
    for raw in instruction.get("attributes") or []:
        text = str(raw).strip()
        if len(text) < 2:
            continue
        if not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in text):
            continue
        out.append(text)
    return out


def build_queries(item: dict, instruction: dict) -> list[str]:
    """造三个「忠实」查询，模拟模型可能写出的三种查询。

    1. 类目叶子（最准，模型若知道类目会这么写）
    2. 用户明确说过的属性（次准）
    3. **指令开头 20 字**（最接近模型真实写法：它会从用户原话里抄词）

    不做任何「知道答案」的取巧（例如直接用商品标题），否则名次会虚高。
    三个查询取**最好名次**，即对模型最有利的口径 ⇒ 选出的题一定真的难。
    """
    queries: list[str] = []
    category = str(item.get("category") or "")
    leaf = category.split("›")[-1].strip()
    if leaf:
        queries.append(leaf)
    attributes = _clean_attributes(instruction)
    if attributes:
        queries.append(" ".join(attributes[:4]))
    text = str(instruction.get("instruction") or "")
    if text:
        head = "".join(ch for ch in text[:24] if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        if len(head) >= 4:
            queries.append(head)
    return [q for q in dict.fromkeys(queries) if q]


def best_rank(searcher, queries: list[str], asin: str) -> int | None:
    """gold 在这些查询下的最好名次（对模型最有利的口径）。"""
    best: int | None = None
    for query in queries:
        try:
            hits = searcher.search(query, k=RECALL_DEPTH)
        except Exception:  # 索引对空 query 会抛，跳过即可
            continue
        for hit in hits:
            if str(hit.asin) == str(asin):
                if best is None or hit.rank < best:
                    best = int(hit.rank)
                break
    return best


def top_hit(searcher, queries: list[str]):
    """第 1 个（最宽泛的）查询的第 1 名——模拟「模型看一眼列表顶部看到什么」。"""
    for query in queries:
        try:
            hits = searcher.search(query, k=RECALL_DEPTH)
        except Exception:
            continue
        if hits:
            return hits[0]
    return None


def spec_axes(item: dict) -> dict[str, int]:
    """每个规格轴的候选数（近似变体越多的轴，越容易挑错）。"""
    axes: dict[str, int] = {}
    raw = item.get("customization_options") or {}
    if not isinstance(raw, dict):
        return axes
    for axis, entries in raw.items():
        if not isinstance(entries, list):
            continue
        axes[str(axis)] = len([e for e in entries if isinstance(e, dict) and e.get("value")])
    return axes


def has_near_variant(item: dict, *, min_candidates: int = 3, threshold: float = 0.8) -> bool:
    """同轴里是否存在「候选 ≥3 且有两个值相似度 ≥0.8」的近似变体。

    只看候选数会把「颜色有 20 种」也算成难例，那样 42% 的题都会被判进来，
    等于没有区分度。必须要求**存在长得像的两个值**——那才是挑错的来源。
    """
    from difflib import SequenceMatcher

    raw = item.get("customization_options") or {}
    if not isinstance(raw, dict):
        return False
    for entries in raw.values():
        if not isinstance(entries, list):
            continue
        values = [str(e.get("value")).strip() for e in entries if isinstance(e, dict) and e.get("value")]
        if len(values) < min_candidates:
            continue
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if SequenceMatcher(None, values[i], values[j]).ratio() >= threshold:
                    return True
    return False


def classify(
    *,
    rank: int | None,
    top_asin: str | None,
    top_price: float | None,
    gold_price: float | None,
    budget: float | None,
    same_category_top1: bool,
    max_axis: int,
    near_variant: bool,
) -> str:
    if rank is None or rank > FLIP_MAX_RANK:
        return "S1b_change_query"
    if rank > PAGE_SIZE:
        return "S1_paging"
    if 2 <= rank <= 5 and top_asin and same_category_top1:
        return "S2_compare"
    if (
        budget is not None
        and top_price is not None
        and gold_price is not None
        and top_price > budget
        and gold_price <= budget
    ):
        return "S3_budget_exclude"
    if max_axis >= 3 and near_variant:
        return "S4_spec_hard"
    return "easy"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "collection" / "m10_targets.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "plans" / "m10-step1-report.md")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--limit", type=int, default=500, help="只扫前 N 个未使用商品（试跑）")
    parser.add_argument("--all", action="store_true", help="扫全部未使用商品")
    parser.add_argument("--per-tier", type=int, default=0, help="每类最多输出多少条（0=不限）")
    parser.add_argument(
        "--step2-out", type=Path, default=None,
        help="同时写出 Step 2 小样本验证清单（每类抽 N 题）",
    )
    parser.add_argument("--step2-per-tier", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args(argv)

    with gzip.open(CORPUS, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    used = load_used_ids()
    print(f"语料 {len(products)} 条；已用 task_id {len(used)} 个；剩余 {len(products) - len(used)} 条")

    sys.path.append(str(SHOP_ENV))
    from web_agent_site.engine.constraints import explicit_budget_from_instruction

    searcher = load_searcher(args.index)
    asin_to_index = {str(item.get("asin")): idx for idx, item in enumerate(products)}

    scanned = 0
    tiers: Counter = Counter()
    selected: list[dict] = []
    rank_hist: Counter = Counter()
    for task_id, item in enumerate(products):
        if task_id in used:
            continue
        instructions = item.get("instructions") or []
        if not instructions:
            continue
        instruction = instructions[0]
        if not (instruction.get("attributes") or []):
            continue
        if args.limit and not args.all and scanned >= args.limit:
            break
        scanned += 1

        asin = str(item.get("asin"))
        queries = build_queries(item, instruction)
        rank = best_rank(searcher, queries, asin)
        if rank is None:
            rank_hist["miss"] += 1
        elif rank <= 20:
            rank_hist["1-20"] += 1
        elif rank <= 60:
            rank_hist["21-60"] += 1
        elif rank <= 150:
            rank_hist["61-150"] += 1

        top = top_hit(searcher, queries)
        top_asin = str(top.asin) if top is not None else None
        top_item = products[asin_to_index[top_asin]] if top_asin in asin_to_index else None
        same_cat = bool(
            top_item and str(top_item.get("category")) == str(item.get("category"))
        )
        pricing = item.get("pricing") or []
        gold_price = float(pricing[0]) if pricing else None
        top_pricing = (top_item or {}).get("pricing") or []
        top_price = float(top_pricing[0]) if top_pricing else None
        budget = explicit_budget_from_instruction(str(instruction.get("instruction") or ""))
        axes = spec_axes(item)
        max_axis = max(axes.values()) if axes else 0
        near_variant = max_axis >= 3 and has_near_variant(item)

        tier = classify(
            rank=rank,
            top_asin=top_asin,
            top_price=top_price,
            gold_price=gold_price,
            budget=budget,
            same_category_top1=same_cat,
            max_axis=max_axis,
            near_variant=near_variant,
        )
        tiers[tier] += 1
        if tier == "easy":
            continue
        selected.append(
            {
                "task_id": task_id,
                "tier": tier,
                "gold_rank": rank,
                "queries": queries,
                "top1_asin": top_asin,
                "top1_same_category": same_cat,
                "gold_price": gold_price,
                "top1_price": top_price,
                "budget": budget,
                "max_spec_axis": max_axis,
                "category": item.get("category"),
                "title": item.get("title"),
                "instruction": instruction.get("instruction"),
            }
        )

    # 按档位截断，保持类别均衡
    if args.per_tier:
        kept: list[dict] = []
        per: Counter = Counter()
        for row in selected:
            if per[row["tier"]] >= args.per_tier:
                continue
            per[row["tier"]] += 1
            kept.append(row)
        selected = kept

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = max(scanned, 1)

    if args.step2_out is not None:
        import random

        rng = random.Random(args.seed)
        picks: list[dict] = []
        # S1 优先取名次最靠后的（最难，最可能逼出翻页）；其余随机
        s1 = [r for r in selected if r["tier"] == "S1_paging"]
        s1.sort(key=lambda r: -(r.get("gold_rank") or 0))
        picks += s1[: args.step2_per_tier]
        for tier in ("S2_compare", "S3_budget_exclude"):
            pool = [r for r in selected if r["tier"] == tier]
            rng.shuffle(pool)
            picks += pool[: args.step2_per_tier]
        args.step2_out.parent.mkdir(parents=True, exist_ok=True)
        with args.step2_out.open("w", encoding="utf-8") as handle:
            for row in picks:
                handle.write(
                    json.dumps(
                        {"task_id": row["task_id"], "tier": row["tier"], "gold_rank": row.get("gold_rank")},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"Step 2 验证清单 {len(picks)} 题 → {args.step2_out}")
        for tier in ("S1_paging", "S2_compare", "S3_budget_exclude"):
            print(f"  {tier}: {sum(1 for r in picks if r['tier'] == tier)} 题")

    lines = [
        "# M10 Step 1 报告：选题与供给量",
        "",
        f"> 扫描未使用商品 **{scanned}** 条；已用 task_id **{len(used)}** 个（四池合并，零重叠）。",
        f"> 输出候选 **{len(selected)}** 条 → `{args.out}`",
        "",
        "## 1. gold 名次分布（对模型最有利的口径）",
        "",
        "| 名次 | 条数 | 占比 | 含义 |",
        "|---|---:|---:|---|",
    ]
    meaning = {
        "1-20": "第 1 页就能看到",
        "21-60": "需翻 1–2 页（**S1 翻页型**）",
        "61-150": "召回内但翻 3 页以上（**S1b 换词型**）",
        "miss": "150 召回内都没有（**S1b 换词型**）",
    }
    for key in ("1-20", "21-60", "61-150", "miss"):
        count = rank_hist.get(key, 0)
        lines.append(f"| {key} | {count} | {count / total:.1%} | {meaning[key]} |")
    lines += [
        "",
        "## 2. 分档结果",
        "",
        "| 档 | 含义 | 条数 | 占扫描 |",
        "|---|---|---:|---:|",
        f"| **S1_paging** | gold 名次 21–60，翻 1–2 页可达 | {tiers.get('S1_paging', 0)} | {tiers.get('S1_paging', 0) / total:.1%} |",
        f"| **S2_compare** | gold 名次 2–5，第 1 名是同类不同款 | {tiers.get('S2_compare', 0)} | {tiers.get('S2_compare', 0) / total:.1%} |",
        f"| **S3_budget_exclude** | 第 1 名超预算、gold 在预算内 | {tiers.get('S3_budget_exclude', 0)} | {tiers.get('S3_budget_exclude', 0) / total:.1%} |",
        f"| **S4_spec_hard** | 某规格轴候选 ≥3 | {tiers.get('S4_spec_hard', 0)} | {tiers.get('S4_spec_hard', 0) / total:.1%} |",
        f"| S1b_change_query | 名次 >60 或不在召回（采集价值低） | {tiers.get('S1b_change_query', 0)} | {tiers.get('S1b_change_query', 0) / total:.1%} |",
        f"| easy | 第 1 页靠前且无其他特征 | {tiers.get('easy', 0)} | {tiers.get('easy', 0) / total:.1%} |",
        "",
        "## 3. 供给量外推",
        "",
    ]
    remaining = len(products) - len(used)
    scale = remaining / total if total else 0
    for tier_name, need in (
        ("S1_paging", 289),
        ("S2_compare", 238),
        ("S3_budget_exclude", 200),
        ("S4_spec_hard", 0),
    ):
        hit = tiers.get(tier_name, 0)
        projected = int(hit * scale)
        verdict = "—" if need == 0 else ("**够**" if projected >= need * 1.5 else "**不够**")
        lines.append(
            f"- {tier_name}：本次命中 {hit} → 外推全量约 **{projected}** 条；目标缺口 {need} ⇒ {verdict}"
        )
    lines.append("")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n扫描 {scanned} 条，候选 {len(selected)} 条 → {args.out}")
    print(f"报告 → {args.report}")
    for name, count in tiers.most_common():
        print(f"  {name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
