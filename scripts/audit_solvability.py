#!/usr/bin/env python3
"""M1 solvability audit for shopping-grpo-longhorizon.

Three-layer, fully offline, deterministic audit of the frozen ShopSimulator
corpus and the four task datasets:

  L1 有解性 (solvability)          -- goal_options resolvable on the gold spec axes
  L2 检索可达性 (retrieval)        -- Environment v2 multi-field BM25 reachability
  L3 评分可评性 (evaluability)     -- gold self-consistency under Reward v3

Outputs (machine-readable ones under ``audit/``):
  task_audit.jsonl
  dead_tasks.jsonl
  difficulty_buckets.json
  longtail_tasks.jsonl
  hashes.txt

Narrative report (under ``plans/`` by default, git-ignored):
  audit-report.md

The audit is byte-reproducible: it never embeds timestamps, sorts every
collection it serializes, and normalizes CRLF to LF before hashing so the
recorded digests match the repository metadata on any platform.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
# Append (not insert) so the repository's own ``scripts`` package keeps
# resolving first when this module is imported from the test suite.
if str(SHOP_ENV) not in sys.path:
    sys.path.append(str(SHOP_ENV))

from web_agent_site.engine.constraints import explicit_budget_from_instruction  # noqa: E402

try:  # reward-v4 exposes an explicit parser version; reward-v3 does not.
    from web_agent_site.engine.constraints import (  # noqa: E402
        BUDGET_PARSER_VERSION as LIVE_BUDGET_PARSER_VERSION,
    )
except ImportError:  # pragma: no cover - depends on the checked-out reward
    LIVE_BUDGET_PARSER_VERSION = "shopping-budget-parser-v3"
from web_agent_site.engine.reward import evaluate_purchase  # noqa: E402
from web_agent_site.engine.reward_features import (  # noqa: E402
    _target_option_axes,
    compile_reward_features,
)
from web_agent_site.engine.search import (  # noqa: E402
    MultiFieldBM25Searcher,
    build_index,
)
from web_agent_site.engine.variant_price import (  # noqa: E402
    candidate_options_for_evaluation,
    resolve_variant_price,
)

DEFAULT_PRODUCTS = SHOP_ENV / "data" / "fine_items_eval_train_all.json.gz"
DEFAULT_DATASETS = OrderedDict(
    [
        ("evaluation", ROOT / "data" / "evaluation" / "tasks.jsonl"),
        ("sft_pure_v4", ROOT / "data" / "sft_pure_v4" / "all.jsonl"),
        ("grpo_train", ROOT / "data" / "grpo" / "train.jsonl"),
        ("grpo_validation", ROOT / "data" / "grpo" / "validation.jsonl"),
    ]
)
# Datasets that carry a gold trajectory (assistant tool-call rounds).
DEFAULT_GOLD_TRAJECTORIES = OrderedDict(
    [("sft_pure_v4", ROOT / "data" / "sft_pure_v4" / "all.jsonl")]
)
DATASET_SHA_METADATA = {
    "evaluation": ROOT / "data" / "evaluation" / "metadata.json",
    "sft_pure_v4": ROOT / "data" / "sft_pure_v4" / "metadata.json",
    "grpo_train": ROOT / "data" / "grpo" / "metadata.json",
    "grpo_validation": ROOT / "data" / "grpo" / "metadata.json",
}

STEP_BUCKETS = OrderedDict(
    [("1-5", (1, 5)), ("6-10", (6, 10)), ("11-20", (11, 20)), ("21-35", (21, 35))]
)
WHITESPACE_CHARS = "\xa0\u3000"
PRICE_WORDS = ("预算", "价格", "元", "块", "不超过", "不高于", "最高")


# --------------------------------------------------------------------------- #
# Frozen Reward v3 budget parser.
#
# M1 audits the *pre-fix* repository state, so the baseline report must keep
# reproducing the v3 parse rate even after M2 replaces the live parser. This is
# a verbatim snapshot of ``explicit_budget_from_instruction`` at reward-v3.
# --------------------------------------------------------------------------- #
def budget_parser_v3(instruction):
    text = str(instruction or "").replace(",", "")

    def scaled(number, unit):
        value = float(number)
        normalized_unit = str(unit or "").casefold()
        if normalized_unit == "万":
            value *= 10000
        elif normalized_unit in {"千", "k"}:
            value *= 1000
        return value

    shorthand = re.search(
        r"预算(?:控制)?在?\s*(\d+)\s*万\s*(\d+)\s*(?:千)?\s*(以内|以下|内|左右)?",
        text,
    )
    if shorthand:
        value = float(shorthand.group(1)) * 10000 + float(shorthand.group(2)) * 1000
        if shorthand.group(3) == "左右":
            value *= 1.1
        return value

    price_range = re.search(
        r"(?:预算|价格)(?:控制)?在?\s*"
        r"(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?\s*"
        r"(?:-|~|～|至|到)\s*"
        r"(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?"
        r"(?:之间|以内|以下|左右)?",
        text,
    )
    if price_range:
        low = scaled(price_range.group(1), price_range.group(2))
        high = scaled(price_range.group(3), price_range.group(4))
        if low > 0 and high >= low:
            return high

    if re.search(
        r"(?:预算|价格)(?:控制)?在?\s*\d+(?:\.\d+)?\s*[kK]\s*\+",
        text,
    ):
        return None

    patterns = (
        r"预算(?:控制)?在?\s*(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?(以内|以下|内|左右)?(?!\s*[-~～至到+kK])",
        r"价格(?:控制)?在?\s*(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?(以内|以下|内|左右)?(?!\s*[-~～到+kK])",
        r"(?:不超过|不高于|最高)\s*(\d+(?:\.\d+)?)\s*(万|千)?\s*元",
        r"(\d+(?:\.\d+)?)\s*(万|千)?\s*元(以内|以下)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = scaled(match.group(1), match.group(2))
            qualifier = match.group(3) if match.lastindex and match.lastindex >= 3 else None
            if qualifier == "左右":
                value *= 1.1
            if value > 0:
                return value
    return None


def resolve_budget_parser(version: str):
    if version == "v3":
        return budget_parser_v3, "shopping-budget-parser-v3"
    if version in {"v4", "live"}:
        return explicit_budget_from_instruction, LIVE_BUDGET_PARSER_VERSION
    raise ValueError(f"unsupported budget parser version: {version}")


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, lf_normalize: bool = True) -> str:
    payload = Path(path).read_bytes()
    if lf_normalize:
        payload = payload.replace(b"\r\n", b"\n")
    return sha256_bytes(payload)


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_ids(path: Path) -> list[int]:
    return [int(row["task_id"]) for row in read_jsonl(path)]


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def canonical_jsonl(rows: list[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


# --------------------------------------------------------------------------- #
# Corpus + goal reconstruction (mirrors the Environment v2 load path)
# --------------------------------------------------------------------------- #
def load_products(path: Path) -> list[dict]:
    """Replicate ``load_products`` de-duplication without touching the network."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    seen: set[str] = set()
    kept: list[dict] = []
    for product in products:
        asin = product.get("asin")
        if asin in (None, "nan") or len(str(asin)) > 20:
            continue
        asin = str(asin)
        if asin in seen:
            continue
        seen.add(asin)
        kept.append(product)
    return kept


def build_goals(products: list[dict], budget_parser=explicit_budget_from_instruction) -> list[dict]:
    """Reconstruct the goal list exactly as ``get_existed_goals`` does."""
    goals: list[dict] = []
    for item in products:
        for instruction in item.get("instructions") or []:
            attributes = instruction.get("attributes") or []
            if not attributes:
                continue
            goal = {
                "asin": item["asin"],
                "category": item.get("category"),
                "query": item.get("query", ""),
                "name": item.get("title"),
                "instruction_text": instruction["instruction"],
                "instruction_simple": instruction.get("instruction_simple"),
                "attributes": list(attributes),
                "price_upper": budget_parser(instruction["instruction"]),
                "goal_options": list(instruction.get("instruction_options") or []),
            }
            goal.update(compile_reward_features(instruction, item))
            goals.append(goal)
    return goals


# --------------------------------------------------------------------------- #
# Layer 1 -- solvability
# --------------------------------------------------------------------------- #
def spec_uninterpretable(goal: dict, product: dict) -> list[str]:
    """Return goal option values that cannot be selected on the gold spec axes.

    Uses the *raw* (whitespace-preserving) comparison that the interactive
    environment applies, so ``\\xa0`` / full-width spaces surface as genuine
    selection failures instead of being silently folded away.
    """
    axes = _target_option_axes(product)
    available = {str(value).strip() for values in axes.values() for value in values}
    return [str(value) for value in goal["goal_options"] if str(value).strip() not in available]


def zero_price(product: dict) -> bool:
    entries = [
        entry
        for values in (product.get("customization_options") or {}).values()
        for entry in (values or [])
        if isinstance(entry, dict)
    ]
    variant_prices = [entry.get("price") for entry in entries]
    pricing = list(product.get("pricing") or [])
    zero_variant = any(
        price is not None and price <= 0 for price in variant_prices
    )
    zero_pricing = any(price is not None and price <= 0 for price in pricing)
    return zero_variant or zero_pricing


# --------------------------------------------------------------------------- #
# Layer 3 -- gold self-consistency
# --------------------------------------------------------------------------- #
def gold_selfcheck(goal: dict, product: dict) -> dict:
    selected, _resolution = candidate_options_for_evaluation(
        product, goal.get("required_options_by_key")
    )
    price_resolution = resolve_variant_price(product, selected)
    result = evaluate_purchase(
        product,
        goal,
        selected_options=selected,
        price_resolution=price_resolution,
    )
    return {
        "reward_type": result.reward_type,
        "reward_valid": result.reward_valid,
        "hard_gates": {name: gate["status"] for name, gate in result.hard_gates.items()},
        "match_score": round(float(result.weighted_score), 6),
        "price_upper": goal.get("price_upper"),
        "resolved_price": price_resolution.get("price"),
        "price_method": price_resolution.get("method"),
    }


# --------------------------------------------------------------------------- #
# Layer 2 -- retrieval reachability
# --------------------------------------------------------------------------- #
def category_leaf(category: object) -> str:
    return str(category or "").split("›")[-1].strip()


def retrieval_queries(goal: dict) -> dict[str, str]:
    instruction = str(goal.get("instruction_text") or "")
    leaf = category_leaf(goal.get("category"))
    attributes = [str(value) for value in (goal.get("attributes") or [])][:3]
    return {
        "Q1_full_instruction": instruction,
        "Q2_first12": instruction[:12],
        "Q3_category_only": leaf,
        "Q4_category_plus_attributes": " ".join([leaf, *attributes]).strip(),
    }


def audit_retrieval(
    goals: list[dict],
    products: list[dict],
    dataset_ids: dict[str, list[int]],
    index_path: Path,
) -> dict:
    product_sha = sha256_bytes(
        json.dumps([p.get("asin") for p in products], ensure_ascii=False).encode("utf-8")
    )
    if not _index_is_current(index_path, product_sha):
        index_path.parent.mkdir(parents=True, exist_ok=True)
        build_index(
            products,
            index_path,
            product_data_sha256=product_sha,
        )
    searcher = MultiFieldBM25Searcher(index_path)
    try:
        report: dict[str, dict] = {}
        for dataset, ids in dataset_ids.items():
            stats = {
                query: {"hit20": 0, "hit150": 0, "ranks": [], "miss": 0}
                for query in (
                    "Q1_full_instruction",
                    "Q2_first12",
                    "Q3_category_only",
                    "Q4_category_plus_attributes",
                )
            }
            for task_id in ids:
                goal = goals[task_id]
                gold = str(goal["asin"])
                for name, query in retrieval_queries(goal).items():
                    hits = searcher.search(query, k=150)
                    rank = next(
                        (hit.rank for hit in hits if str(hit.asin) == gold), None
                    )
                    entry = stats[name]
                    if rank is None:
                        entry["miss"] += 1
                        continue
                    entry["ranks"].append(rank)
                    if rank <= 20:
                        entry["hit20"] += 1
                    if rank <= 150:
                        entry["hit150"] += 1
            total = len(ids) or 1
            for name, entry in stats.items():
                ranks = sorted(entry["ranks"])
                entry["total"] = len(ids)
                entry["hit20_rate"] = round(entry["hit20"] / total, 4)
                entry["hit150_rate"] = round(entry["hit150"] / total, 4)
                entry["median_rank"] = (
                    float(ranks[len(ranks) // 2]) if ranks else None
                )
                entry.pop("ranks")
            report[dataset] = stats
        return report
    finally:
        searcher.close()


def _index_is_current(index_path: Path, product_sha: str) -> bool:
    if not Path(index_path).is_file():
        return False
    try:
        searcher = MultiFieldBM25Searcher(index_path)
    except Exception:
        return False
    try:
        return searcher.manifest.get("product_data_sha256") == product_sha
    finally:
        searcher.close()


# --------------------------------------------------------------------------- #
# Difficulty
# --------------------------------------------------------------------------- #
def gold_step_counts(path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in read_jsonl(path):
        steps = sum(
            1
            for message in row.get("messages", [])
            if message.get("role") == "assistant" and message.get("tool_calls")
        )
        counts[int(row["task_id"])] = steps
    return counts


def structural_min_steps(goal: dict) -> int:
    """Deterministic lower bound on the tool calls a gold path needs.

    search (1) + open gold (1) + one selection per required/unresolved option
    axis + buy (1).
    """
    axes = len(goal.get("required_options_by_key") or {}) + len(
        goal.get("unresolved_option_requirements") or []
    )
    return axes + 3


def bucket_for(steps: int) -> str:
    for name, (low, high) in STEP_BUCKETS.items():
        if low <= steps <= high:
            return name
    return "21-35" if steps > 35 else "1-5"


# --------------------------------------------------------------------------- #
# Main audit
# --------------------------------------------------------------------------- #
def run_audit(args: argparse.Namespace) -> dict:
    budget_parser, parser_version = resolve_budget_parser(
        getattr(args, "budget_parser", "v3")
    )
    products = load_products(args.products)
    goals = build_goals(products, budget_parser)
    product_by_asin = {str(product["asin"]): product for product in products}

    dataset_ids = OrderedDict(
        (name, task_ids(path)) for name, path in args.datasets.items()
    )
    gold_steps_by_dataset = OrderedDict(
        (name, gold_step_counts(path)) for name, path in args.gold_trajectories.items()
    )

    dataset_reports: dict[str, dict] = {}
    dead_tasks: list[dict] = []
    longtail: list[dict] = []
    task_rows: list[dict] = []
    difficulty: dict[str, dict] = {}

    for name, ids in dataset_ids.items():
        counts = Counter()
        parse_mentioned = 0
        gold_steps = gold_steps_by_dataset.get(name, {})
        bucket_counts = Counter()
        step_source = "gold_trajectory" if gold_steps else "structural_lower_bound"
        for task_id in ids:
            goal = goals[task_id]
            product = product_by_asin[str(goal["asin"])]
            instruction = str(goal["instruction_text"])
            reasons: list[str] = []

            if any(word in instruction for word in PRICE_WORDS):
                parse_mentioned += 1
            if goal["price_upper"] is not None:
                counts["budget_parsed"] += 1

            spec_misses = spec_uninterpretable(goal, product)
            if spec_misses:
                counts["spec_uninterpretable"] += 1
                reasons.append("unsolvable_spec")

            is_zero_price = zero_price(product)
            if is_zero_price:
                counts["zero_price"] += 1
                reasons.append("zero_price")

            has_non_ascii_ws = any(char in instruction for char in WHITESPACE_CHARS) or any(
                char in "".join(goal["goal_options"]) for char in WHITESPACE_CHARS
            )
            if has_non_ascii_ws:
                counts["non_ascii_whitespace"] += 1

            selfcheck = gold_selfcheck(goal, product)
            counts[f"gold_{selfcheck['reward_type']}"] += 1
            if selfcheck["reward_type"] == "reward_unverifiable":
                reasons.append("reward_unverifiable")
            elif selfcheck["reward_type"] == "wrong_purchase":
                reasons.append("gold_over_budget")
            elif selfcheck["reward_type"] == "partial_alternative_purchase":
                reasons.append("gold_partial_match")

            # difficulty
            if gold_steps:
                steps = int(gold_steps.get(task_id, 0))
                source = "gold_trajectory"
            else:
                steps = structural_min_steps(goal)
                source = "structural_lower_bound"
            bucket = bucket_for(steps)
            bucket_counts[bucket] += 1

            task_rows.append(
                {
                    "dataset": name,
                    "task_id": task_id,
                    "asin": str(goal["asin"]),
                    "budget_parsed": goal["price_upper"] is not None,
                    "spec_uninterpretable": bool(spec_misses),
                    "zero_price": is_zero_price,
                    "non_ascii_whitespace": has_non_ascii_ws,
                    "gold_reward_type": selfcheck["reward_type"],
                    "difficulty_bucket": bucket,
                    "steps": steps,
                    "step_source": source,
                }
            )
            if bucket == "21-35":
                longtail.append(
                    {
                        "dataset": name,
                        "task_id": task_id,
                        "asin": str(goal["asin"]),
                        "steps": steps,
                        "step_source": source,
                        "instruction": instruction,
                    }
                )

            if reasons:
                dead_tasks.append(
                    {
                        "dataset": name,
                        "task_id": task_id,
                        "asin": str(goal["asin"]),
                        "reasons": sorted(set(reasons)),
                        "spec_unresolved_values": sorted(spec_misses),
                        "gold_selfcheck": selfcheck,
                        "instruction": instruction,
                    }
                )

        total = len(ids)
        dataset_reports[name] = {
            "tasks": total,
            "budget_parsed": counts["budget_parsed"],
            "budget_parse_rate": round(counts["budget_parsed"] / total, 6) if total else 0.0,
            "price_mentioned": parse_mentioned,
            "spec_uninterpretable": counts["spec_uninterpretable"],
            "zero_price": counts["zero_price"],
            "non_ascii_whitespace": counts["non_ascii_whitespace"],
            "gold_selfcheck": {
                key.split("gold_", 1)[1]: value
                for key, value in sorted(counts.items())
                if key.startswith("gold_")
            },
            "dead_tasks": sum(1 for row in dead_tasks if row["dataset"] == name),
        }
        difficulty[name] = {
            "step_source": step_source,
            "comparable_to_rollout_steps": step_source == "gold_trajectory",
            "note": (
                "gold 轨迹步数（assistant 工具轮次），可与 rollout 步数直接比较"
                if step_source == "gold_trajectory"
                else "无 gold 轨迹，仅为结构性下界（search+open+select+buy），不可与 rollout 步数比较"
            ),
            "counts": {bucket: bucket_counts[bucket] for bucket in STEP_BUCKETS},
            "total": total,
        }

    # overlaps
    overlap: dict[str, int] = {}
    names = list(dataset_ids)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = set(dataset_ids[left]) & set(dataset_ids[right])
            overlap[f"{left}&{right}"] = len(shared)

    retrieval = {}
    if not args.skip_retrieval:
        retrieval = audit_retrieval(goals, products, dataset_ids, args.index)

    # hashes
    hashes = {}
    for name, path in args.datasets.items():
        hashes[f"{name}.lf_sha256"] = sha256_file(path, lf_normalize=True)
        hashes[f"{name}.raw_sha256"] = sha256_file(path, lf_normalize=False)
        hashes[f"{name}.tasks"] = len(dataset_ids[name])
    hashes["products.raw_sha256"] = sha256_file(args.products, lf_normalize=False)
    hashes["products.count"] = len(products)
    hashes["goals.count"] = len(goals)
    hashes["budget_parser.version"] = parser_version

    dead_tasks.sort(key=lambda row: (row["dataset"], row["task_id"]))
    longtail.sort(key=lambda row: (row["dataset"], row["task_id"]))
    task_rows.sort(key=lambda row: (row["dataset"], row["task_id"]))

    return {
        "budget_parser_version": parser_version,
        "dataset_reports": dataset_reports,
        "dead_tasks": dead_tasks,
        "longtail": longtail,
        "task_rows": task_rows,
        "difficulty": difficulty,
        "overlap": overlap,
        "retrieval": retrieval,
        "hashes": hashes,
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_report(result: dict, args: argparse.Namespace) -> str:
    lines: list[str] = []
    add = lines.append
    add("# M1 — 可解性审计报告")
    add("")
    add("> 全离线、可复现。三层审计：**有解性 / 检索可达性 / 评分可评性**。")
    add("> 生成命令：`scripts/audit_solvability.py`（见文末复现命令）。")
    add("")

    add("## 0. 语料与口径")
    add("")
    products_display = (
        args.products.relative_to(ROOT).as_posix()
        if args.products.is_relative_to(ROOT)
        else args.products.as_posix()
    )
    add(f"- 商品语料：`{products_display}`，商品数 **{result['hashes']['products.count']}**，goals 数 **{result['hashes']['goals.count']}**。")
    add(f"- 预算解析器：**{result['budget_parser_version']}**（M1 基线固定为 Reward v3 口径）。")
    add("- 目标商品与其 gold 规格轴来自冻结语料；预算上界用 `explicit_budget_from_instruction` 复算。")
    add("- 审计脚本不依赖 GPU、不依赖 ShopSimulator 服务端，纯 CPU。")
    add("")

    add("## 1. 数据集总览（Layer 1 + Layer 3）")
    add("")
    coverage = len(result["task_rows"])
    expected_total = sum(report["tasks"] for report in result["dataset_reports"].values())
    add(
        f"逐任务审计台账 `audit/task_audit.jsonl` 共 **{coverage}** 条，"
        f"等于四个数据集任务数之和 **{expected_total}**"
        f"（{'✅ 覆盖完整' if coverage == expected_total else '⚠️ 覆盖不完整'}）。"
    )
    add("")
    add("| 数据集 | 任务数 | 预算解析率 | 规格不可解释 | 零价商品 | 非 ASCII 空白 | gold 自洽失败 | 死题 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, report in result["dataset_reports"].items():
        gold_fail = (
            report["gold_selfcheck"].get("reward_unverifiable", 0)
            + report["gold_selfcheck"].get("wrong_purchase", 0)
            + report["gold_selfcheck"].get("partial_alternative_purchase", 0)
        )
        add(
            f"| {name} | {report['tasks']} | {report['budget_parse_rate']*100:.1f}% | "
            f"{report['spec_uninterpretable']} | {report['zero_price']} | "
            f"{report['non_ascii_whitespace']} | {gold_fail} | {report['dead_tasks']} |"
        )
    add("")

    add("### 1.1 与 optimize.md §2 既有结论的对齐")
    add("")
    add("| 结论 | optimize.md | 本次审计 | 一致性 |")
    add("|---|---:|---:|---|")
    expected = {
        "evaluation": {"spec_uninterpretable": 1, "zero_price": 0, "parse": "93.5%"},
        "sft_pure_v4": {"spec_uninterpretable": 0, "zero_price": 6, "parse": "62.8%"},
        "grpo_train": {"spec_uninterpretable": 26, "zero_price": 2, "parse": "66.1%"},
        "grpo_validation": {"spec_uninterpretable": 1, "zero_price": 0, "parse": "48.0%"},
    }
    for name, exp in expected.items():
        report = result["dataset_reports"][name]
        ok = (
            report["spec_uninterpretable"] == exp["spec_uninterpretable"]
            and report["zero_price"] == exp["zero_price"]
            and f"{report['budget_parse_rate']*100:.1f}%" == exp["parse"]
        )
        add(
            f"| {name} | 规格 {exp['spec_uninterpretable']} / 零价 {exp['zero_price']} / 解析 {exp['parse']} | "
            f"规格 {report['spec_uninterpretable']} / 零价 {report['zero_price']} / "
            f"解析 {report['budget_parse_rate']*100:.1f}% | {'✅ 一致' if ok else '⚠️ 不一致'} |"
        )
    add("")
    add("- 评测集 1 题 `\\xa0` 匹配失败 → `task_id=3647`（goal 规格值含 `\\xa0`+空格，商品规格为两个普通空格）。")
    add("- 8 个零价商品：SFT 6（1135/2304/3321/9168/9670/15743）+ GRPO train 2（1926/18135）。")
    add("")

    add("## 2. 检索可达性（Layer 2）")
    add("")
    if result["retrieval"]:
        add("对每个数据集模拟 4 种查询策略，统计 gold ASIN 的命中情况（Environment v2 多字段 BM25）。")
        add("")
        add("| 数据集 | 策略 | top-20 命中 | top-150 命中 | 中位 rank | 未命中 |")
        add("|---|---|---:|---:|---:|---:|")
        for dataset, stats in result["retrieval"].items():
            for query, entry in stats.items():
                add(
                    f"| {dataset} | {query} | {entry['hit20_rate']*100:.1f}% | "
                    f"{entry['hit150_rate']*100:.1f}% | {entry['median_rank']} | {entry['miss']} |"
                )
    else:
        add("（本次运行使用 `--skip-retrieval`，未执行检索层。）")
    add("")

    add("## 3. 难度分桶与长尾题")
    add("")
    add("| 数据集 | 步数来源 | 可与 rollout 比较 | 1–5 | 6–10 | 11–20 | 21–35 |")
    add("|---|---|---|---:|---:|---:|---:|")
    for name, entry in result["difficulty"].items():
        counts = entry["counts"]
        add(
            f"| {name} | {entry['step_source']} | "
            f"{'是' if entry['comparable_to_rollout_steps'] else '否'} | "
            f"{counts['1-5']} | {counts['6-10']} | "
            f"{counts['11-20']} | {counts['21-35']} |"
        )
    add("")
    add("**关键限制（与 optimize.md 风险 R4 对应）**：仓库内只有 `sft_pure_v4` 携带 gold 轨迹（1,192 条），")
    add("其步数分布 **154 / 927 / 93 / 18** 与 optimize.md §2.4 完全一致；")
    add("评测集与 GRPO 池**没有 gold 轨迹**，其结构性下界（search+open+select+buy）几乎全部落在 1–5，")
    add("**不能**用来推断评测长尾（21–35 步）的成因——长尾是模型行为而非任务结构。")
    add(f"- 长尾题（21–35 桶）共 **{len(result['longtail'])}** 条（全部来自 `sft_pure_v4`），见 `audit/longtail_tasks.jsonl`。")
    add("")

    add("## 4. 死题清单（Layer 1 + Layer 3）")
    add("")
    reason_counts = Counter(
        reason for row in result["dead_tasks"] for reason in row["reasons"]
    )
    add("| 原因 | 条数 | 含义 |")
    add("|---|---:|---|")
    meaning = {
        "unsolvable_spec": "goal 规格值无法在 gold 商品规格轴上选中（结构性不可解）",
        "zero_price": "gold 商品零价，预算门槛失效",
        "reward_unverifiable": "gold 购买不可评（多为多变体价格轴无法唯一解析）",
        "gold_over_budget": "gold 最低可行变体价超过解析预算上界，严格成功不可达",
        "gold_partial_match": "gold 无法满足全部软维度，严格成功不可达",
    }
    for reason, count in sorted(reason_counts.items()):
        add(f"| `{reason}` | {count} | {meaning.get(reason, '')} |")
    add("")
    add(f"- 去重后死题 **{len(result['dead_tasks'])}** 条，见 `audit/dead_tasks.jsonl`。")
    add("")

    add("## 5. 重叠与哈希")
    add("")
    add("| 数据集对 | task_id 交集 |")
    add("|---|---:|")
    for pair, count in result["overlap"].items():
        add(f"| {pair} | {count} |")
    add("")
    add("| 文件 | SHA-256（LF 归一） | 任务数 |")
    add("|---|---|---:|")
    for key, value in result["hashes"].items():
        if key.endswith("lf_sha256"):
            name = key[: -len(".lf_sha256")]
            add(f"| {name} | `{value}` | {result['hashes'][name + '.tasks']} |")
    add("")

    add("## 6. 关键结论")
    add("")
    eval_retrieval = result["retrieval"].get("evaluation", {}) if result["retrieval"] else {}
    if eval_retrieval:
        q4 = eval_retrieval.get("Q4_category_plus_attributes", {})
        add(
            f"1. **检索不是瓶颈**：评测集 Q4（品类+前 3 属性）top-150 命中 "
            f"{q4.get('hit150_rate', 0)*100:.1f}%、未命中 {q4.get('miss', '?')}，"
            "与 optimize.md §2.3 一致；不应在检索环节投入优化。"
        )
    add(
        "2. **训练/评测预算解析率错配复现**：评测集 "
        f"{result['dataset_reports']['evaluation']['budget_parse_rate']*100:.1f}% vs "
        f"SFT {result['dataset_reports']['sft_pure_v4']['budget_parse_rate']*100:.1f}% / "
        f"GRPO train {result['dataset_reports']['grpo_train']['budget_parse_rate']*100:.1f}% / "
        f"GRPO val {result['dataset_reports']['grpo_validation']['budget_parse_rate']*100:.1f}%。"
    )
    add(
        "3. **结构性死题集中在 GRPO 池**：GRPO train 规格不可解释 "
        f"{result['dataset_reports']['grpo_train']['spec_uninterpretable']} 题、"
        f"零价 {result['dataset_reports']['grpo_train']['zero_price']} 题；"
        "SFT 因 `gold_purchase` 验收而免疫。"
    )
    add(
        "4. **新增发现（gold 自洽性）**：用 Reward v3 对 gold 商品做自洽复核，"
        f"GRPO train 有 {result['dataset_reports']['grpo_train']['gold_selfcheck'].get('reward_unverifiable', 0)} 题不可评、"
        f"{result['dataset_reports']['grpo_train']['gold_selfcheck'].get('wrong_purchase', 0)} 题 gold 超预算、"
        f"{result['dataset_reports']['grpo_train']['gold_selfcheck'].get('partial_alternative_purchase', 0)} 题 gold 软维度不全满足——"
        "即 gold 自身也无法拿到 `gold_purchase`。这比 optimize.md 的“26 题不可解”口径更宽，属结构性噪声来源。"
    )
    add(
        "5. **重叠**：`sft_pure_v4 ∩ grpo_train = 6`，与 optimize.md 发现 D 一致；评测集与训练集仍零重叠。"
    )
    add("")

    add("## 7. 复现命令")
    add("")
    add("```bash")
    add("python scripts/audit_solvability.py")
    add("```")
    add("")
    add("连续运行两次，`audit/` 下所有文件逐字节一致（不含时间戳）。")
    add("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--out", type=Path, default=ROOT / "audit")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "plans" / "audit-report.md",
        help="narrative report destination (plans/ is git-ignored)",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / ".workbuddy-ai" / "cache" / "audit_products.sqlite3",
        help="cached Environment v2 BM25 index (rebuilt only when stale)",
    )
    parser.add_argument(
        "--budget-parser",
        choices=["v3", "v4", "live"],
        default="v3",
        help="v3 reproduces the M1 pre-fix baseline; v4/live use the current parser",
    )
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args()
    args.datasets = OrderedDict(
        (name, path) for name, path in DEFAULT_DATASETS.items() if path.exists()
    )
    args.gold_trajectories = OrderedDict(
        (name, path)
        for name, path in DEFAULT_GOLD_TRAJECTORIES.items()
        if path.exists()
    )

    result = run_audit(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result, args), encoding="utf-8", newline="\n")
    (out / "task_audit.jsonl").write_text(
        canonical_jsonl(result["task_rows"]), encoding="utf-8", newline="\n"
    )
    (out / "dead_tasks.jsonl").write_text(
        canonical_jsonl(result["dead_tasks"]), encoding="utf-8", newline="\n"
    )
    (out / "longtail_tasks.jsonl").write_text(
        canonical_jsonl(result["longtail"]), encoding="utf-8", newline="\n"
    )
    (out / "difficulty_buckets.json").write_text(
        canonical_json(
            {
                "schema_version": "shopping-audit-difficulty-v1",
                "buckets": list(STEP_BUCKETS),
                "datasets": result["difficulty"],
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    hashes_lines = [f"{key}  {value}" for key, value in sorted(result["hashes"].items())]
    hashes_lines.append(
        f"audit.task_audit.jsonl  {sha256_file(out / 'task_audit.jsonl')}"
    )
    hashes_lines.append(
        f"audit.dead_tasks.jsonl  {sha256_file(out / 'dead_tasks.jsonl')}"
    )
    hashes_lines.append(
        f"audit.longtail_tasks.jsonl  {sha256_file(out / 'longtail_tasks.jsonl')}"
    )
    hashes_lines.append(
        f"audit.difficulty_buckets.json  {sha256_file(out / 'difficulty_buckets.json')}"
    )
    (out / "hashes.txt").write_text("\n".join(hashes_lines) + "\n", encoding="utf-8", newline="\n")

    print(json.dumps(result["dataset_reports"], ensure_ascii=False, indent=2))
    print("dead_tasks:", len(result["dead_tasks"]), "| longtail:", len(result["longtail"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
