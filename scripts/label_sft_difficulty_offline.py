#!/usr/bin/env python3
"""Deterministic offline difficulty labels for SFT rows the teacher never saw.

``scripts/label_sft_difficulty.py`` needs an OpenAI-compatible endpoint. When no
endpoint is reachable this script fills the gap with a **transparent rule set**
over features that are already observable offline:

* ``ambiguity``        -- list-page near-duplicate count from the M2 target list;
* ``opens``/``searches``/``distinct_queries`` -- teacher trajectory shape;
* ``axes``             -- number of option axes on the gold product;
* ``price_upper``      -- parsed budget ceiling.

The rules mirror the axes in the DeepSeek prompt (category/variant ambiguity,
candidate comparison, query rewrite, budget sensitivity), but they are a
heuristic, not a model judgement. Every row is tagged
``label_input_mode: "offline_heuristic_v1"`` so it is always distinguishable, and
``--drop-heuristic`` removes them again in one step, after which the real
labeler can be re-run:

    python scripts/label_sft_difficulty_offline.py --drop-heuristic
    python scripts/label_sft_difficulty.py --input data/sft_pure_v4/all.jsonl \\
        --output data/sft_pure_v4/difficulty_labels.jsonl
    python scripts/prepare_sft_curriculum.py --admission-contract shopping-task-admission-v1
    python scripts/update_sft_metadata.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
if str(SHOP_ENV) not in sys.path:
    sys.path.append(str(SHOP_ENV))

_SPEC = importlib.util.spec_from_file_location(
    "audit_solvability", ROOT / "scripts" / "audit_solvability.py"
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)

SFT = ROOT / "data" / "sft_pure_v4"
COLLECTION = ROOT / "data" / "collection"
HEURISTIC_MODE = "offline_heuristic_v1"
HEURISTIC_CONFIDENCE = 0.4


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tool_calls(messages: list[dict]) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append((str(function.get("name") or ""), arguments if isinstance(arguments, dict) else {}))
    return calls


def trajectory_features(messages: list[dict]) -> dict:
    calls = _tool_calls(messages)
    queries = [str(args.get("query") or "") for name, args in calls if name == "search_products"]
    repeated = sum(
        1
        for left, right in zip(calls, calls[1:])
        if left == right
    )
    return {
        "steps": len(calls),
        "searches": len(queries),
        "distinct_queries": len(set(queries)),
        "opens": sum(1 for name, _ in calls if name == "open_product"),
        "repeated_actions": repeated,
    }


def price_band(price_upper: float | None) -> str:
    if price_upper is None:
        return "unknown"
    if price_upper < 50:
        return "low"
    if price_upper <= 200:
        return "medium"
    return "high"


def label_row(
    *,
    task_id: int,
    features: dict,
    ambiguity: int | None,
    axes: int,
    price_upper: float | None,
    domain: str,
) -> dict:
    needs_comparison = features["opens"] >= 2 or (ambiguity is not None and ambiguity >= 20)
    needs_rewrite = features["distinct_queries"] >= 2
    variant_complexity = "high" if axes >= 2 else ("medium" if axes == 1 else "low")
    budget_risk = "medium" if price_upper is not None else "low"
    loop_risk = "high" if features["repeated_actions"] >= 2 else "low"

    if (ambiguity is not None and ambiguity >= 60 and needs_comparison) or features["steps"] >= 16:
        difficulty = "hard"
    elif (ambiguity is not None and ambiguity >= 20) or needs_rewrite or features["steps"] >= 8 or axes >= 2:
        difficulty = "medium"
    else:
        difficulty = "simple"

    if features["steps"] <= 5:
        trajectory_complexity = "simple"
    elif features["steps"] <= 12:
        trajectory_complexity = "medium"
    else:
        trajectory_complexity = "hard"

    evidence = (
        f"离线规则：ambiguity={ambiguity}，候选比较={needs_comparison}，"
        f"改写查询={needs_rewrite}，步数={features['steps']}，规格轴={axes}"
    )[:80]
    return {
        "task_id": int(task_id),
        "difficulty": difficulty,
        "trajectory_complexity": trajectory_complexity,
        "domain": domain,
        "price_band": price_band(price_upper),
        "needs_query_rewrite": bool(needs_rewrite),
        "needs_candidate_comparison": bool(needs_comparison),
        "variant_complexity": variant_complexity,
        "budget_risk": budget_risk,
        "loop_risk": loop_risk,
        "confidence": HEURISTIC_CONFIDENCE,
        "evidence": evidence,
        "label_input_mode": HEURISTIC_MODE,
    }


def load_ambiguity(path: Path) -> dict[int, int]:
    if not path.exists():
        return {}
    return {int(row["task_id"]): int(row["ambiguity"]) for row in read_jsonl(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SFT / "all.jsonl")
    parser.add_argument("--output", type=Path, default=SFT / "difficulty_labels.jsonl")
    parser.add_argument(
        "--targets-detail", type=Path, default=COLLECTION / "collect_targets_detail.jsonl"
    )
    parser.add_argument(
        "--drop-heuristic",
        action="store_true",
        help="Remove existing offline_heuristic_v1 labels and exit (upgrade path).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.drop_heuristic:
        rows = read_jsonl(args.output) if args.output.exists() else []
        kept = [row for row in rows if row.get("label_input_mode") != HEURISTIC_MODE]
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in kept),
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps({"before": len(rows), "after": len(kept), "dropped": len(rows) - len(kept)}))
        return 0

    products = audit.load_products(audit.DEFAULT_PRODUCTS)
    goals = audit.build_goals(products)
    product_by_asin = {str(product["asin"]): product for product in products}
    ambiguity = load_ambiguity(args.targets_detail)

    rows = read_jsonl(args.input)
    existing = read_jsonl(args.output) if args.output.exists() else []
    existing_ids = {int(row["task_id"]) for row in existing}
    pending = [row for row in rows if int(row["task_id"]) not in existing_ids]

    new_labels = []
    for row in pending:
        task_id = int(row["task_id"])
        goal = goals[task_id]
        product = product_by_asin[str(goal["asin"])]
        axes = len(product.get("customization_options") or {})
        new_labels.append(
            label_row(
                task_id=task_id,
                features=trajectory_features(row["messages"]),
                ambiguity=ambiguity.get(task_id),
                axes=axes,
                price_upper=goal.get("price_upper"),
                domain=str(product.get("domain_zh") or goal.get("category") or "").split("›")[-1],
            )
        )

    if new_labels:
        args.output.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in existing + new_labels
            ),
            encoding="utf-8",
            newline="\n",
        )

    histogram: dict[str, int] = {}
    for label in new_labels:
        histogram[label["difficulty"]] = histogram.get(label["difficulty"], 0) + 1
    print(
        json.dumps(
            {
                "existing": len(existing),
                "labelled": len(new_labels),
                "total": len(existing) + len(new_labels),
                "difficulty_histogram": histogram,
                "label_input_mode": HEURISTIC_MODE,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
