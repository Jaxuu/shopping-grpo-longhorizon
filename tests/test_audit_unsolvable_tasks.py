"""`audit_unsolvable_tasks.py` 的可解性判定口径。"""

from __future__ import annotations

import json
import pathlib

from scripts.audit_unsolvable_tasks import (
    classify,
    collect,
    extract,
    format_report,
    summarize,
)


def make_row(
    task_id: str,
    *,
    reward_type: str = "wrong_purchase",
    gold: str = "G",
    purchased: str | None = "X",
    budget_status: str = "pass",
    required: float | None = 50,
    actual: float | None = 80,
    dims: dict | None = None,
) -> dict:
    return {
        "task_id": task_id,
        "steps": [],
        "terminal_result": {
            "goal": {"asin": gold},
            "purchase": {"asin": purchased} if purchased else {},
            "reward_detail": {
                "reward_type": reward_type,
                "target_asin_match": purchased == gold,
                "hard_gates": {
                    "budget": {"status": budget_status, "required": required, "actual": actual}
                },
                "dimension_scores": dims or {},
            },
        },
    }


def test_classify_solved() -> None:
    records = [extract(make_row("1", reward_type="gold_purchase", purchased="G"))]
    assert classify(records)["status"] == "solved"


def test_classify_product_mismatch_when_brand_misses() -> None:
    """买到 gold，但品牌维度不满分 ⇒ 商品级对不上，疑似无解。"""
    records = [
        extract(
            make_row(
                "2",
                reward_type="partial_alternative_purchase",
                purchased="G",
                dims={"brand": 0.0, "model": 1.0, "core_functions": 1.0, "key_options": 1.0},
            )
        )
    ]
    item = classify(records)
    assert item["status"] == "product_mismatch"
    assert item["product_miss"] == {"brand": 1}


def test_classify_variant_only_when_only_options_miss() -> None:
    """只有 key_options 失分 ⇒ 变体选错，可救。"""
    records = [
        extract(
            make_row(
                "3",
                reward_type="partial_alternative_purchase",
                purchased="G",
                dims={"brand": 1.0, "model": 1.0, "core_functions": 1.0, "key_options": 0.0},
            )
        )
    ]
    assert classify(records)["status"] == "variant_only"


def test_classify_gold_over_budget() -> None:
    records = [
        extract(make_row("4", purchased="G", budget_status="fail", required=50, actual=88))
    ]
    item = classify(records)
    assert item["status"] == "gold_over_budget"
    assert item["budget_fail_on_gold"] == 1
    assert "50" in item["budget_hint"] and "88" in item["budget_hint"]


def test_classify_never_solved_is_not_blocked() -> None:
    """从没买到 gold 只是难，不能算无解。"""
    records = [extract(make_row("5", purchased="X"))]
    assert classify(records)["status"] == "never_solved"


def test_solved_wins_over_blocked() -> None:
    """只要有一个模型做对过，就不是无解。"""
    records = [
        extract(make_row("5", reward_type="partial_alternative_purchase", purchased="G")),
        extract(make_row("5", reward_type="gold_purchase", purchased="G")),
    ]
    item = classify(records)
    assert item["status"] == "solved"
    assert item["models_solved"] == 1


def test_collect_merges_across_dirs(tmp: pathlib.Path) -> None:
    for name, rows in (
        ("a", [make_row("1", purchased="G", budget_status="fail")]),
        ("b", [make_row("1", purchased="G", budget_status="fail"),
               make_row("2", reward_type="gold_purchase", purchased="G")]),
    ):
        d = tmp / name
        d.mkdir()
        (d / "trajectories.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    per_task, dirs = collect([tmp / "a" / "trajectories.jsonl", tmp / "b" / "trajectories.jsonl"])
    assert dirs == 2
    assert len(per_task["1"]) == 2
    assert len({r["model"] for r in per_task["1"]}) == 2


def test_summary_counts_and_report(tmp: pathlib.Path) -> None:
    rows_a = [
        make_row("1", reward_type="gold_purchase", purchased="G"),
        make_row("2", reward_type="partial_alternative_purchase", purchased="G",
                 dims={"key_options": 0.0}),
        make_row("3", purchased="X"),
        make_row("4", reward_type="partial_alternative_purchase", purchased="G",
                 dims={"brand": 0.0}),
    ]
    d = tmp / "m"
    d.mkdir()
    (d / "trajectories.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows_a), encoding="utf-8"
    )
    per_task, dirs = collect([d / "trajectories.jsonl"])
    summary = summarize([classify(v) for v in per_task.values()], dirs)
    assert summary["status"] == {
        "solved": 1,
        "product_mismatch": 1,
        "gold_over_budget": 0,
        "variant_only": 1,
        "never_solved": 1,
    }
    text = format_report(summary)
    assert "疑似无解" in text
    assert "可救" in text
    assert "task_id" in text
