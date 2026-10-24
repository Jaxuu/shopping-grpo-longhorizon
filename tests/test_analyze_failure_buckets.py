"""`analyze_failure_buckets.py` 的分桶与汇总口径。"""

from __future__ import annotations

import json
import pathlib

from scripts.analyze_failure_buckets import (
    bucket,
    extract,
    format_report,
    load_rows,
    summarize,
    verdict,
)


def make_row(
    task_id: str,
    *,
    reward_type: str = "wrong_purchase",
    gold: str | None = "G",
    purchased: str | None = "X",
    opened: list[str] | None = None,
    pages: int = 0,
    searches: int = 1,
    budget_status: str = "pass",
    category_status: str = "pass",
    dims: dict | None = None,
    buy: bool = True,
) -> dict:
    opened = ["X"] if opened is None else opened
    steps = [{"tool_name": "search_products", "parameters": {"query": "q"}}] * searches
    steps += [{"tool_name": "open_product", "parameters": {"asin": a}} for a in opened]
    steps += [{"tool_name": "next_page", "parameters": {}}] * pages
    if buy:
        steps += [{"tool_name": "buy_now", "parameters": {}}]
    return {
        "task_id": task_id,
        "steps": steps,
        "terminal_result": {
            "goal": {"asin": gold} if gold else {},
            "purchase": {"asin": purchased} if purchased else {},
            "reward_detail": {
                "reward_type": reward_type,
                "target_asin_match": purchased == gold,
                "hard_gates": {
                    "budget": {"status": budget_status, "required": 50, "actual": 80},
                    "category": {"status": category_status},
                },
                "dimension_scores": dims or {},
            },
        },
    }


def test_bucket_covers_four_failure_shapes() -> None:
    ok = make_row("1", reward_type="gold_purchase", purchased="G")
    assert bucket(extract(ok)) == "success"

    gate = make_row("2", purchased="G", budget_status="fail")
    assert bucket(extract(gate)) == "bought_gold_gate_failed"

    other = make_row("3", purchased="X", opened=["G", "X"])
    assert bucket(extract(other)) == "saw_gold_bought_other"

    none_bought = make_row("4", purchased=None, opened=["G"], buy=False)
    assert bucket(extract(none_bought)) == "saw_gold_no_purchase"

    never = make_row("5", purchased="X", opened=["X", "Y"])
    assert bucket(extract(never)) == "never_saw_gold"

    unknown = make_row("6", gold=None, purchased="X")
    assert bucket(extract(unknown)) == "unknown"


def test_purchased_falls_back_to_last_opened() -> None:
    row = make_row("7", purchased=None, opened=["A", "B"])
    assert extract(row)["purchased"] == "B"


def test_summarize_counts_and_medians() -> None:
    rows = [
        make_row("1", reward_type="gold_purchase", purchased="G", opened=["G"]),
        make_row("2", purchased="G", opened=["G"], budget_status="fail"),
        make_row("3", purchased="X", opened=["G", "X"]),
        make_row("4", purchased="X", opened=["X", "Y", "Z"], pages=2),
    ]
    records = [extract(r) for r in rows]
    summary = summarize(records)

    assert summary["total"] == 4
    assert summary["successes"] == 1
    assert summary["rate"] == 0.25
    assert summary["buckets"]["bought_gold_gate_failed"] == 1
    assert summary["buckets"]["saw_gold_bought_other"] == 1
    assert summary["buckets"]["never_saw_gold"] == 1
    assert summary["buckets"]["saw_gold_no_purchase"] == 0
    assert summary["gates"]["budget:fail"] == 1
    # 失败轨迹打开数：1, 2, 3 → 中位数 2
    assert summary["fail_opened_median"] == 2


def test_dimension_rows_sorted_by_miss_rate() -> None:
    rows = [
        make_row("1", dims={"capacity": 0.0, "color": 1.0}),
        make_row("2", dims={"capacity": 0.5, "color": 1.0}),
        make_row("3", dims={"capacity": 1.0, "color": 0.0}),
    ]
    summary = summarize([extract(r) for r in rows])
    names = [row["name"] for row in summary["dims"]]
    assert names[0] == "capacity"  # 2/3 没满分，高于 color 的 1/3
    assert summary["dims"][0]["below_full"] == 2 / 3


def test_verdict_points_at_largest_bucket() -> None:
    rows = [
        make_row("1", purchased="G", budget_status="fail"),
        make_row("2", purchased="G", budget_status="fail"),
        make_row("3", purchased="X", opened=["X", "Y"]),
    ]
    summary = summarize([extract(r) for r in rows])
    lines = verdict(summary)
    assert "买对了商品但没过硬门" in lines[0]
    assert any("超预算" in line for line in lines)


def test_report_renders_without_dimensions() -> None:
    summary = summarize([extract(make_row("1", purchased="G", budget_status="fail"))])
    text = format_report("demo", summary)
    assert "bought_gold_gate_failed" in text
    assert "失败桶" in text
    assert "判读" in text


def test_load_rows_skips_blank_lines(tmp: pathlib.Path) -> None:
    path = tmp / "t.jsonl"
    path.write_text(
        json.dumps(make_row("1")) + "\n\n" + json.dumps(make_row("2")) + "\n",
        encoding="utf-8",
    )
    assert len(load_rows(path)) == 2
