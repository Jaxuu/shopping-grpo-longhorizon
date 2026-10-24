"""`analyze_unstable_gap.py` 的归类口径与并集计算。"""

from __future__ import annotations

import json
import pathlib

from scripts.analyze_unstable_gap import (
    build_gap,
    classify,
    collect,
    extract,
    format_report,
    parse_listing,
)

GOLD_ASIN = "831919364430"


def listing(page: int, rows: list[tuple[int, str]], total_pages: int = 8) -> str:
    head = (
        "[SHOPPING_OBSERVATION_V2]\npage_type: search_results\nquery: q\n"
        f"Page {page} of {total_pages} (Total results: 150; ranks 1-20)\n"
        "格式: rank|asin|price|brand|category|key_attributes|title\n"
    )
    body = "\n".join(f"{rank}|{asin}|9.0|b|c|a|t" for rank, asin in rows)
    return head + body


def make_row(
    task_id: str,
    *,
    reward_type: str = "wrong_purchase",
    gold: str | None = GOLD_ASIN,
    purchased: str | None = "999999999999",
    queries: list[str] | None = None,
    listings: list[tuple[int, list[tuple[int, str]]]] | None = None,
    opened: list[str] | None = None,
    pages: int = 0,
    budget_status: str = "pass",
    dims: dict | None = None,
) -> dict:
    queries = ["default query"] if queries is None else queries
    opened = [] if opened is None else opened
    steps: list[dict] = [
        {
            "tool_name": "search_products",
            "parameters": {"query": query},
            "observation": listing(1, []),
        }
        for query in queries
    ]
    if listings:
        # 第一页的结果挂在第一次搜索上，其余页用 next_page（同一查询的延续）
        steps[0]["observation"] = listing(*listings[0])
        for page, rows in listings[1:]:
            steps.append(
                {"tool_name": "next_page", "parameters": {}, "observation": listing(page, rows)}
            )
    steps += [{"tool_name": "open_product", "parameters": {"asin": a}} for a in opened]
    steps += [{"tool_name": "next_page", "parameters": {}}] * pages
    return {
        "task_id": task_id,
        "steps": steps,
        "terminal_result": {
            "goal": {"asin": gold} if gold else {},
            "purchase": {"asin": purchased} if purchased else {},
            "reward_detail": {
                "reward_type": reward_type,
                "target_asin_match": bool(gold) and purchased == gold,
                "hard_gates": {"budget": {"status": budget_status}},
                "dimension_scores": dims or {},
            },
        },
    }


def test_parse_listing_reads_page_and_rows() -> None:
    page, rows = parse_listing(listing(2, [(1, "111"), (2, "222")]))
    assert page == 2
    assert rows == [(1, "111"), (2, "222")]


def test_extract_records_gold_sighting_with_query() -> None:
    row = make_row(
        "1",
        queries=["q one"],
        listings=[(1, [(1, "111"), (3, GOLD_ASIN)])],
        opened=["111"],
    )
    record = extract(row)
    assert record["queries"] == ["q one"]
    assert len(record["sightings"]) == 1
    sighting = record["sightings"][0]
    assert (sighting["page"], sighting["rank"], sighting["query"]) == (1, 3, "q one")


def test_extract_records_deep_sighting_page() -> None:
    row = make_row(
        "1",
        queries=["q"],
        listings=[(1, [(1, "111")]), (2, [(5, GOLD_ASIN)])],
        opened=["111"],
    )
    record = extract(row)
    assert [(s["page"], s["rank"]) for s in record["sightings"]] == [(2, 5)]


def test_classify_variant_and_budget() -> None:
    variant = extract(
        make_row("1", reward_type="partial_alternative_purchase", purchased=GOLD_ASIN)
    )
    assert classify(variant, [])[0] == "选择-规格"

    over = extract(make_row("2", purchased=GOLD_ASIN, budget_status="fail"))
    assert classify(over, [])[0] == "门控-预算"

    other = extract(make_row("3", purchased=GOLD_ASIN, reward_type="repeat_loop"))
    assert classify(other, [])[0] == "门控-其他"


def test_classify_gold_visible_in_own_listing() -> None:
    target = extract(
        make_row(
            "1",
            queries=["q"],
            listings=[(1, [(2, GOLD_ASIN)])],
            opened=["111"],
        )
    )
    category, reason = classify(target, [])
    assert category == "选择-列表取舍"
    assert "第 1 页第 2 名" in reason


def test_classify_never_visible_splits_on_winner_page() -> None:
    target = extract(make_row("1", queries=["q"], opened=["111"]))
    winner_first = extract(
        make_row("1", reward_type="gold_purchase", purchased=GOLD_ASIN,
                 listings=[(1, [(7, GOLD_ASIN)])], opened=[GOLD_ASIN])
    )
    assert classify(target, [winner_first])[0] == "检索-查询词"

    winner_deep = extract(
        make_row("1", reward_type="gold_purchase", purchased=GOLD_ASIN,
                 listings=[(1, [(1, "111")]), (2, [(4, GOLD_ASIN)])], opened=[GOLD_ASIN])
    )
    assert classify(target, [winner_deep])[0] == "翻页-深度"


def test_classify_missing_gold_is_unknown() -> None:
    record = extract(make_row("1", gold=None, purchased=None))
    assert classify(record, [])[0] == "未知"


def _write(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


def test_build_gap_uses_union_minus_target(tmp_path: pathlib.Path) -> None:
    win = _write(
        tmp_path / "winner" / "trajectories.jsonl",
        [
            make_row("1", reward_type="gold_purchase", purchased=GOLD_ASIN,
                     listings=[(1, [(2, GOLD_ASIN)])], opened=[GOLD_ASIN]),
            make_row("2", reward_type="gold_purchase", purchased=GOLD_ASIN, opened=[GOLD_ASIN]),
            make_row("3", reward_type="gold_purchase", purchased=GOLD_ASIN, opened=[GOLD_ASIN]),
        ],
    )
    target = _write(
        tmp_path / "m10r-v2" / "trajectories.jsonl",
        [
            make_row("1", queries=["q"], opened=["111"]),
            make_row("2", reward_type="gold_purchase", purchased=GOLD_ASIN, opened=[GOLD_ASIN]),
            make_row("3", reward_type="gold_purchase", purchased=GOLD_ASIN, opened=[GOLD_ASIN]),
        ],
    )
    per_model, solved = collect([win, target])
    result = build_gap(per_model, solved, "m10r-v2", min_tasks=1)

    assert result["union"] == ["1", "2", "3"]
    assert len(result["target_solved"]) == 2
    assert [item["task_id"] for item in result["gap"]] == ["1"]
    item = result["gap"][0]
    assert item["category"] == "检索-查询词"
    assert item["n_winners"] == 1


def test_report_renders_checks_and_verdict(tmp_path: pathlib.Path) -> None:
    win = _write(
        tmp_path / "winner" / "trajectories.jsonl",
        [make_row("1", reward_type="gold_purchase", purchased=GOLD_ASIN,
                  listings=[(1, [(2, GOLD_ASIN)])], opened=[GOLD_ASIN])],
    )
    target = _write(tmp_path / "m10r-v2" / "trajectories.jsonl", [make_row("1", opened=["111"])])
    per_model, solved = collect([win, target])
    result = build_gap(per_model, solved, "m10r-v2", min_tasks=1)
    text = format_report(result, {"union": 1, "target": 0, "gap": 1})
    assert "口径校验" in text
    assert "检索-查询词" in text
    assert "预登记" in text
    assert "✅" in text
