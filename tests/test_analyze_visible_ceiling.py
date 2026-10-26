"""`scripts/analyze_visible_ceiling.py` 的单测。

守两条：
1. 只认 `page_type: search_results` 里的结果行，别把商品详情页的 asin 当搜索结果；
2. 「可见」取**最好名次**（同一商品可能在第 1 页和第 3 页都出现）。
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_visible_ceiling", ROOT / "scripts" / "analyze_visible_ceiling.py"
)
vc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vc)


SEARCH = """[SHOPPING_OBSERVATION_V2]
page_type: search_results
query: 木梳
Page 1 of 8 (Total results: 150; ranks 1-20)
格式: rank|asin|price|brand|category|key_attributes|title
1|831919364430|5.0 to…8.0|漫步喜|节庆用品|高档|结婚梳子
2|830660260530|6.0 to…8.0|缘来饰你|节庆用品|结婚|结婚梳子
"""

DETAIL = """[SHOPPING_OBSERVATION_V2]
page_type: product_detail
asin: 999999999999
title: 商品
price: 10.0
"""


def _record(*observations: str, gold: str | None = None, success: bool = False) -> dict:
    return {
        "steps": [{"observation": item} for item in observations],
        "terminal_result": {
            "goal": {"asin": gold} if gold else {},
            "reward_detail": {"reward_type": "gold_purchase" if success else "wrong_purchase"},
        },
    }


def test_search_rows_only_reads_search_pages() -> None:
    assert vc.search_rows(SEARCH) == [(1, "831919364430"), (2, "830660260530")]
    assert vc.search_rows(DETAIL) == []
    assert vc.search_rows("") == []


def test_visible_asins_keeps_best_rank() -> None:
    later = SEARCH.replace("1|831919364430", "15|831919364430")
    seen = vc.visible_asins(_record(SEARCH, later))
    assert seen["831919364430"] == 1


def test_visible_asins_ignores_detail_pages() -> None:
    assert vc.visible_asins(_record(DETAIL)) == {}


def test_gold_asin_reads_goal() -> None:
    assert vc.gold_asin(_record(gold="ABC")) == "ABC"
    assert vc.gold_asin({"terminal_result": {"goal": {}}}) is None
    assert vc.gold_asin({}) is None


def test_bucket_of() -> None:
    assert vc.bucket_of(1) == "1-3"
    assert vc.bucket_of(10) == "4-10"
    assert vc.bucket_of(20) == "11-20"
    assert vc.bucket_of(99) == ">60"


def test_summarize_counts_visible_and_misses() -> None:
    per_model = {
        "champ": {
            1: _record(SEARCH, gold="831919364430", success=True),  # 可见 + 买对
            2: _record(SEARCH, gold="830660260530", success=False),  # 可见 + 没买对
            3: _record(DETAIL, gold="111111111111", success=False),  # 没看见
            4: _record(DETAIL, gold=None, success=False),  # 无解，不计
        },
        "other": {
            3: _record(SEARCH.replace("831919364430", "111111111111"), gold="111111111111"),
        },
    }
    stats = vc.summarize(per_model, champion="champ")
    assert stats["solvable"] == 3
    assert stats["champion_visible"] == 2
    assert stats["champion_success"] == 1
    assert stats["champion_success_and_visible"] == 1
    assert stats["champion_visible_but_failed"] == 1
    assert stats["union_visible"] == 3
    assert stats["champion_invisible_but_others_visible"] == 1
    assert stats["ranks"] == {"1-3": 2}


def test_summarize_marks_upper_bound_as_insufficient_when_visible_is_low() -> None:
    per_model = {"champ": {1: _record(DETAIL, gold="111111111111")}}
    stats = vc.summarize(per_model, champion="champ")
    assert stats["champion_visible"] == 0
    assert stats["solvable"] == 1


def test_summarize_reports_success_by_rank_bucket() -> None:
    per_model = {
        "champ": {
            1: _record(SEARCH, gold="831919364430", success=True),
            2: _record(SEARCH, gold="830660260530", success=False),
        }
    }
    stats = vc.summarize(per_model, champion="champ")
    assert stats["bucket_stats"]["1-3"] == {"total": 2, "success": 1}
