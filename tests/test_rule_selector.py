"""`scripts/rule_selector.py` 的单测。

守三条：
1. 解析要对（预算/价格/品牌/规格），错了整条规则就废；
2. 硬约束真的能一票否决（超预算、品牌不符）；
3. **同分取票数高的**——结果必须确定，不能靠字典顺序。
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rule_selector", ROOT / "scripts" / "rule_selector.py"
)
rs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rs)


DETAIL = """[SHOPPING_OBSERVATION_V2]
page_type: product_detail
asin: 831919364430
title: 结婚梳子一对红色木梳礼盒装婚礼新娘陪嫁喜梳子高档婚庆用品
brand: 漫步喜旗舰店
category: 节庆用品/礼品›婚庆用品›新娘配件
price: 5.0 to 18.0
key_attributes: 高档, 结婚, 陪嫁, 卡通-永结同心
selected_options: {}
available_options: {"颜色分类": ["【永结同心】2个装", "【郎才女貌】2个装"]}
"""


def test_parse_detail_reads_fields() -> None:
    fields = rs.parse_detail(DETAIL)
    assert fields["asin"] == "831919364430"
    assert fields["price"] == "5.0 to 18.0"
    assert "永结同心" in fields["key_attributes"]


def test_parse_detail_requires_title() -> None:
    assert rs.parse_detail("page_type: search_results\nquery: 木梳\n") == {}
    assert rs.parse_detail("") == {}


def test_parse_price_handles_range_and_single() -> None:
    assert rs.parse_price("5.0 to 18.0") == (5.0, 18.0)
    assert rs.parse_price("27.0") == (27.0, 27.0)
    assert rs.parse_price("") is None


def test_parse_budget_hard_ceiling() -> None:
    assert rs.parse_budget("预算在400元以内") == (0.0, 400.0)
    assert rs.parse_budget("价格预算在100元以内") == (0.0, 100.0)


def test_parse_budget_soft_ceiling_gets_slack() -> None:
    low, high = rs.parse_budget("价格在20元左右")
    assert low == 0.0
    assert high == 30.0


def test_parse_budget_context_without_unit_words() -> None:
    assert rs.parse_budget("预算70元，给我推荐一款") == (0.0, 70.0)


def test_parse_budget_missing() -> None:
    assert rs.parse_budget("想买个木梳") is None


def test_brand_core_strips_shop_suffix() -> None:
    assert rs.brand_core("漫步喜旗舰店") == "漫步喜"
    assert rs.brand_core("亿力") == "亿力"


def test_brand_mentioned_finds_prefix() -> None:
    assert rs.brand_mentioned("亿力", "想买个专业去污的亿力牌清洁机") is True
    assert rs.brand_mentioned("漫步喜旗舰店", "求一对卡通-永结同心款的高档木梳") is False
    assert rs.brand_mentioned("荷兰", "荷兰品牌的，可折叠") is True


def test_option_values_flattens_json() -> None:
    values = rs.option_values('{"颜色分类": ["A", "B"]}')
    assert values == ["A", "B"]
    assert rs.option_values("not json") == []
    assert rs.option_values("") == []


def test_overlap_score_rewards_longer_matches() -> None:
    instruction = "求一对卡通-永结同心款的高档木梳"
    long_score, long_pieces = rs.overlap_score("卡通-永结同心", instruction)
    short_score, _ = rs.overlap_score("木梳", instruction)
    assert "卡通-永结同心" in long_pieces
    assert long_score > short_score


def test_overlap_score_ignores_unrelated_text() -> None:
    score, pieces = rs.overlap_score("完全无关的字符串", "买一把梳子")
    assert score == 0.0
    assert pieces == []


def test_over_budget_candidate_is_penalised() -> None:
    instruction = "预算在10元以内的木梳"
    candidates = [
        {"detail": rs.parse_detail(DETAIL), "options": {}},
        {
            "detail": {
                "title": "木梳",
                "brand": "别家",
                "price": "50.0 to 60.0",
                "category": "",
                "key_attributes": "",
                "available_options": "",
            },
            "options": {},
        },
    ]
    scored = rs.score_candidates(instruction, candidates)
    assert scored[1]["score"] < scored[0]["score"]
    assert any("超预算" in reason for reason in scored[1]["reasons"])


def test_brand_mismatch_is_penalised() -> None:
    instruction = "想买个亿力牌清洁机"
    good = {"detail": {"title": "亿力清洁机", "brand": "亿力", "price": "100"}, "options": {}}
    bad = {"detail": {"title": "清洁机", "brand": "别家", "price": "100"}, "options": {}}
    scored = rs.score_candidates(instruction, [good, bad])
    assert scored[1]["score"] < scored[0]["score"]
    assert any("品牌不符" in reason for reason in scored[1]["reasons"])


def test_select_breaks_ties_by_vote_order() -> None:
    candidates = [{"key": "HIGH_VOTES"}, {"key": "LOW_VOTES"}]
    scored = [{"score": 5.0}, {"score": 5.0}]
    assert rs.select(candidates, scored) == 0


def test_select_takes_higher_score() -> None:
    candidates = [{"key": "A"}, {"key": "B"}]
    assert rs.select(candidates, [{"score": 1.0}, {"score": 9.0}]) == 1


def test_run_picks_the_matching_candidate() -> None:
    instruction = "求一对卡通-永结同心款的高档木梳，预算20元左右"
    tasks = [{"task_id": 1, "candidates": ["GOOD|", "BAD|"], "good_keys": ["GOOD|"]}]
    details = {
        (1, "GOOD"): DETAIL,
        (1, "BAD"): "title: 塑料梳子\nbrand: 杂牌\ncategory: 家居\nprice: 90.0\n",
    }
    results = rs.run(tasks, details=details, instructions={1: instruction}, top_k=2)
    assert len(results) == 1
    assert results[0]["picked"] == "GOOD|"
    assert results[0]["hit"] is True


def test_run_skips_tasks_without_detail() -> None:
    tasks = [{"task_id": 1, "candidates": ["A|", "B|"], "good_keys": ["A|"]}]
    assert rs.run(tasks, details={}, instructions={1: "x"}, top_k=2) == []
