"""``scripts/analyze_candidate_exploration.py`` 的口径回归。

这里的每条断言都对应一个曾经算错的点，改脚本时先看这里：
* 搜索行必须以行首名次解析（正则漏 ``MULTILINE`` 时全部解析为 0 条）；
* 「打开过正确商品」要看商品页 ASIN 与 ``open_product`` 参数两处；
* 缺 ``goal.asin`` 的行不能丢，否则"只开 1 个商品"的比例会假性变成 100%；
* 探索上界只能算「正确商品确实出现在结果里」的失败题。
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

import scripts.analyze_candidate_exploration as ace

SEARCH_PAGE = """[SHOPPING_OBSERVATION_V2]
page_type: search_results
query: 木梳
格式: rank|asin|price|brand|category|key_attributes|title
1|111|5.0|甲|节庆|高档|A款
2|222|6.0|乙|节庆|高档|B款
3|333|7.0|丙|节庆|高档|C款
products_shown: 3

搜索功能是否可用: True
可点击的按钮: ["111", "222", "333"]
"""

PRODUCT_PAGE = """[SHOPPING_OBSERVATION_V2]
page_type: product_detail
asin: {asin}
title: 测试商品
brand: 甲
category: 节庆
price: 5.0
key_attributes: 高档
selected_options: {{}}
available_options: {{}}

搜索功能是否可用: False
可点击的按钮: ["back to search", "< prev", "buy now"]
"""


def _step(name, observation, parameters=None):
    return {
        "tool_name": name,
        "parameters": parameters or {},
        "observation": observation,
    }


def _row(task_id, reward_type, goal_asin, steps, bought_gold=False, purchased=None):
    if purchased is None:
        purchased = reward_type in {
            "gold_purchase",
            "valid_alternative_purchase",
            "partial_alternative_purchase",
            "wrong_purchase",
            "reward_unverifiable",
        }
    return {
        "task_id": task_id,
        "steps": steps,
        "terminal_result": {
            "goal": {"asin": goal_asin} if goal_asin else {},
            "purchase": {"asin": "x"} if purchased else {},
            "reward_detail": {
                "reward_type": reward_type,
                "target_asin_match": bought_gold,
            },
        },
    }


def _happy_steps(open_asin, gold_asin="222"):
    return [
        _step("search_products", SEARCH_PAGE, {"query": "木梳"}),
        _step("open_product", PRODUCT_PAGE.format(asin=open_asin), {"asin": open_asin}),
        _step("buy_now", PRODUCT_PAGE.format(asin=open_asin)),
    ]


def test_search_result_asins_parses_ranked_lines():
    assert ace.search_result_asins(SEARCH_PAGE) == ["111", "222", "333"]


def test_search_result_asins_ignores_product_pages():
    assert ace.search_result_asins(PRODUCT_PAGE.format(asin="222")) == []


def test_product_page_asin_reads_asin_line():
    assert ace.product_page_asin(PRODUCT_PAGE.format(asin="999")) == "999"


def test_gold_visible_but_not_opened_is_saw_gold_not_opened():
    row = _row(1, "partial_alternative_purchase", "222", _happy_steps("111"))
    result = ace.diagnose(row)
    assert result["bucket"] == "saw_gold_not_opened"
    assert result["gold_best_rank"] == 2
    assert result["opens"] == 1
    assert result["first_search_rank1_is_gold"] is False


def test_gold_absent_from_results_is_never_saw_gold():
    row = _row(2, "partial_alternative_purchase", "999", _happy_steps("111"))
    result = ace.diagnose(row)
    assert result["bucket"] == "never_saw_gold"
    assert result["gold_best_rank"] is None


def test_bought_gold_but_unsatisfied_is_incomplete_bucket():
    row = _row(3, "partial_alternative_purchase", "222", _happy_steps("222"), bought_gold=True)
    assert ace.diagnose(row)["bucket"] == "bought_gold_incomplete"


def test_opened_gold_but_bought_other_is_judgement_bucket():
    steps = _happy_steps("222")
    steps[2] = _step("buy_now", PRODUCT_PAGE.format(asin="111"))
    row = _row(4, "partial_alternative_purchase", "222", steps)
    assert ace.diagnose(row)["bucket"] == "opened_gold_bought_other"


def test_missing_gold_keeps_row_and_counts_opens():
    row = _row(5, "max_steps", None, _happy_steps("111"))
    result = ace.diagnose(row)
    assert result["bucket"] == "no_gold_reference"
    assert result["opens"] == 1


def test_single_open_share_counts_rows_without_gold():
    rows = [
        _row(1, "gold_purchase", "222", _happy_steps("222"), bought_gold=True),
        _row(2, "max_steps", None, _happy_steps("111")),
    ]
    summary = ace.aggregate([ace.diagnose(row) for row in rows])
    assert summary["total"] == 2
    assert summary["single_open_share"] == pytest.approx(1.0)
    assert summary["buckets"]["no_gold_reference"] == 1


def test_rank1_split_separates_hit_and_miss():
    rows = [
        _row(1, "gold_purchase", "111", _happy_steps("111"), bought_gold=True),
        _row(2, "partial_alternative_purchase", "222", _happy_steps("111")),
    ]
    summary = ace.aggregate([ace.diagnose(row) for row in rows])
    assert summary["rank1_hit"] == {"tasks": 1, "successes": 1}
    assert summary["rank1_miss"] == {"tasks": 1, "successes": 0}


def test_reachable_ceiling_only_counts_visible_gold_failures():
    rows = [
        _row(1, "gold_purchase", "111", _happy_steps("111"), bought_gold=True),
        _row(2, "partial_alternative_purchase", "222", _happy_steps("111")),
        _row(3, "partial_alternative_purchase", "999", _happy_steps("111")),
    ]
    summary = ace.aggregate([ace.diagnose(row) for row in rows])
    assert summary["reachable_by_exploration"] == 1
    assert summary["reachable_rate_ceiling"] == pytest.approx(2 / 3)


def test_report_mentions_bucket_labels_and_ceiling():
    rows = [_row(1, "partial_alternative_purchase", "222", _happy_steps("111"))]
    diagnosed = [ace.diagnose(row) for row in rows]
    summary = ace.aggregate(diagnosed)
    summary["missing_gold"] = 0
    report = ace.render_report(summary, diagnosed, "demo.jsonl", examples=3)
    assert "正确商品在结果里，却没打开" in report
    assert "上界" in report


def test_main_writes_report():
    directory = tempfile.mkdtemp()
    path = Path(directory) / "trajectories.jsonl"
    path.write_text(
        json.dumps(_row(1, "partial_alternative_purchase", "222", _happy_steps("111")))
        + "\n",
        encoding="utf-8",
    )
    out = Path(directory) / "report.md"
    try:
        assert ace.main([str(path), "--out", str(out)]) == 0
        assert "候选探索解剖" in out.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_main_reports_missing_file(capsys):
    assert ace.main([str(Path(tempfile.gettempdir()) / "no-such-trajectories.jsonl")]) == 2
    assert "不存在" in capsys.readouterr().err
