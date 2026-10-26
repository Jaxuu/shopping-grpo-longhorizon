"""查询行为对比脚本的口径回归。

它只读轨迹、不碰 gold，所以最需要钉死的是「读得对」：
搜索步要按出现顺序取、空查询要忽略、成功率只认 gold_purchase。
"""

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_query_behaviour", ROOT / "scripts" / "compare_query_behaviour.py"
)
qb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qb)

compare_records = qb.compare_records
is_success = qb.is_success
render_markdown = qb.render_markdown
search_queries = qb.search_queries
task_terms = qb.task_terms
term_count = qb.term_count


def _record(task_id=1, queries=("木梳",), reward_type="gold_purchase"):
    steps = [
        {"tool_name": "search_products", "parameters": {"query": query}} for query in queries
    ]
    return {
        "task_id": task_id,
        "steps": steps,
        "terminal_result": {"reward_detail": {"reward_type": reward_type}},
    }


def test_search_queries_keeps_order_and_skips_blanks():
    record = _record(queries=("a b", "", "c"))
    assert search_queries(record) == ["a b", "c"]


def test_search_queries_ignores_other_tools():
    record = {
        "steps": [
            {"tool_name": "open_product", "parameters": {"asin": "1"}},
            {"tool_name": "search_products", "parameters": {"query": "x"}},
        ]
    }
    assert search_queries(record) == ["x"]


def test_term_count_splits_on_whitespace():
    assert term_count("女士单肩包 黑色 简约") == 3
    assert term_count("") == 0


def test_task_terms_is_union_across_searches():
    assert task_terms(_record(queries=("a b", "b c"))) == {"a", "b", "c"}


def test_is_success_only_counts_gold_purchase():
    assert is_success(_record(reward_type="gold_purchase")) is True
    assert is_success(_record(reward_type="valid_alternative_purchase")) is False


def test_compare_detects_longer_query_and_fixed_task():
    base = {1: _record(queries=("古筝 紫色",), reward_type="wrong_purchase")}
    variant = {1: _record(queries=("古筝 紫色 檀木 玉饰",), reward_type="gold_purchase")}
    summary = compare_records(base, variant)
    assert summary["tasks"] == 1
    assert summary["tasks_query_changed"] == 1
    assert summary["tasks_first_query_longer"] == 1
    assert summary["mean_first_query_terms"] == {"base": 2.0, "variant": 4.0}
    assert summary["outcome"] == {"fixed": 1, "broke": 0, "same": 0}


def test_compare_only_uses_shared_tasks():
    base = {1: _record(1), 2: _record(2)}
    variant = {2: _record(2)}
    summary = compare_records(base, variant)
    assert summary["tasks"] == 1
    assert summary["rows"][0]["task_id"] == 2


def test_unchanged_queries_are_reported_as_unchanged():
    base = {1: _record(queries=("木梳",))}
    variant = {1: _record(queries=("木梳",))}
    summary = compare_records(base, variant)
    assert summary["tasks_query_changed"] == 0
    assert summary["mean_term_jaccard"] == 1.0


def test_render_flags_prompt_ignored_when_nothing_changed():
    base = {1: _record(queries=("木梳",))}
    variant = {1: _record(queries=("木梳",))}
    text = render_markdown(compare_records(base, variant))
    assert "这次实验没有测到提示词" in text


def test_render_flags_bottleneck_when_queries_did_change():
    base = {1: _record(queries=("古筝 紫色",), reward_type="wrong_purchase")}
    variant = {1: _record(queries=("古筝 紫色 檀木",), reward_type="wrong_purchase")}
    text = render_markdown(compare_records(base, variant))
    assert "瓶颈不在查询质量" in text
    assert "| 1 | same |" not in text  # 结果没变的题不进明细表


def test_cli_round_trip(tmp_path):
    base_path = tmp_path / "base.jsonl"
    variant_path = tmp_path / "variant.jsonl"
    base_path.write_text(
        json.dumps(_record(1, ("a b",), "wrong_purchase"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    variant_path.write_text(
        json.dumps(_record(1, ("a b c",), "gold_purchase"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.md"
    as_json = tmp_path / "out.json"
    assert (
        qb.main(["--base", str(base_path), "--variant", str(variant_path),
                 "--output", str(out), "--json", str(as_json)])
        == 0
    )
    assert out.exists() and as_json.exists()
    assert json.loads(as_json.read_text(encoding="utf-8"))["outcome"]["fixed"] == 1
