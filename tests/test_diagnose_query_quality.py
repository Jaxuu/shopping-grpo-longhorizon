"""`scripts/diagnose_query_quality.py` 的单测：抽首句查询与名次汇总。"""

from __future__ import annotations

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_query_quality", ROOT / "scripts" / "diagnose_query_quality.py"
)
qq = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qq)


def test_first_query_teacher_skips_non_search_tools() -> None:
    row = {
        "messages": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "view_features", "arguments": "{}"}}]},
            {"role": "assistant", "tool_calls": [{"function": {"name": "search_products", "arguments": '{"query": "木梳 礼盒"}'}}]},
            {"role": "assistant", "tool_calls": [{"function": {"name": "search_products", "arguments": '{"query": "第二次"}'}}]},
        ]
    }
    assert qq.first_query_teacher(row) == "木梳 礼盒"


def test_first_query_teacher_handles_bad_json() -> None:
    row = {"messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "search_products", "arguments": "{bad"}}]}]}
    assert qq.first_query_teacher(row) is None


def test_first_query_model_reads_steps() -> None:
    record = {
        "steps": [
            {"tool_name": "open_product", "parameters": {"asin": "X"}},
            {"tool_name": "search_products", "parameters": {"query": "水彩笔 24色"}},
        ]
    }
    assert qq.first_query_model(record) == "水彩笔 24色"
    assert qq.first_query_model({"steps": []}) is None


def test_top_page_uses_rendered_page_size() -> None:
    assert qq.top_page(1) is True
    assert qq.top_page(20) is True
    assert qq.top_page(21) is False
    assert qq.top_page(None) is False


def test_summarize_counts_page_hits_and_missing() -> None:
    stats = qq.summarize([1, 20, 21, None, 5])
    assert stats["n"] == 5
    assert stats["in_page"] == 3
    assert stats["missing"] == 1
    # 找到的名次排序后是 [1, 5, 20, 21]，取上中位 = 20
    assert stats["median"] == 20


def test_summarize_handles_all_missing() -> None:
    stats = qq.summarize([None, None])
    assert stats["in_page"] == 0
    assert stats["median"] is None
    assert stats["missing"] == 2


class _Hit:
    def __init__(self, asin: str, rank: int) -> None:
        self.asin = asin
        self.rank = rank


def test_rank_for_returns_none_on_empty_query() -> None:
    class _Never:
        def search(self, query: str, k: int):
            raise AssertionError("空查询不应该调用检索")

    assert qq.rank_for(_Never(), None, "GOLD") is None
    assert qq.rank_for(_Never(), "", "GOLD") is None


def test_rank_for_finds_gold_rank() -> None:
    class _Searcher:
        def search(self, query: str, k: int):
            return [_Hit("A", 1), _Hit("GOLD", 2), _Hit("B", 3)]

    assert qq.rank_for(_Searcher(), "x", "GOLD") == 2
    assert qq.rank_for(_Searcher(), "x", "MISSING") is None


def test_is_success_reads_purchase_success() -> None:
    ok = {"terminal_result": {"reward_detail": {"purchase_success": True}}}
    assert qq.is_success(ok) is True
    assert qq.is_success({"terminal_result": {"reward_detail": {"reward_type": "gold_purchase"}}}) is True
    assert qq.is_success({}) is False


def test_read_jsonl_skips_blank_lines(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"task_id": 1}\n\n{"task_id": 2}\n', encoding="utf-8")
    assert [row["task_id"] for row in qq.read_jsonl(path)] == [1, 2]


def test_pool_and_run_schemas_line_up(tmp_path: pathlib.Path) -> None:
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps({"task_id": 7, "messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "search_products", "arguments": '{"query": "q"}'}}]}]})
        + "\n",
        encoding="utf-8",
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "trajectories.jsonl").write_text(
        json.dumps({"task_id": 7, "steps": [{"tool_name": "search_products", "parameters": {"query": "q"}}]}) + "\n",
        encoding="utf-8",
    )
    assert qq.first_query_teacher(qq.read_jsonl(pool)[0]) == "q"
    assert qq.first_query_model(qq.read_jsonl(run / "trajectories.jsonl")[0]) == "q"
