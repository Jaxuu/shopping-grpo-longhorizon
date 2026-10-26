"""Guards for the retrieval-reachability audit.

The point of this script is to keep two claims apart:

* "the gold product is not in the BM25 recall" — a lexical gap;
* "the gold product is in the recall but beyond the rendered page" — a
  behaviour gap that more ``next_page`` demonstrations can fix.

Every test below pins one of those two branches.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO_ROOT / "scripts" / "audit_retrieval_reachability.py"
    spec = importlib.util.spec_from_file_location("audit_retrieval_reachability", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _row(
    *,
    reward_type="wrong_purchase",
    gold="11111111",
    purchased=None,
    search_rows=None,
    opened=(),
    queries=("foo",),
    task_id=None,
):
    steps = []
    for query in queries:
        rows = "\n".join(f"{rank}|{asin}|title" for rank, asin in (search_rows or []))
        steps.append(
            {
                "tool_name": "search_products",
                "parameters": {"query": query},
                "observation": rows,
            }
        )
    for asin in opened:
        steps.append({"tool_name": "open_product", "parameters": {"asin": asin}})
    return {
        "task_id": task_id,
        "steps": steps,
        "terminal_result": {
            "reward_detail": {"reward_type": reward_type},
            "goal": {"asin": gold} if gold else {},
            "purchase": {"asin": purchased} if purchased else {},
        },
    }


def test_success_rows_are_not_failures():
    row = _row(reward_type="gold_purchase", purchased="11111111")
    assert audit.bucket_row(row)[0] == "success"


def test_missing_gold_reference_is_its_own_bucket():
    assert audit.bucket_row(_row(gold=None))[0] == "no_gold_reference"


def test_bought_gold_without_full_marks_is_bucket_d():
    row = _row(purchased="11111111", search_rows=[(1, "11111111")], opened=("11111111",))
    assert audit.bucket_row(row)[0] == "D_bought_gold_incomplete"


def test_opened_gold_bought_other_is_bucket_c():
    row = _row(
        purchased="22222222",
        search_rows=[(1, "11111111"), (2, "22222222")],
        opened=("11111111",),
    )
    assert audit.bucket_row(row)[0] == "C_opened_gold_bought_other"


def test_seen_but_not_opened_is_bucket_b():
    row = _row(purchased="22222222", search_rows=[(1, "11111111"), (2, "22222222")])
    assert audit.bucket_row(row)[0] == "B_seen_not_opened"


def test_absent_from_results_is_bucket_a():
    row = _row(purchased="22222222", search_rows=[(1, "22222222")])
    assert audit.bucket_row(row)[0] == "A_never_seen"


def test_search_rows_need_multiline_regex():
    """Without re.MULTILINE the parser sees one line and reports everything as A."""

    row = _row(purchased="22222222", search_rows=[(1, "11111111"), (2, "22222222")])
    bucket, info = audit.bucket_row(row)
    assert info["ranks"] == {"11111111": 1, "22222222": 2}
    assert bucket == "B_seen_not_opened"


class _FakeSearcher:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k=150):
        class Hit:
            def __init__(self, asin, rank):
                self.asin = asin
                self.rank = rank

        return [Hit(asin, rank) for asin, rank in self.hits.get(query, [])][:k]


def test_recall_rank_takes_the_best_of_several_queries():
    searcher = _FakeSearcher(
        {"a": [("99999999", 1), ("11111111", 40)], "b": [("11111111", 23)]}
    )
    assert audit.recall_rank(searcher, ["a", "b"], "11111111", 150) == 23


def test_recall_rank_returns_none_when_absent():
    searcher = _FakeSearcher({"a": [("99999999", 1)]})
    assert audit.recall_rank(searcher, ["a"], "11111111", 150) is None


def test_recall_rank_ignores_empty_queries():
    searcher = _FakeSearcher({"a": [("11111111", 5)]})
    assert audit.recall_rank(searcher, [None, "", "a"], "11111111", 150) == 5


def test_bucket_row_carries_task_id():
    row = _row(task_id=8187)
    _, info = audit.bucket_row(row)
    assert info["task_id"] == 8187


def test_pages_needed_maps_rank_to_page_flips():
    # 第 1 页（1-20）不需要翻页；第 21 名要翻 1 次；第 150 名要翻 7 次。
    assert audit.pages_needed(1, 20) == 0
    assert audit.pages_needed(20, 20) == 0
    assert audit.pages_needed(21, 20) == 1
    assert audit.pages_needed(150, 20) == 7
    assert audit.pages_needed(None, 20) is None
