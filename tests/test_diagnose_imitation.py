"""`scripts/diagnose_imitation.py` 的单测：对齐与分歧分类是判定的全部依据。"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_imitation", ROOT / "scripts" / "diagnose_imitation.py"
)
imit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(imit)


def _call(name: str, **arguments) -> dict:
    import json

    return {"function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


def _pool_row(*calls) -> dict:
    messages = [{"role": "user", "content": "Instruction: 买点东西"}]
    for call in calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        messages.append({"role": "tool", "content": "[SHOPPING_OBSERVATION_V2] ..."})
    return {"task_id": 1, "messages": messages}


def test_canon_folds_width_and_spaces() -> None:
    assert imit.canon("  水彩笔   24色 ") == "水彩笔 24色"
    assert imit.canon("ＡＢＣ") == "abc"
    assert imit.canon(None) == ""


def test_canon_args_parses_json_string() -> None:
    assert imit.canon_args('{"query": "水彩 笔"}') == {"query": "水彩 笔"}
    assert imit.canon_args({}) == {}
    assert imit.canon_args(None) == {}


def test_canon_args_keeps_unparsable_payload() -> None:
    result = imit.canon_args("not json")
    assert list(result) == ["__raw__"]


def test_teacher_actions_extracts_tool_calls_in_order() -> None:
    row = _pool_row(_call("search_products", query="a"), _call("buy_now"))
    actions = imit.teacher_actions(row)
    assert [name for name, _ in actions] == ["search_products", "buy_now"]
    assert actions[0][1] == {"query": "a"}


def test_model_actions_uses_steps() -> None:
    record = {
        "steps": [
            {"tool_name": "search_products", "parameters": {"query": "a"}},
            {"tool_name": "buy_now", "parameters": {}},
        ]
    }
    assert imit.model_actions(record) == [
        ("search_products", {"query": "a"}),
        ("buy_now", {}),
    ]


def test_first_divergence_returns_none_when_identical() -> None:
    seq = [("search_products", {"query": "a"}), ("buy_now", {})]
    assert imit.first_divergence(seq, seq) is None


def test_first_divergence_reports_earliest_index() -> None:
    teacher = [("search_products", {"query": "a"}), ("open_product", {"asin": "X"})]
    model = [("search_products", {"query": "a"}), ("open_product", {"asin": "Y"})]
    assert imit.first_divergence(teacher, model) == 1


def test_first_divergence_handles_truncation() -> None:
    teacher = [("search_products", {"query": "a"}), ("buy_now", {})]
    model = [("search_products", {"query": "a"})]
    assert imit.first_divergence(teacher, model) == 1


def test_classify_detects_tool_swap() -> None:
    teacher = [("open_product", {"asin": "X"})]
    model = [("search_products", {"query": "a"})]
    assert imit.classify(teacher, model, 0).startswith("换工具")


def test_classify_distinguishes_query_product_option() -> None:
    assert imit.classify(
        [("search_products", {"query": "a"})], [("search_products", {"query": "b"})], 0
    ) == "换查询"
    assert imit.classify(
        [("open_product", {"asin": "X"})], [("open_product", {"asin": "Y"})], 0
    ) == "换商品"
    assert imit.classify(
        [("select_option", {"value": "红"})], [("select_option", {"value": "蓝"})], 0
    ) == "换规格"


def test_classify_flags_early_stop_and_extra_turn() -> None:
    assert imit.classify([("buy_now", {})], [], 0) == "提前结束"
    assert imit.classify([], [("buy_now", {})], 0) == "多走回合"


def test_is_success_reads_purchase_success() -> None:
    ok = {"terminal_result": {"reward_detail": {"purchase_success": True}}}
    bad = {"terminal_result": {"reward_detail": {"purchase_success": False}}}
    assert imit.is_success(ok) is True
    assert imit.is_success(bad) is False
    assert imit.is_success({}) is False
