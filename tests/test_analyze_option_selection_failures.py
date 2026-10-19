"""守护「买对商品却没满分」的统计口径。

最容易被写错的两处，都在 2026-10-19 真出过事：
1. 把「买了别的商品」的行也算进来 —— 正确商品的规格轴根本不在页面上，
   那样算出来的「模型没选对规格」是循环论证。
2. 不区分「要求值在不在页面上」—— 会把「没机会」当成「没能力」。
"""

import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_option_selection_failures.py"
)
_spec = importlib.util.spec_from_file_location("analyze_option_selection", SCRIPT)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)

GOLD = "831919364430"
OTHER = "11111111"


def _trajectory(reward_type, bought, observation="", options=None):
    """构造一条最小轨迹：只带 terminal_result 与一条 product_detail 观测。"""
    dimensions = {
        "brand": {"active": False, "score": 0.0},
        "core_functions": {"active": True, "score": 1.0},
        "key_options": {
            "active": True,
            "score": 0.0 if options else 1.0,
            "results": options or [],
        },
    }
    return {
        "task_id": 1,
        "steps": [
            {
                "tool_name": "open_product",
                "observation": (
                    f"[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\n"
                    f"asin: {GOLD}\n{observation}"
                ),
            }
        ],
        "terminal_result": {
            "goal": {"asin": GOLD},
            "purchase": {"asin": bought},
            "reward_detail": {
                "reward_type": reward_type,
                "evidence": {"preference_scoring": {"dimensions": dimensions}},
            },
        },
    }


def _option(required, selected):
    return {
        "passed": False,
        "required": {"color": {"value": required}},
        "evidence": {"wrong_values": [{"axis": "color", "selected": selected}]},
    }


def test_success_rows_are_skipped():
    row = _trajectory("gold_purchase", GOLD, "颜色分类: 红", [_option("红", "蓝")])
    assert module.diagnose([row]) == []


def test_rows_that_bought_another_product_are_skipped():
    """这是 2026-10-19 撤回结论的那个错：买别的商品时没有机会选对规格。"""
    row = _trajectory("wrong_purchase", OTHER, "颜色分类: 红", [_option("红", "蓝")])
    assert module.diagnose([row]) == []


def test_visibility_verbatim():
    row = _trajectory("partial_alternative_purchase", GOLD, "颜色分类: 红", [_option("红", "蓝")])
    check = module.diagnose([row])[0]["option_checks"][0]
    assert check["visibility"] == "verbatim"


def test_visibility_normalized_handles_case_and_separators():
    """页面上写的是 `145升高配版 | pcm钢胆`，要求写的是 `145升高配版/PCM钢胆`。"""
    row = _trajectory(
        "partial_alternative_purchase",
        GOLD,
        "颜色分类: 145升高配版 | pcm钢胆 | 减霜80%",
        [_option("145升高配版/PCM钢胆/减霜80%", "301升高配版 | pcm钢胆")],
    )
    check = module.diagnose([row])[0]["option_checks"][0]
    assert check["visibility"] == "normalized"


def test_visibility_absent_when_value_never_rendered():
    row = _trajectory(
        "partial_alternative_purchase", GOLD, "颜色分类: 蓝", [_option("红", "蓝")]
    )
    check = module.diagnose([row])[0]["option_checks"][0]
    assert check["visibility"] == "absent"


def test_summary_counts_only_key_options_and_visibility():
    rows = [
        _trajectory("partial_alternative_purchase", GOLD, "颜色分类: 红", [_option("红", "蓝")]),
        _trajectory("partial_alternative_purchase", GOLD, "颜色分类: 蓝", [_option("红", "蓝")]),
    ]
    summary = module.summarize(module.diagnose(rows))
    assert summary["tasks"] == 2
    assert summary["only_key_options"] == 2
    assert summary["option_visibility_counts"] == {"verbatim": 1, "absent": 1}


def test_render_reports_reachable_ratio():
    row = _trajectory("partial_alternative_purchase", GOLD, "颜色分类: 红", [_option("红", "蓝")])
    items = module.diagnose([row])
    report = module.render(items, module.summarize(items))
    assert "**1/1**" in report
    assert "verbatim" in report


def test_cli_writes_report():
    row = _trajectory("partial_alternative_purchase", GOLD, "颜色分类: 红", [_option("红", "蓝")])
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "t.jsonl"
        src.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        out = Path(tmp) / "report.md"
        module.main([str(src), "--out", str(out)])
        text = out.read_text(encoding="utf-8")
        assert "买对了商品却没满分" in text
        assert "task 1" in text


def test_cli_json_mode():
    row = _trajectory("partial_alternative_purchase", GOLD, "颜色分类: 红", [_option("红", "蓝")])
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "t.jsonl"
        src.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            module.main([str(src), "--json"])
        payload = json.loads(buffer.getvalue())
        assert payload["summary"]["tasks"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
