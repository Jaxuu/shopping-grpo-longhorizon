"""`scripts/filter_m8_verification.py` 的语义固定。

钉死四件事（改动前先看这里）：

1. **只收 `gold_purchase`**——它已隐含 `key_options == 1.0` 与 `coverage == 1.0`。
2. **`select_option` 次数必须 ≥ 要求轴数**——过程看不出来的规格选择对 SFT 没价值。
3. **不用 `open_product ≥ 2`**——实测它会误杀绝大多数 gold 轨迹。
4. **每题有保留上限**，避免采集结果被少数题占满。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import filter_m8_verification as module  # noqa: E402


def trajectory(
    task_id: object = 1,
    reward_type: str = "gold_purchase",
    *,
    selects: int = 2,
    required: int = 2,
    opens: int = 1,
    with_detail: bool = True,
    with_option_dimension: bool = True,
) -> dict:
    steps = [{"tool_name": "search_products", "parameters": {"query": "q"}}]
    steps += [
        {"tool_name": "open_product", "parameters": {"asin": f"A{i}"}}
        for i in range(opens)
    ]
    steps += [
        {"tool_name": "select_option", "parameters": {"value": f"v{i}"}}
        for i in range(selects)
    ]
    steps.append({"tool_name": "buy_now", "parameters": {}})

    dimensions: dict = {}
    if with_option_dimension:
        dimensions["key_options"] = {"active": True, "required_count": required}
    detail = {
        "reward_type": reward_type,
        "evidence": {"preference_scoring": {"dimensions": dimensions}},
    }
    payload: dict = {"task_id": task_id, "steps": steps, "messages": []}
    if with_detail:
        payload["terminal_result"] = {"reward_detail": detail}
    return payload


def inspect(**kwargs) -> dict:
    return module.inspect(trajectory(**kwargs))


class InspectTest(unittest.TestCase):
    def test_gold_with_complete_selection_passes(self) -> None:
        record = inspect(selects=2, required=2)
        self.assertEqual(record["reason"], "")
        self.assertEqual(record["selects"], 2)
        self.assertEqual(record["required"], 2)

    def test_extra_selections_still_pass(self) -> None:
        """选完又改主意会多调用几次，不应被拒。"""
        self.assertEqual(inspect(selects=4, required=2)["reason"], "")

    def test_non_gold_is_rejected(self) -> None:
        self.assertEqual(
            inspect(reward_type="partial_alternative_purchase")["reason"],
            module.REASON_NOT_GOLD,
        )

    def test_incomplete_selection_is_rejected(self) -> None:
        self.assertEqual(
            inspect(selects=1, required=2)["reason"],
            module.REASON_SELECTS_INCOMPLETE,
        )

    def test_task_without_option_requirement_is_rejected(self) -> None:
        """该题不要求规格 ⇒ 教不了「选规格」，即便买对了也不收。"""
        self.assertEqual(
            inspect(required=0)["reason"], module.REASON_NO_OPTION_REQUIREMENT
        )

    def test_missing_detail_is_rejected(self) -> None:
        self.assertEqual(inspect(with_detail=False)["reason"], module.REASON_NO_DETAIL)

    def test_single_open_is_not_a_rejection_reason(self) -> None:
        """`open_product` 只有 1 次也必须通过——原方案的 open≥2 是误杀。"""
        self.assertEqual(inspect(opens=1, selects=2, required=2)["reason"], "")


class SelectTest(unittest.TestCase):
    def test_caps_rows_per_task(self) -> None:
        rows = [trajectory(task_id=7) for _ in range(5)]
        result = module.select(rows, max_per_task=2)
        self.assertEqual(len(result["kept"]), 2)
        self.assertEqual(result["rejected"][module.REASON_OVER_TASK_CAP], 3)

    def test_zero_cap_keeps_everything(self) -> None:
        rows = [trajectory(task_id=7) for _ in range(5)]
        result = module.select(rows, max_per_task=0)
        self.assertEqual(len(result["kept"]), 5)

    def test_counts_rejections_by_reason(self) -> None:
        rows = [
            trajectory(),
            trajectory(reward_type="wrong_purchase"),
            trajectory(selects=1),
            trajectory(required=0),
        ]
        result = module.select(rows, max_per_task=0)
        self.assertEqual(len(result["kept"]), 1)
        self.assertEqual(result["rejected"][module.REASON_NOT_GOLD], 1)
        self.assertEqual(result["rejected"][module.REASON_SELECTS_INCOMPLETE], 1)
        self.assertEqual(result["rejected"][module.REASON_NO_OPTION_REQUIREMENT], 1)


class ReportTest(unittest.TestCase):
    def test_report_lists_reasons_and_gates(self) -> None:
        rows = [trajectory(task_id=i) for i in range(90)]
        rows.append(trajectory(task_id=999, reward_type="wrong_purchase"))
        result = module.select(rows, max_per_task=0)
        text = module.render_report(result, source="demo.jsonl", max_per_task=0)
        self.assertIn("拒收原因", text)
        self.assertIn(module.REASON_NOT_GOLD, text)
        self.assertIn("可收条数 ≥ 80", text)
        self.assertIn("平均工具步数 ≤ 14", text)

    def test_gate_fails_below_threshold(self) -> None:
        result = module.select([trajectory()], max_per_task=0)
        text = module.render_report(result, source="demo.jsonl", max_per_task=0)
        self.assertIn("❌", text)


class CliTest(unittest.TestCase):
    def test_writes_sft_rows_with_four_keys(self) -> None:
        directory = tempfile.mkdtemp()
        source = Path(directory) / "raw.jsonl"
        out = Path(directory) / "m8.jsonl"
        report = Path(directory) / "m8.md"
        with source.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(trajectory(task_id=1), ensure_ascii=False) + "\n")
            handle.write(
                json.dumps(trajectory(task_id=2, reward_type="wrong_purchase"), ensure_ascii=False)
                + "\n"
            )
        code = module.main(
            [str(source), "--out", str(out), "--report", str(report)]
        )
        self.assertEqual(code, 0)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {"trajectory_id", "task_id", "messages", "tools"})
        self.assertTrue(report.exists())

    def test_missing_input_returns_two(self) -> None:
        directory = tempfile.mkdtemp()
        code = module.main(
            [str(Path(directory) / "nope.jsonl"), "--out", str(Path(directory) / "o.jsonl")]
        )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
