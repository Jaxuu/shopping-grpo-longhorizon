"""`scripts/analyze_selection_vs_retrieval.py` 的语义固定。

钉死四件事（改动前先看这里）：

1. **正确商品必须从 `gold_purchase` 轨迹反推**（`reward_detail` 不落 gold ASIN），
   且同题出现两个不同 gold ASIN 时必须报 `ambiguous_gold` 而不是随便挑一个。
2. **`valid_alternative_purchase` 算成功**，不能进失败桶（否则会虚增「没看到」）。
3. **三桶互斥**：看到并买了别的 / 看到但没下单 / 从没看到。
4. **判据是「看到过」占失败尝试的比例**，三档阈值必须与报告文本一致。
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

import analyze_selection_vs_retrieval as module  # noqa: E402


def row(
    task_id: object,
    reward_type: str,
    *,
    opened: tuple[str, ...] = (),
    buy: bool = True,
    match_score: float = 0.4,
    searches: int = 1,
    nested: bool = False,
    target_asin_match: bool = False,
    budget_status: str = "pass",
    category_status: str | None = None,
    selects: int = 0,
    compact: bool = False,
) -> dict:
    steps = [
        {"tool_name": "search_products", "parameters": {"query": "q"}}
        for _ in range(searches)
    ]
    steps += [{"tool_name": "open_product", "parameters": {"asin": asin}} for asin in opened]
    steps += [
        {"tool_name": "select_option", "parameters": {"value": f"v{i}"}}
        for i in range(selects)
    ]
    if buy:
        steps.append({"tool_name": "buy_now", "parameters": {}})
    if compact:
        # 精简形态（process_evidence）：状态是裸字符串，且不带 required/actual。
        gates: dict = {"budget": budget_status}
        if category_status is not None:
            gates["category"] = category_status
    else:
        # 完整形态（RewardResult.hard_gates）：每个门是带 status/required/actual 的 dict。
        gates = {
            "budget": {
                "status": budget_status,
                "required": 30.0,
                "actual": 40.0 if budget_status == "fail" else 25.0,
            }
        }
        if category_status is not None:
            gates["category"] = {"status": category_status}
    detail = {
        "reward_type": reward_type,
        "weighted_score": match_score,
        "target_asin_match": target_asin_match,
        "hard_gates": gates,
    }
    payload = {"task_id": task_id, "steps": steps}
    if nested:
        payload["terminal_result"] = {"reward_detail": detail}
    else:
        payload["reward_detail"] = detail
    return payload


class ExtractTest(unittest.TestCase):
    def test_purchased_is_last_open_before_buy(self) -> None:
        record = module.extract(row(1, "wrong_purchase", opened=("A", "B")))
        self.assertEqual(record["purchased_asin"], "B")
        self.assertEqual(record["opened"], ["A", "B"])

    def test_no_purchase_yields_none(self) -> None:
        record = module.extract(row(1, "max_steps", opened=("A",), buy=False))
        self.assertIsNone(record["purchased_asin"])

    def test_nested_reward_detail_is_read(self) -> None:
        record = module.extract(row(1, "gold_purchase", opened=("A",), nested=True))
        self.assertEqual(record["reward_type"], "gold_purchase")

    def test_full_gate_shape_reads_fail(self) -> None:
        record = module.extract(row(1, "wrong_purchase", opened=("A",), budget_status="fail"))
        self.assertEqual(record["budget_status"], "fail")
        self.assertTrue(record["budget_failed"])
        self.assertTrue(record["budget_price_known"])

    def test_compact_gate_shape_reads_fail(self) -> None:
        """精简形态的状态是裸字符串，不能被当成「没有门槛」。"""
        record = module.extract(
            row(1, "wrong_purchase", opened=("A",), budget_status="fail", compact=True)
        )
        self.assertEqual(record["budget_status"], "fail")
        self.assertTrue(record["budget_failed"])
        # 精简形态不带 required/actual，超支幅度不可算，必须显式标记。
        self.assertFalse(record["budget_price_known"])

    def test_compact_gate_shape_reads_pass(self) -> None:
        record = module.extract(
            row(1, "wrong_purchase", opened=("A",), compact=True)
        )
        self.assertEqual(record["budget_status"], "pass")
        self.assertFalse(record["budget_failed"])

    def test_missing_gate_is_not_counted_as_failure(self) -> None:
        payload = row(1, "max_steps", opened=("A",), buy=False)
        payload["reward_detail"]["hard_gates"] = {}
        record = module.extract(payload)
        self.assertEqual(record["budget_status"], "")
        self.assertFalse(record["budget_failed"])

    def test_valid_alternative_counts_as_success(self) -> None:
        self.assertTrue(module.is_success("valid_alternative_purchase"))
        self.assertTrue(module.is_success("gold_purchase"))
        self.assertFalse(module.is_success("partial_alternative_purchase"))


class DiagnoseTest(unittest.TestCase):
    def group(self, *rows) -> list[dict]:
        return [module.extract(item) for item in rows]

    def test_saw_gold_bought_other(self) -> None:
        result = module.diagnose_group(
            self.group(
                row(1, "gold_purchase", opened=("GOLD",)),
                row(1, "wrong_purchase", opened=("GOLD", "OTHER"), match_score=0.5),
            )
        )
        self.assertEqual(result["gold_asin"], "GOLD")
        self.assertEqual(result["failures"], 1)
        self.assertEqual(result["saw_gold_bought_other"], 1)
        self.assertEqual(result["detail"][0]["bucket"], "saw_gold_bought_other")

    def test_never_saw_gold(self) -> None:
        result = module.diagnose_group(
            self.group(
                row(1, "gold_purchase", opened=("GOLD",)),
                row(1, "wrong_purchase", opened=("OTHER",)),
            )
        )
        self.assertEqual(result["never_saw"], 1)
        self.assertEqual(result["detail"][0]["bucket"], "never_saw_gold")

    def test_saw_gold_without_purchase(self) -> None:
        result = module.diagnose_group(
            self.group(
                row(1, "gold_purchase", opened=("GOLD",)),
                row(1, "max_steps", opened=("GOLD",), buy=False),
            )
        )
        self.assertEqual(result["saw_gold_no_purchase"], 1)

    def test_success_rows_are_not_failures(self) -> None:
        result = module.diagnose_group(
            self.group(
                row(1, "gold_purchase", opened=("GOLD",)),
                row(1, "valid_alternative_purchase", opened=("ALT",)),
            )
        )
        self.assertEqual(result["failures"], 0)

    def test_no_gold_row_is_undiagnosable(self) -> None:
        result = module.diagnose_group(
            self.group(
                row(1, "wrong_purchase", opened=("A",)),
                row(1, "wrong_purchase", opened=("B",)),
            )
        )
        self.assertIsNone(result)

    def test_ambiguous_gold_is_reported_not_guessed(self) -> None:
        result = module.diagnose_group(
            self.group(
                row(1, "gold_purchase", opened=("G1",)),
                row(1, "gold_purchase", opened=("G2",)),
            )
        )
        self.assertEqual(result["status"], "ambiguous_gold")
        self.assertEqual(result["gold_asins"], ["G1", "G2"])

    def test_target_asin_match_supplies_gold_without_gold_row(self) -> None:
        """整组都因超预算被判 wrong_purchase，但仍能从 target_asin_match 反推 gold。

        漏掉这个来源会把「买对了商品却超支」的整批组跳过，而那是最容易修的一类。
        """
        result = module.diagnose_group(
            self.group(
                row(1, "wrong_purchase", opened=("G",), target_asin_match=True,
                    match_score=1.0, budget_status="fail"),
                row(1, "wrong_purchase", opened=("X",), match_score=0.3,
                    budget_status="fail"),
            )
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["gold_asin"], "G")
        self.assertEqual(result["bought_gold_failed_gate"], 1)
        self.assertEqual(result["never_saw"], 1)

    def test_bought_gold_counts_as_saw(self) -> None:
        result = module.diagnose_group(
            self.group(
                row(1, "gold_purchase", opened=("G",)),
                row(1, "wrong_purchase", opened=("G",), target_asin_match=True,
                    budget_status="fail"),
            )
        )
        self.assertEqual(result["saw_gold"], 1)
        self.assertEqual(result["never_saw"], 0)


class CollectTest(unittest.TestCase):
    def test_counts_skipped_and_ambiguous(self) -> None:
        groups = {
            "1": [
                module.extract(row(1, "gold_purchase", opened=("G",))),
                module.extract(row(1, "wrong_purchase", opened=("G", "X"))),
            ],
            "2": [
                module.extract(row(2, "wrong_purchase", opened=("A",))),
                module.extract(row(2, "wrong_purchase", opened=("B",))),
            ],
            "3": [
                module.extract(row(3, "gold_purchase", opened=("G1",))),
                module.extract(row(3, "gold_purchase", opened=("G2",))),
            ],
        }
        collected = module.collect(groups, min_group_size=2)
        self.assertEqual(collected["groups"], 1)
        self.assertEqual(collected["skipped_no_gold"], 1)
        self.assertEqual(collected["ambiguous_gold"], 1)


class VerdictTest(unittest.TestCase):
    def summary(self, saw: int, total: int) -> dict:
        return {"failures": total, "saw_share": saw / total if total else None}

    def test_high_share_is_selection_bound(self) -> None:
        self.assertEqual(
            module.verdict(self.summary(8, 10), high=0.60, low=0.30)["status"],
            "selection_bound",
        )

    def test_low_share_is_retrieval_bound(self) -> None:
        self.assertEqual(
            module.verdict(self.summary(2, 10), high=0.60, low=0.30)["status"],
            "retrieval_bound",
        )

    def test_middle_is_mixed(self) -> None:
        self.assertEqual(
            module.verdict(self.summary(4, 10), high=0.60, low=0.30)["status"],
            "mixed",
        )

    def test_empty_is_not_measured(self) -> None:
        self.assertEqual(
            module.verdict(self.summary(0, 0), high=0.60, low=0.30)["status"],
            "not_measured",
        )


class AggregateGateTest(unittest.TestCase):
    """门槛口径自检：两种形态都要能被计进「超预算」。"""

    def aggregate(self, *rows) -> dict:
        group = module.diagnose_group([module.extract(item) for item in rows])
        return module.aggregate([group])

    def test_compact_shape_counts_toward_budget_failed(self) -> None:
        summary = self.aggregate(
            row(1, "gold_purchase", opened=("GOLD",)),
            row(
                1, "wrong_purchase", opened=("GOLD",),
                budget_status="fail", compact=True,
            ),
        )
        self.assertEqual(summary["budget_failed_total"], 1)
        self.assertEqual(summary["gate_readable"], 1)
        # 精简形态拿不到 required/actual，必须如实报 0。
        self.assertEqual(summary["gate_price_known"], 0)
        self.assertEqual(summary["gate_unexplained"], 0)

    def test_full_shape_reports_price_known(self) -> None:
        summary = self.aggregate(
            row(1, "gold_purchase", opened=("GOLD",)),
            row(1, "wrong_purchase", opened=("GOLD",), budget_status="fail"),
        )
        self.assertEqual(summary["budget_failed_total"], 1)
        self.assertEqual(summary["gate_price_known"], 1)

    def test_wrong_purchase_without_gate_fail_is_flagged(self) -> None:
        """判定树里 wrong_purchase 必须至少一个硬门 fail；不满足即为解析漏计。"""
        summary = self.aggregate(
            row(1, "gold_purchase", opened=("GOLD",)),
            row(1, "wrong_purchase", opened=("GOLD",), compact=True),
        )
        self.assertEqual(summary["budget_failed_total"], 0)
        self.assertEqual(summary["gate_unexplained"], 1)

    def test_non_purchase_failure_is_not_flagged(self) -> None:
        summary = self.aggregate(
            row(1, "gold_purchase", opened=("GOLD",)),
            row(1, "max_steps", opened=("OTHER",), buy=False, compact=True),
        )
        self.assertEqual(summary["gate_unexplained"], 0)

    def test_bought_gold_splits_into_gate_failed_and_short(self) -> None:
        """买对 gold 却判失败：硬门没过 与 只拿到 partial 必须分开计数。"""
        summary = self.aggregate(
            row(1, "gold_purchase", opened=("GOLD",)),
            row(1, "wrong_purchase", opened=("GOLD",), budget_status="fail"),
            row(1, "partial_alternative_purchase", opened=("GOLD",), match_score=0.8),
        )
        self.assertEqual(summary["gold_bought_total"], 2)
        self.assertEqual(summary["gold_gate_failed"], 1)
        self.assertEqual(summary["gold_short_of_gold"], 1)
        self.assertEqual(
            summary["gold_short_types"], {"partial_alternative_purchase": 1}
        )

    def test_bought_gold_with_category_fail_counts_as_gate_failed(self) -> None:
        summary = self.aggregate(
            row(1, "gold_purchase", opened=("GOLD",)),
            row(
                1, "wrong_purchase", opened=("GOLD",),
                category_status="fail",
            ),
        )
        self.assertEqual(summary["gold_gate_failed"], 1)
        self.assertEqual(summary["gold_short_of_gold"], 0)


class BuildReportTest(unittest.TestCase):
    def _write(self, rows) -> Path:
        directory = tempfile.mkdtemp()
        path = Path(directory) / "trajectories.jsonl"
        path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_report_renders_all_sections(self) -> None:
        rows = [
            row(1, "gold_purchase", opened=("GOLD",)),
            row(1, "wrong_purchase", opened=("GOLD", "OTHER"), match_score=0.8),
            row(1, "wrong_purchase", opened=("X",), match_score=0.2),
        ]
        report = module.build_report(self._write(rows))
        text = module.render_report(
            report["summary"],
            report["verdict"],
            report["diagnosed"],
            source=report["source"],
            rows=report["rows"],
            groups=report["groups"],
            skipped_no_gold=report["skipped_no_gold"],
            ambiguous_gold=report["ambiguous_gold"],
        )
        for section in ("## 1. 总览", "## 2. 分桶", "## 3. 按失败类型分层",
                        "## 4.", "## 5.", "## 7. 裁定"):
            self.assertIn(section, text)
        self.assertIn("saw_gold_bought_other", text)
        self.assertIn("never_saw_gold", text)

    def test_empty_input_is_not_measured(self) -> None:
        report = module.build_report(self._write([row(1, "wrong_purchase", opened=("A",))]))
        self.assertEqual(report["verdict"]["status"], "not_measured")

    def test_by_type_seen_includes_bought_gold(self) -> None:
        """§3 的「看到过」必须含 `bought_gold_failed_gate`，否则会漏掉最易修的一类。"""
        rows = [
            row(1, "gold_purchase", opened=("G",)),
            row(1, "wrong_purchase", opened=("G",), target_asin_match=True,
                budget_status="fail"),
            row(1, "wrong_purchase", opened=("X",), budget_status="fail"),
        ]
        report = module.build_report(self._write(rows))
        counter = report["summary"]["by_type"]["wrong_purchase"]
        self.assertEqual(counter["bought_gold_failed_gate"], 1)
        text = module.render_report(
            report["summary"],
            report["verdict"],
            report["diagnosed"],
            source=report["source"],
            rows=report["rows"],
            groups=report["groups"],
            skipped_no_gold=report["skipped_no_gold"],
            ambiguous_gold=report["ambiguous_gold"],
        )
        self.assertIn("| `wrong_purchase` | 2 | 1 | 1 | 50.0% |", text)


class SelectOptionTest(unittest.TestCase):
    """「规格没选全 ⇒ 价格还是区间 ⇒ 猜低价 ⇒ 超支」这条假设的直接检验。"""

    def _text(self, rows) -> str:
        report = module.build_report(self._write(rows))
        return module.render_report(
            report["summary"],
            report["verdict"],
            report["diagnosed"],
            source=report["source"],
            rows=report["rows"],
            groups=report["groups"],
            skipped_no_gold=report["skipped_no_gold"],
            ambiguous_gold=report["ambiguous_gold"],
        )

    def _write(self, rows) -> Path:
        directory = tempfile.mkdtemp()
        path = Path(directory) / "trajectories.jsonl"
        path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_fewer_selects_on_failure_supports_hypothesis(self) -> None:
        rows = [
            row(1, "gold_purchase", opened=("G",), selects=2),
            row(1, "wrong_purchase", opened=("G",), target_asin_match=True,
                selects=0, budget_status="fail"),
        ]
        self.assertIn("支持「规格没选全」", self._text(rows))

    def test_equal_selects_refutes_hypothesis(self) -> None:
        rows = [
            row(1, "gold_purchase", opened=("G",), selects=2),
            row(1, "wrong_purchase", opened=("G",), target_asin_match=True,
                selects=2, budget_status="fail"),
        ]
        self.assertIn("不支持「规格没选全」", self._text(rows))


if __name__ == "__main__":
    unittest.main()
