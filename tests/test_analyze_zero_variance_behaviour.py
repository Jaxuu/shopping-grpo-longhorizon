"""`scripts/analyze_zero_variance_behaviour.py` 的语义固定。

钉死四件事（改动前先看这里）：

1. **`think` 必须从 `act` 签名里排除**——否则自由文本推理会让每组都显得「不一样」，
   结论被彻底带偏（这是本脚本存在的理由）。
2. **分桶必须三分类**，且 `same_destination` 的前提是「买的 ASIN 相同」，
   不能只凭「动作不同」就判成殊途同归。
3. **零方差组的定义是「全组都是 `wrong_purchase` 且 `ms×prox` 方差为 0」**，
   混档组不得混进来（否则统计量无意义）。
4. **判据是复制品占比**，三档阈值（≥0.70 / ≤0.30 / 中间）必须与报告文本一致。
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

import analyze_zero_variance_behaviour as module  # noqa: E402


def step(tool: str, **parameters) -> dict:
    return {"tool_name": tool, "parameters": parameters}


def row(
    task_id: object,
    *,
    match_score: float = 0.4,
    required: float = 100.0,
    actual: float = 100.0,
    steps: list[dict] | None = None,
    reward_type: str = "wrong_purchase",
    nested: bool = False,
) -> dict:
    detail = {
        "reward_type": reward_type,
        "weighted_score": match_score,
        "hard_gates": {
            "budget": {"required": required, "actual": actual, "status": "fail"},
            "category": {"status": "pass"},
        },
    }
    payload = {"task_id": task_id, "steps": steps or []}
    if nested:
        payload["terminal_result"] = {"reward_detail": detail}
    else:
        payload["reward_detail"] = detail
    return payload


def purchase_flow(asin: str, *, query: str = "usb hub", note: str = "") -> list[dict]:
    steps = [step("search_products", query=query), step("open_product", asin=asin)]
    if note:
        steps.append(step("think", note=note))
    steps.append(step("buy_now"))
    return steps


class BehaviourTest(unittest.TestCase):
    def test_think_is_excluded_from_act_signature(self) -> None:
        left = row(1, steps=purchase_flow("A1", note="第一种想法"))
        right = row(1, steps=purchase_flow("A1", note="完全不同的想法"))
        left_print = module.behaviour(left)
        right_print = module.behaviour(right)
        self.assertNotEqual(left_print["full"], right_print["full"])
        self.assertEqual(left_print["act"], right_print["act"])

    def test_purchased_asin_is_last_open_before_buy(self) -> None:
        steps = [
            step("search_products", query="q"),
            step("open_product", asin="A1"),
            step("back_to_search"),
            step("open_product", asin="A2"),
            step("buy_now"),
        ]
        self.assertEqual(module.behaviour(row(1, steps=steps))["purchased_asin"], "A2")

    def test_no_purchase_yields_none(self) -> None:
        steps = [step("search_products", query="q"), step("open_product", asin="A1")]
        self.assertIsNone(module.behaviour(row(1, steps=steps))["purchased_asin"])

    def test_axis_value_matches_gate2_product_axis(self) -> None:
        record = row(1, match_score=0.5, required=100.0, actual=200.0)
        # prox = min(1, 100/200) = 0.5 ⇒ 0.5 * 0.5 = 0.25
        self.assertAlmostEqual(module.axis_value(record), 0.25, places=9)


class BucketTest(unittest.TestCase):
    def test_identical_trajectories_are_replica(self) -> None:
        rows = [row(7, steps=purchase_flow("A1", note=f"想法{i}")) for i in range(4)]
        stats = module.classify_group(rows)
        self.assertEqual(stats["bucket"], "replica")
        self.assertTrue(stats["replica_act"])
        self.assertFalse(stats["replica_full"])  # think 文本不同
        self.assertIsNone(stats["first_divergence_act"])

    def test_different_paths_same_asin_is_same_destination(self) -> None:
        rows = [
            row(7, steps=purchase_flow("A1", query="usb hub")),
            row(7, steps=purchase_flow("A1", query="usb c hub")),
            row(7, steps=purchase_flow("A1", query="hub 4 port")),
            row(7, steps=purchase_flow("A1", query="usb hub")),
        ]
        stats = module.classify_group(rows)
        self.assertEqual(stats["bucket"], "same_destination")
        self.assertFalse(stats["replica_act"])
        self.assertEqual(stats["purchased_unique"], 1)
        self.assertEqual(stats["first_divergence_act"], 0)  # 第 0 步搜索词就不同

    def test_different_asin_is_coincidence(self) -> None:
        rows = [
            row(7, steps=purchase_flow("A1")),
            row(7, steps=purchase_flow("A2")),
        ]
        stats = module.classify_group(rows)
        self.assertEqual(stats["bucket"], "coincidence")
        self.assertEqual(stats["purchased_unique"], 2)

    def test_first_divergence_locates_the_step(self) -> None:
        base = [step("search_products", query="q"), step("open_product", asin="A1")]
        other = [step("search_products", query="q"), step("open_product", asin="A2")]
        stats = module.classify_group([row(7, steps=base), row(7, steps=other)])
        self.assertEqual(stats["first_divergence_act"], 1)


class SelectionTest(unittest.TestCase):
    def test_only_all_wrong_zero_variance_groups_enter(self) -> None:
        groups = {
            "1": [row(1, steps=purchase_flow("A1")) for _ in range(3)],
            "2": [
                row(2, match_score=0.3, steps=purchase_flow("A1")),
                row(2, match_score=0.6, steps=purchase_flow("A1")),
            ],
            "3": [
                row(3, steps=purchase_flow("A1")),
                row(3, steps=purchase_flow("A1"), reward_type="gold_purchase"),
            ],
            "4": [row(4, steps=purchase_flow("A1"))],  # 单条，不够一组
        }
        zero, nonzero, mixed = module.select_zero_variance_groups(groups, min_group_size=2)
        self.assertEqual([item["task_id"] for item in zero], ["1"])
        self.assertEqual([item["task_id"] for item in nonzero], ["2"])
        self.assertEqual([item["task_id"] for item in mixed], ["3"])

    def test_nested_reward_detail_is_read(self) -> None:
        rows = [row(9, steps=purchase_flow("A1"), nested=True) for _ in range(2)]
        self.assertEqual(module.behaviour(rows[0])["reward_type"], "wrong_purchase")


class VerdictTest(unittest.TestCase):
    def summary(self, replica: int, total: int) -> dict:
        return {
            "zero_groups": total,
            "replica_share": replica / total if total else None,
            "buckets": {"replica": replica},
        }

    def test_high_replica_share_is_deterministic(self) -> None:
        result = module.verdict(self.summary(8, 10), high=0.70, low=0.30)
        self.assertEqual(result["status"], "deterministic")

    def test_low_replica_share_has_headroom(self) -> None:
        result = module.verdict(self.summary(2, 10), high=0.70, low=0.30)
        self.assertEqual(result["status"], "behavioural_headroom")

    def test_middle_is_mixed(self) -> None:
        result = module.verdict(self.summary(5, 10), high=0.70, low=0.30)
        self.assertEqual(result["status"], "mixed")

    def test_empty_is_not_measured(self) -> None:
        result = module.verdict(self.summary(0, 0), high=0.70, low=0.30)
        self.assertEqual(result["status"], "not_measured")


class RenderTest(unittest.TestCase):
    def test_report_mentions_bucket_and_verdict(self) -> None:
        zero = [
            module.classify_group(
                [row(7, steps=purchase_flow("A1", note=f"n{i}")) for i in range(3)]
            )
        ]
        summary = module.aggregate(zero, [])
        judgement = module.verdict(summary, high=0.70, low=0.30)
        text = module.render_report(
            summary, judgement, zero, [], source="toy.jsonl", rows=3
        )
        self.assertIn("`replica`", text)
        self.assertIn("deterministic", text)
        self.assertIn("think", text)


class BuildReportTest(unittest.TestCase):
    def _write(self, rows) -> Path:
        directory = tempfile.mkdtemp()
        path = Path(directory) / "trajectories.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_empty_step_rows_do_not_become_replica(self) -> None:
        """无工具步的轨迹必须被剔除，否则「空动作序列都相同」会伪装成 replica。"""
        rows = [row(1, steps=purchase_flow("A1")) for _ in range(2)]
        rows += [{"task_id": 2, "steps": [], "reward_detail": {
            "reward_type": "wrong_purchase", "weighted_score": 0.4,
            "hard_gates": {"budget": {"required": 1.0, "actual": 1.0}},
        }} for _ in range(2)]
        report = module.build_report(self._write(rows))
        self.assertEqual(report["empty_step_rows"], 2)
        self.assertEqual(report["summary"]["zero_groups"], 1)  # 只剩 task 1
        self.assertEqual(report["zero_groups"][0]["task_id"], "1")

    def test_verdict_not_measured_on_empty_input(self) -> None:
        report = module.build_report(self._write([row(1, steps=purchase_flow("A1"))]))
        self.assertEqual(report["verdict"]["status"], "not_measured")


if __name__ == "__main__":
    unittest.main()
