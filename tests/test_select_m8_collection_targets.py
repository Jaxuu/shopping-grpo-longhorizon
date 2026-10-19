"""`scripts/select_m8_collection_targets.py` 的语义固定。

钉死四件事（改动前先看这里）：

1. **分档只看「成功+失败」混合的题**——全对或全错的题不是「边缘」，不入池。
2. **A 档优先级最高**：出现 `bought_gold_failed_gate`（买对商品却超支）即 A，
   不管组里是否还有别的桶。
3. **held-out 必须剔除**，且 `--limit` 在排序之后生效（否则会先截断再排序）。
4. **输出格式必须是 `{"task_id": N}`**，可直接传给 `collect_sft_data.py --tasks`。
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

import select_m8_collection_targets as module  # noqa: E402


def row(
    task_id: object,
    reward_type: str,
    *,
    opened: tuple[str, ...] = (),
    buy: bool = True,
    match_score: float = 0.4,
    target_asin_match: bool = False,
    budget_status: str = "pass",
) -> dict:
    steps = [{"tool_name": "search_products", "parameters": {"query": "q"}}]
    steps += [{"tool_name": "open_product", "parameters": {"asin": a}} for a in opened]
    if buy:
        steps.append({"tool_name": "buy_now", "parameters": {}})
    return {
        "task_id": task_id,
        "steps": steps,
        "reward_detail": {
            "reward_type": reward_type,
            "weighted_score": match_score,
            "target_asin_match": target_asin_match,
            "hard_gates": {"budget": {"status": budget_status}},
        },
    }


def records(*rows) -> list[dict]:
    return [module.base.extract(item) for item in rows]


class ClassifyTest(unittest.TestCase):
    def test_tier_a_when_bought_gold_but_overspent(self) -> None:
        result = module.classify_task(
            records(
                row(1, "gold_purchase", opened=("G",)),
                row(1, "wrong_purchase", opened=("G",), target_asin_match=True,
                    budget_status="fail"),
            )
        )
        self.assertEqual(result["tier"], "A")
        self.assertEqual(result["bought_gold_failed_gate"], 1)

    def test_tier_b_when_close_substitute(self) -> None:
        result = module.classify_task(
            records(
                row(1, "gold_purchase", opened=("G",)),
                row(1, "wrong_purchase", opened=("G", "X"), match_score=0.85,
                    budget_status="fail"),
            )
        )
        self.assertEqual(result["tier"], "B")
        self.assertEqual(result["close_substitute"], 1)

    def test_far_substitute_falls_to_c(self) -> None:
        result = module.classify_task(
            records(
                row(1, "gold_purchase", opened=("G",)),
                row(1, "wrong_purchase", opened=("G", "X"), match_score=0.2,
                    budget_status="fail"),
            )
        )
        self.assertEqual(result["tier"], "C")

    def test_tier_d_without_budget_failure(self) -> None:
        result = module.classify_task(
            records(
                row(1, "gold_purchase", opened=("G",)),
                row(1, "wrong_purchase", opened=("X",), budget_status="pass"),
            )
        )
        self.assertEqual(result["tier"], "D")

    def test_all_success_is_skipped(self) -> None:
        result = module.classify_task(
            records(
                row(1, "gold_purchase", opened=("G",)),
                row(1, "valid_alternative_purchase", opened=("A",)),
            )
        )
        self.assertIsNone(result)

    def test_all_failure_is_skipped(self) -> None:
        result = module.classify_task(
            records(
                row(1, "wrong_purchase", opened=("X",)),
                row(1, "wrong_purchase", opened=("Y",)),
            )
        )
        self.assertIsNone(result)

    def test_ambiguous_gold_is_skipped(self) -> None:
        result = module.classify_task(
            records(
                row(1, "gold_purchase", opened=("G1",)),
                row(1, "gold_purchase", opened=("G2",)),
                row(1, "wrong_purchase", opened=("X",)),
            )
        )
        self.assertIsNone(result)


class SelectTest(unittest.TestCase):
    def rows(self) -> list[dict]:
        return [
            # A 档
            row(101, "gold_purchase", opened=("G1",)),
            row(101, "wrong_purchase", opened=("G1",), target_asin_match=True,
                budget_status="fail"),
            # C 档
            row(102, "gold_purchase", opened=("G2",)),
            row(102, "wrong_purchase", opened=("X",), budget_status="fail"),
            # 全对，跳过
            row(103, "gold_purchase", opened=("G3",)),
            row(103, "gold_purchase", opened=("G3",)),
        ]

    def test_tier_order_and_counts(self) -> None:
        result = module.select(self.rows(), held_out=set(), limit=None)
        self.assertEqual([item["task_id"] for item in result["selected"]], [101, 102])
        self.assertEqual(result["tiers"], {"A": 1, "C": 1})
        self.assertEqual(result["skipped_no_mix"], 1)

    def test_held_out_is_excluded(self) -> None:
        result = module.select(self.rows(), held_out={101}, limit=None)
        self.assertEqual([item["task_id"] for item in result["selected"]], [102])
        self.assertEqual(result["excluded_held_out"], 1)

    def test_limit_applies_after_sorting(self) -> None:
        result = module.select(self.rows(), held_out=set(), limit=1)
        self.assertEqual([item["task_id"] for item in result["selected"]], [101])

    def test_include_d(self) -> None:
        rows = self.rows() + [
            row(104, "gold_purchase", opened=("G4",)),
            row(104, "wrong_purchase", opened=("Y",), budget_status="pass"),
        ]
        without = module.select(rows, held_out=set(), limit=None)
        with_d = module.select(rows, held_out=set(), include_d=True, limit=None)
        self.assertEqual(len(without["selected"]), 2)
        self.assertEqual(len(with_d["selected"]), 3)


class OutputTest(unittest.TestCase):
    def test_writes_task_id_only_jsonl(self) -> None:
        directory = tempfile.mkdtemp()
        out = Path(directory) / "targets.jsonl"
        report = Path(directory) / "report.md"
        trajectories = Path(directory) / "traj.jsonl"
        rows = [
            row(101, "gold_purchase", opened=("G1",)),
            row(101, "wrong_purchase", opened=("G1",), target_asin_match=True,
                budget_status="fail"),
        ]
        trajectories.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n",
            encoding="utf-8",
        )
        held_out = Path(directory) / "held.jsonl"
        held_out.write_text("", encoding="utf-8")

        code = module.main([
            str(trajectories), "--out", str(out), "--report", str(report),
            "--held-out", str(held_out), "--limit", "10",
        ])
        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(payload["task_id"], 101)
        self.assertEqual(payload["tier"], "A")
        text = report.read_text(encoding="utf-8")
        self.assertIn("A 档", text)
        self.assertIn("collect_sft_data.py", text)


if __name__ == "__main__":
    unittest.main()
