"""Admission-contract regression tests (``shopping-task-admission-v1``).

Contract checks run against the **frozen** pre-rebuild task ids recorded in
``audit/task_audit.jsonl``, so the analysis stays stable after the pools are
rebuilt; a separate class verifies the rebuilt pools themselves.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
if str(SHOP_ENV) not in sys.path:
    sys.path.append(str(SHOP_ENV))

from web_agent_site.engine.admission import (  # noqa: E402
    CONTRACT_VERSION,
    admit_task,
    has_non_ascii_whitespace,
)

_SPEC = importlib.util.spec_from_file_location(
    "audit_solvability", ROOT / "scripts" / "audit_solvability.py"
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)

# M1 reasons that a later, more accurate pass may legitimately clear:
#   * unsolvable_spec  -- the M1 audit used a *raw* exact match, so pure
#     whitespace/case/"/" differences were flagged; the contract's A1 uses the
#     same NFKC-folding resolver as scoring, so those tasks are fine;
#   * gold_over_budget -- a more accurate budget parse can raise the upper bound
#     so the gold fits again.
REPAIRABLE_M1_REASONS = {"unsolvable_spec", "gold_over_budget"}


def _baseline_ids() -> dict[str, set[int]]:
    rows = audit.read_jsonl(ROOT / "audit" / "task_audit.jsonl")
    per_dataset: dict[str, set[int]] = {}
    for row in rows:
        per_dataset.setdefault(row["dataset"], set()).add(int(row["task_id"]))
    return per_dataset


class AdmissionUnitTest(unittest.TestCase):
    def test_non_ascii_whitespace_detection(self):
        self.assertTrue(has_non_ascii_whitespace("全脂纯400克×1袋\u00a0 满2袋"))
        self.assertTrue(has_non_ascii_whitespace(["正常", "全角　空格"]))
        self.assertFalse(has_non_ascii_whitespace("正常 空格"))
        self.assertFalse(has_non_ascii_whitespace(None, [], ()))

    def test_contract_version(self):
        self.assertEqual(CONTRACT_VERSION, "shopping-task-admission-v1")


class AdmissionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.products = audit.load_products(audit.DEFAULT_PRODUCTS)
        cls.goals = audit.build_goals(cls.products)
        cls.product_by_asin = {str(p["asin"]): p for p in cls.products}
        cls.baseline = _baseline_ids()

    def _verdict(self, task_id, held_out=()):
        goal = self.goals[task_id]
        return admit_task(
            task_id,
            goal,
            self.product_by_asin[str(goal["asin"])],
            held_out_task_ids=held_out,
        )

    def _verdicts(self, dataset, held_out=()):
        return [self._verdict(t, held_out) for t in sorted(self.baseline[dataset])]

    def test_evaluation_pool_is_fully_admitted(self):
        """The curated evaluation set is the reference: it must pass 200/200."""
        verdicts = self._verdicts("evaluation")
        rejected = [v["task_id"] for v in verdicts if not v["admitted"]]
        self.assertEqual(rejected, [])
        self.assertEqual(len(verdicts), 200)

    def test_evaluation_nbsp_task_is_a_warning_not_a_rejection(self):
        """3647 carries \\xa0 but scores correctly, so it must not be rejected."""
        verdict = self._verdict(3647)
        self.assertTrue(verdict["admitted"])
        self.assertIn("A5_non_ascii_whitespace", verdict["warnings"])
        self.assertEqual(verdict["reasons"], [])

    def test_zero_price_task_is_rejected(self):
        verdict = self._verdict(1135)
        self.assertFalse(verdict["admitted"])
        self.assertIn("A3_zero_price", verdict["reasons"])

    def test_cross_pool_overlap_is_rejected(self):
        """A6 fires whenever a task id is in the held-out set."""
        grpo_ids = sorted(self.baseline["grpo_train"])[:3]
        for task_id in grpo_ids:
            verdict = self._verdict(task_id, held_out=set(grpo_ids))
            self.assertIn("A6_cross_pool_overlap", verdict["reasons"], task_id)
            self.assertFalse(verdict["admitted"])

    def test_verdicts_are_deterministic(self):
        first = self._verdicts("grpo_validation")
        second = self._verdicts("grpo_validation")
        self.assertEqual(first, second)

    def test_contract_supersedes_the_m1_dead_task_list(self):
        """Directional invariant between the M1 dead list and the contract.

        The contract may legitimately *repair* an M1 dead task, but only for the
        two budget/normalisation-dependent reasons (see
        ``REPAIRABLE_M1_REASONS``). It must never admit a task the M1 audit
        flagged for ``zero_price``, ``reward_unverifiable`` or
        ``gold_partial_match``.
        """
        dead = audit.read_jsonl(ROOT / "audit" / "dead_tasks.jsonl")
        dead_rows: dict[str, list[dict]] = {}
        for row in dead:
            dead_rows.setdefault(row["dataset"], []).append(row)

        for dataset, rows in dead_rows.items():
            if dataset not in self.baseline:
                continue
            held_out = set()
            if dataset.startswith("grpo"):
                held_out = self.baseline.get("sft_pure_v4", set())
            verdicts = {v["task_id"]: v for v in self._verdicts(dataset, held_out)}
            violations = []
            for row in rows:
                task_id = int(row["task_id"])
                verdict = verdicts.get(task_id)
                if verdict is None or not verdict["admitted"]:
                    continue
                reasons = set(row["reasons"])
                if not reasons <= REPAIRABLE_M1_REASONS:
                    violations.append((task_id, sorted(reasons)))
            self.assertEqual(violations, [], f"{dataset}: unexplained admissions")

    def test_repaired_dead_tasks_are_budget_or_normalisation_only(self):
        """Every M1 dead task the contract admits is a legitimate repair."""
        dead = audit.read_jsonl(ROOT / "audit" / "dead_tasks.jsonl")
        sft_ids = self.baseline["sft_pure_v4"]
        repaired = []
        for row in dead:
            if row["dataset"] != "grpo_train":
                continue
            task_id = int(row["task_id"])
            if self._verdict(task_id, held_out=sft_ids)["admitted"]:
                repaired.append((task_id, sorted(row["reasons"])))
        self.assertTrue(repaired, "expected some repairs under v4")
        for task_id, reasons in repaired:
            self.assertTrue(
                set(reasons) <= REPAIRABLE_M1_REASONS, (task_id, reasons)
            )


class CleanedPoolAdmissionTest(unittest.TestCase):
    """The rebuilt pools must contain no rejected task."""

    def test_every_live_task_is_admitted(self):
        verdicts = audit.read_jsonl(ROOT / "audit" / "admission_verdicts.jsonl")
        live = {
            name: set(audit.task_ids(path))
            for name, path in audit.DEFAULT_DATASETS.items()
            if path.exists()
        }
        leaked = [
            (row["dataset"], row["task_id"])
            for row in verdicts
            if not row["admitted"]
            and int(row["task_id"]) in live.get(row["dataset"], set())
        ]
        self.assertEqual(leaked, [])


if __name__ == "__main__":
    unittest.main()
