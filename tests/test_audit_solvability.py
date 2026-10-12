"""M1 solvability-audit regression tests.

Two layers:

* ``AuditUnitTests`` -- pure functions against the frozen product corpus;
* ``AuditBaselineTests`` -- the **frozen** ``audit/`` artifacts. The M1 audit
  documents the pre-fix (reward-v3) state, so once the pools were rebuilt those
  numbers can no longer be recomputed from the live datasets; the frozen
  ``audit/task_audit.jsonl`` / ``difficulty_buckets.json`` / ``dead_tasks.jsonl``
  are the record;
* ``CleanedPoolTests`` -- the post-rebuild state of the live pools.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "audit_solvability", ROOT / "scripts" / "audit_solvability.py"
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)

AUDIT = ROOT / "audit"


class AuditUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.products = audit.load_products(audit.DEFAULT_PRODUCTS)
        cls.goals = audit.build_goals(cls.products)
        cls.product_by_asin = {str(p["asin"]): p for p in cls.products}

    def test_corpus_and_goal_counts(self):
        self.assertEqual(len(self.products), 23421)
        self.assertEqual(len(self.goals), 23421)

    def test_spec_uninterpretable_flags_nbsp_task(self):
        # evaluation task 3647 carries \xa0 in its goal option value.
        goal = self.goals[3647]
        product = self.product_by_asin[str(goal["asin"])]
        self.assertTrue(audit.spec_uninterpretable(goal, product))

    def test_zero_price_flags_known_sft_tasks(self):
        for task_id in (1135, 2304, 3321, 9168, 9670, 15743):
            goal = self.goals[task_id]
            product = self.product_by_asin[str(goal["asin"])]
            self.assertTrue(audit.zero_price(product), task_id)

    def test_zero_price_ignores_healthy_task(self):
        goal = self.goals[3647]
        product = self.product_by_asin[str(goal["asin"])]
        self.assertFalse(audit.zero_price(product))

    def test_bucket_boundaries(self):
        self.assertEqual(audit.bucket_for(1), "1-5")
        self.assertEqual(audit.bucket_for(5), "1-5")
        self.assertEqual(audit.bucket_for(6), "6-10")
        self.assertEqual(audit.bucket_for(10), "6-10")
        self.assertEqual(audit.bucket_for(11), "11-20")
        self.assertEqual(audit.bucket_for(20), "11-20")
        self.assertEqual(audit.bucket_for(21), "21-35")
        self.assertEqual(audit.bucket_for(35), "21-35")


class AuditBaselineTests(unittest.TestCase):
    """The frozen pre-fix baseline recorded under ``audit/``."""

    @classmethod
    def setUpClass(cls):
        cls.task_rows = audit.read_jsonl(AUDIT / "task_audit.jsonl")
        cls.by_dataset: dict[str, list[dict]] = {}
        for row in cls.task_rows:
            cls.by_dataset.setdefault(row["dataset"], []).append(row)

    def test_coverage_completeness(self):
        difficulty = json.loads(
            (AUDIT / "difficulty_buckets.json").read_text(encoding="utf-8")
        )
        recorded = sum(entry["total"] for entry in difficulty["datasets"].values())
        self.assertEqual(len(self.task_rows), recorded)
        self.assertEqual(len(self.task_rows), 200 + 1192 + 1000 + 50)

    def test_alignment_with_optimize_md(self):
        expected = {
            "evaluation": 0.935,
            "sft_pure_v4": 0.627517,
            "grpo_train": 0.661,
            "grpo_validation": 0.48,
        }
        for dataset, rate in expected.items():
            rows = self.by_dataset[dataset]
            parsed = sum(1 for row in rows if row["budget_parsed"])
            self.assertAlmostEqual(parsed / len(rows), rate, places=6, msg=dataset)

    def test_spec_uninterpretable_counts(self):
        expected = {
            "evaluation": 1,
            "sft_pure_v4": 0,
            "grpo_train": 26,
            "grpo_validation": 1,
        }
        for dataset, count in expected.items():
            rows = self.by_dataset[dataset]
            self.assertEqual(
                sum(1 for row in rows if row["spec_uninterpretable"]), count, dataset
            )

    def test_zero_price_counts(self):
        expected = {
            "evaluation": 0,
            "sft_pure_v4": 6,
            "grpo_train": 2,
            "grpo_validation": 0,
        }
        for dataset, count in expected.items():
            rows = self.by_dataset[dataset]
            self.assertEqual(
                sum(1 for row in rows if row["zero_price"]), count, dataset
            )

    def test_dead_task_reasons_present(self):
        reasons = {
            reason
            for row in audit.read_jsonl(AUDIT / "dead_tasks.jsonl")
            for reason in row["reasons"]
        }
        for expected in (
            "unsolvable_spec",
            "zero_price",
            "reward_unverifiable",
            "gold_over_budget",
            "gold_partial_match",
        ):
            self.assertIn(expected, reasons)

    def test_sft_difficulty_buckets_match_optimize_md(self):
        difficulty = json.loads(
            (AUDIT / "difficulty_buckets.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            difficulty["datasets"]["sft_pure_v4"]["counts"],
            {"1-5": 154, "6-10": 927, "11-20": 93, "21-35": 18},
        )

    def test_baseline_overlap_is_six(self):
        sft = {row["task_id"] for row in self.by_dataset["sft_pure_v4"]}
        grpo = {row["task_id"] for row in self.by_dataset["grpo_train"]}
        self.assertEqual(len(sft & grpo), 6)
        for other in ("evaluation", "grpo_validation"):
            ids = {row["task_id"] for row in self.by_dataset[other]}
            self.assertEqual(len(sft & ids), 0, other)

    def test_baseline_used_the_v3_parser(self):
        hashes = (AUDIT / "hashes.txt").read_text(encoding="utf-8")
        self.assertIn("shopping-budget-parser-v3", hashes)

    def test_frozen_evaluation_sha_matches_metadata(self):
        hashes = dict(
            line.split("  ", 1)
            for line in (AUDIT / "hashes.txt").read_text(encoding="utf-8").splitlines()
            if "  " in line
        )
        metadata = json.loads(
            (ROOT / "data" / "evaluation" / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(hashes["evaluation.lf_sha256"], metadata["sha256"])

    def test_outputs_are_byte_reproducible(self):
        """Re-running the audit under the frozen v3 parser is deterministic."""
        with tempfile.TemporaryDirectory() as tmp:
            args = audit.argparse.Namespace(
                products=audit.DEFAULT_PRODUCTS,
                out=Path(tmp),
                index=Path(tmp) / "index.sqlite3",
                skip_retrieval=True,
                budget_parser="v3",
                datasets=audit.OrderedDict(
                    (name, path)
                    for name, path in audit.DEFAULT_DATASETS.items()
                    if path.exists()
                ),
                gold_trajectories=audit.OrderedDict(
                    (name, path)
                    for name, path in audit.DEFAULT_GOLD_TRAJECTORIES.items()
                    if path.exists()
                ),
            )
            first = audit.run_audit(args)
            second = audit.run_audit(
                audit.argparse.Namespace(**{**vars(args), "out": Path(tmp) / "b"})
            )
            self.assertEqual(
                audit.canonical_jsonl(first["dead_tasks"]),
                audit.canonical_jsonl(second["dead_tasks"]),
            )
            self.assertEqual(
                audit.canonical_json(first["difficulty"]),
                audit.canonical_json(second["difficulty"]),
            )


class CleanedPoolTests(unittest.TestCase):
    """The live pools after ``scripts/rebuild_pools.py`` (M1-R5)."""

    @classmethod
    def setUpClass(cls):
        cls.ids = {
            name: set(audit.task_ids(path))
            for name, path in audit.DEFAULT_DATASETS.items()
            if path.exists()
        }

    def test_pools_are_pairwise_disjoint(self):
        import itertools

        for left, right in itertools.combinations(sorted(self.ids), 2):
            self.assertEqual(self.ids[left] & self.ids[right], set(), (left, right))

    def test_evaluation_set_is_untouched(self):
        self.assertEqual(len(self.ids["evaluation"]), 200)

    def test_rejected_tasks_are_gone(self):
        verdicts = audit.read_jsonl(AUDIT / "admission_verdicts.jsonl")
        for row in verdicts:
            if row["admitted"]:
                continue
            self.assertNotIn(
                int(row["task_id"]),
                self.ids[row["dataset"]],
                (row["dataset"], row["task_id"]),
            )

    def test_expected_pool_sizes(self):
        self.assertEqual(len(self.ids["sft_pure_v4"]), 1152)
        self.assertEqual(len(self.ids["grpo_train"]), 906)
        self.assertEqual(len(self.ids["grpo_validation"]), 49)

    def test_grpo_parquet_matches_jsonl(self):
        pq = pytest.importorskip(
            "pyarrow.parquet",
            reason="pyarrow is a transitive dep of `datasets` (see uv.lock)",
        )

        for name in ("train", "validation"):
            table = pq.read_table(ROOT / "data" / "grpo" / f"{name}.parquet")
            self.assertEqual(
                table.column_names,
                ["data_source", "prompt", "ability", "reward_model", "extra_info"],
            )
            parquet_ids = [
                row["task_id"] for row in table.column("extra_info").to_pylist()
            ]
            jsonl_ids = audit.task_ids(ROOT / "data" / "grpo" / f"{name}.jsonl")
            # Same rows, same order: the filter only drops rows.
            self.assertEqual(parquet_ids, jsonl_ids, name)
            self.assertEqual(set(parquet_ids), self.ids[f"grpo_{name}"], name)


if __name__ == "__main__":
    unittest.main()
