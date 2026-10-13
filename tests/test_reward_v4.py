"""Reward v4 tests (M3: versioning, soft price layer, process evidence).

Scope (plans/task.md §M3):
* M3-8 / R11: ``REWARD_VERSION`` is ``shopsimulator-reward-v4`` and every
  consumer accepts both v3 (historical) and v4 (current) — 双层口径.
* M3-5 / T5: whitespace-tainted ``goal_options`` (evaluation ``task_id=3647``)
  still match on the reward path.
* M3-6: soft "左右" price preference nominal value + proximity score
  (informational, marked 改标准).
* M3-7: process evidence block for the training side (M6 consumer).
* P1 / T7: hard-gate semantics are byte-for-byte v3; budgets parsed by the
  frozen v3 parser are value-identical under v4 (zero drift where v3 applied).
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

from web_agent_site.engine import reward as reward_module  # noqa: E402
from web_agent_site.engine.constraints import (  # noqa: E402
    explicit_budget_from_instruction,
    explicit_price_preference_from_instruction,
)
from web_agent_site.engine.reward import (  # noqa: E402
    PROCESS_EVIDENCE_VERSION,
    REWARD_VERSION,
    SUPPORTED_REWARD_VERSIONS,
    evaluate_purchase,
)
from web_agent_site.engine.reward_features import (  # noqa: E402
    normalize_option_text,
)

_SPEC = importlib.util.spec_from_file_location(
    "audit_solvability", ROOT / "scripts" / "audit_solvability.py"
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def _minimal_goal(**overrides):
    goal = {
        "asin": "GOLD",
        "category": "Electronics›Cup",
        "expected_brand": [],
        "expected_model": [],
        "expected_core_functions": [],
        "required_options_by_key": {},
        "unresolved_option_requirements": [],
        "price_upper": None,
    }
    goal.update(overrides)
    return goal


def _minimal_product(**overrides):
    product = {
        "asin": "GOLD",
        "category": "Electronics›Cup",
        "brand": "",
        "shop_name": "",
        "title": "Cup",
    }
    product.update(overrides)
    return product


class RewardVersionTest(unittest.TestCase):
    """M3-8 / R11: the runtime emits v4 while still accepting v3."""

    def test_reward_version_is_v4(self):
        self.assertEqual(REWARD_VERSION, "shopsimulator-reward-v4")
        self.assertIn(
            "shopsimulator-reward-v3",
            SUPPORTED_REWARD_VERSIONS,
            "双层口径：旧 v3 轨迹必须仍可读",
        )

    def test_reward_payload_carries_v4_version(self):
        result = evaluate_purchase(
            _minimal_product(),
            _minimal_goal(price_upper=None),
            selected_options={},
        )
        self.assertEqual(result.to_dict()["reward_version"], REWARD_VERSION)


class HardGateSemanticsTest(unittest.TestCase):
    """P1 / M3-T7: ``category`` + ``budget`` gates behave exactly as v3."""

    def test_no_declared_budget_passes_through(self):
        result = evaluate_purchase(
            _minimal_product(),
            _minimal_goal(price_upper=None),
            selected_options={},
        )
        gate = result.hard_gates["budget"]
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["comparator"], "budget_not_declared_v1")
        self.assertEqual(result.reward_type, "gold_purchase")

    def test_actual_le_upper_passes(self):
        result = evaluate_purchase(
            _minimal_product(),
            _minimal_goal(price_upper=100.0),
            selected_options={},
            price=100.0,
        )
        self.assertEqual(result.hard_gates["budget"]["status"], "pass")

    def test_actual_above_upper_fails_strictly(self):
        result = evaluate_purchase(
            _minimal_product(),
            _minimal_goal(price_upper=100.0),
            selected_options={},
            price=100.01,
        )
        self.assertEqual(result.hard_gates["budget"]["status"], "fail")
        self.assertEqual(result.reward_type, "wrong_purchase")

    def test_unverifiable_price_stays_unverifiable(self):
        result = evaluate_purchase(
            _minimal_product(),
            _minimal_goal(price_upper=100.0),
            selected_options={},
            price_resolution={
                "status": "unverifiable",
                "price": None,
                "version": "variant-price-v1",
                "method": "test",
                "evidence": {},
            },
        )
        self.assertEqual(result.reward_type, "reward_unverifiable")
        self.assertFalse(result.reward_valid)

    def test_soft_preference_never_changes_terminal_reward(self):
        """M3-6 red line: the informational layer cannot move the reward."""
        base = evaluate_purchase(
            _minimal_product(),
            _minimal_goal(price_upper=None),
            selected_options={},
            price=450.0,
        )
        soft_goal = _minimal_goal(
            price_upper=None,
            instruction_text="价格控制在450元左右就好",
        )
        soft = evaluate_purchase(
            _minimal_product(),
            soft_goal,
            selected_options={},
            price=450.0,
        )
        self.assertEqual(base.reward, soft.reward)
        self.assertEqual(base.reward_type, soft.reward_type)
        self.assertEqual(
            {k: v["status"] for k, v in base.hard_gates.items()},
            {k: v["status"] for k, v in soft.hard_gates.items()},
        )


class SoftPricePreferenceTest(unittest.TestCase):
    """M3-6 (改标准, informational): nominal value + proximity score."""

    def test_nominal_extraction(self):
        self.assertEqual(
            explicit_price_preference_from_instruction("价格控制在450元左右"),
            450.0,
        )
        self.assertEqual(
            explicit_price_preference_from_instruction("预算两百上下"),
            200.0,
        )
        self.assertIsNone(
            explicit_price_preference_from_instruction("预算不超过300元")
        )
        self.assertIsNone(explicit_price_preference_from_instruction("随便看看"))

    def test_hard_budget_keeps_frozen_v3_tolerance(self):
        # The budget gate still consumes the +10% v3 tolerance, not the nominal.
        self.assertAlmostEqual(
            explicit_budget_from_instruction("价格控制在450元左右"), 495.0
        )

    def test_proximity_reported_in_evidence(self):
        goal = _minimal_goal(
            price_upper=None,
            instruction_text="价格控制在450元左右就好",
        )
        result = evaluate_purchase(
            _minimal_product(),
            goal,
            selected_options={},
            price=495.0,
        )
        soft = result.evidence["soft_price_preference"]
        self.assertTrue(soft["active"])
        self.assertEqual(soft["nominal"], 450.0)
        self.assertEqual(soft["actual"], 495.0)
        self.assertAlmostEqual(soft["proximity"], 1.0 - 45.0 / 450.0)

    def test_proximity_inactive_without_preference(self):
        result = evaluate_purchase(
            _minimal_product(),
            _minimal_goal(price_upper=None),
            selected_options={},
            price=450.0,
        )
        soft = result.evidence["soft_price_preference"]
        self.assertFalse(soft["active"])
        self.assertIsNone(soft["proximity"])


class ProcessEvidenceTest(unittest.TestCase):
    """M3-7: additive terminal diagnostics for the training side."""

    def test_process_evidence_block(self):
        result = evaluate_purchase(
            _minimal_product(),
            _minimal_goal(price_upper=100.0),
            selected_options={},
            price=90.0,
        )
        block = result.evidence["process_evidence"]
        self.assertEqual(block["version"], PROCESS_EVIDENCE_VERSION)
        self.assertEqual(
            block["hard_gates"], {"category": "pass", "budget": "pass"}
        )
        self.assertIn("match_score", block)
        self.assertIn("dimension_scores", block)
        self.assertIn("soft_price_preference", block)
        self.assertTrue(block["target_asin_match"])

    def test_task_facts_carry_process_evidence(self):
        from shopping_grpo.evaluation.rubric import (
            PROCESS_EVIDENCE_VERSION as FACTS_VERSION,
            build_task_facts,
        )

        facts = build_task_facts(
            task_id=1,
            query="预算350元左右买个杯子",
            target_product=_minimal_product(),
            instruction_record={},
        )
        block = facts["process_evidence"]
        self.assertEqual(block["version"], FACTS_VERSION)
        self.assertTrue(block["soft_price_preference"]["active"])
        self.assertEqual(block["soft_price_preference"]["nominal"], 350.0)


class ConsumerDualAcceptanceTest(unittest.TestCase):
    """M3-T10 / R11: consumers accept v3 and v4, reject everything else."""

    def test_adapter_validate_reward_accepts_both(self):
        from shopping_grpo.training.grpo.adapter.runtime import validate_reward

        base = {
            "reward_type": "gold_purchase",
            "reward_valid": True,
            "termination_reason": "gold_purchase",
            "terminal_utility": 1.0,
            "purchase_success": True,
            "sampling_invalid": False,
            "hard_gates": {
                "category": {
                    "status": "pass",
                    "passed": True,
                    "verifiable": True,
                }
            },
            "weighted_score": 1.0,
        }
        for version in ("shopsimulator-reward-v3", "shopsimulator-reward-v4"):
            detail = validate_reward(dict(base, reward_version=version))
            self.assertEqual(detail["reward_version"], version)
        with self.assertRaises(ValueError):
            validate_reward(dict(base, reward_version="unsupported-reward"))

    def test_metrics_strict_success_accepts_v4(self):
        from shopping_grpo.evaluation.summary import summarize_trajectories

        def trajectory(version):
            return {
                "task_id": 1,
                "status": "done",
                "done": True,
                "final_reward": 1.0,
                "steps": [],
                "terminal_result": {
                    "done": True,
                    "over": True,
                    "reward_detail": {
                        "reward_version": version,
                        "reward_type": "gold_purchase",
                        "reward_valid": True,
                        "purchase_success": True,
                        "termination_reason": "gold_purchase",
                        "terminal_utility": 1.0,
                        "weighted_score": 1.0,
                    },
                },
            }

        for version in ("shopsimulator-reward-v3", "shopsimulator-reward-v4"):
            summary = summarize_trajectories([1], [trajectory(version)])
            self.assertEqual(summary["strict_successes"], 1, version)
            self.assertEqual(summary["gold_purchases"], 1, version)

    def test_metrics_reject_unknown_version(self):
        from shopping_grpo.evaluation.metrics import (
            compute_deterministic_metrics,
        )

        normalized = {
            "schema_version": "shopping-normalized-trajectory-v1",
            "trajectory_id": "t",
            "task_id": 1,
            "status": "done",
            "done": True,
            "events": [],
            "terminal": {
                "done": True,
                "over": True,
                "reward_detail": {
                    "reward_version": "unsupported-reward",
                    "reward_type": "gold_purchase",
                    "reward_valid": True,
                },
            },
        }
        metrics = compute_deterministic_metrics(normalized)
        self.assertIn(
            "unexpected_reward_version",
            metrics["validity"]["trajectory_contract_issues"],
        )

    def test_environment_config_accepts_v3_and_v4(self):
        from web_agent_site.engine.config import load_config

        config_path = SHOP_ENV / "configs" / "environment.json"
        config = load_config(config_path)
        self.assertEqual(config["reward"]["version"], REWARD_VERSION)
        for version in ("shopsimulator-reward-v3", "shopsimulator-reward-v4"):
            mutated = dict(config)
            mutated["reward"] = dict(config["reward"], version=version)
            import json
            import tempfile

            with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, encoding="utf-8"
            ) as handle:
                json.dump(mutated, handle)
                path = handle.name
            try:
                self.assertEqual(load_config(path)["reward"]["version"], version)
            finally:
                Path(path).unlink(missing_ok=True)

    def test_frozen_environment_json_is_v4(self):
        import json

        for relative in (
            "data/environment.json",
            "environments/ShopSimulator/shop_env/configs/environment.json",
        ):
            payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
            self.assertEqual(
                payload["reward"]["version"],
                "shopsimulator-reward-v4",
                relative,
            )


class EvaluationTask3647WhitespaceTest(unittest.TestCase):
    """M3-T5: \\xa0-tainted goal_options still match on the reward path."""

    @classmethod
    def setUpClass(cls):
        cls.products = audit.load_products(audit.DEFAULT_PRODUCTS)
        cls.goals = audit.build_goals(cls.products)
        cls.product_by_asin = {
            str(product["asin"]): product for product in cls.products
        }

    def test_task_3647_options_match_after_normalization(self):
        goal = self.goals[3647]
        product = self.product_by_asin[str(goal["asin"])]
        tainted = [
            value
            for value in goal["goal_options"]
            if "\xa0" in str(value) or "\u3000" in str(value)
        ]
        self.assertTrue(tainted, "fixture drifted: task 3647 lost its \\xa0 value")
        for value in tainted:
            normalized = normalize_option_text(value)
            available = {
                normalize_option_text(entry.get("value"))
                for values in (product.get("customization_options") or {}).values()
                for entry in (values or [])
                if isinstance(entry, dict)
            }
            self.assertIn(normalized, available)
        # And the full gold purchase remains resolvable on the reward path.
        selfcheck = audit.gold_selfcheck(goal, product)
        self.assertIn(
            selfcheck["reward_type"],
            {"gold_purchase", "valid_alternative_purchase"},
        )
        self.assertEqual(
            set(selfcheck["hard_gates"].values()), {"pass"}
        )


class BudgetValueIdentityV3V4Test(unittest.TestCase):
    """M3-T7 (offline replay): v3/v4 value drift is fully attributable.

    Classification over the 23,421-goal corpus (2026-10 audit):
    * 1,381 softener extensions (v4 = v3 * 1.1): v3 silently dropped the
      左右/上下 suffix when a colloquial unit (块/块钱) preceded it; v4
      applies the frozen v3 +10% tolerance.
    * 33 upward re-parses: v4 covers phrasings v3 truncated (Chinese
      numerals such as 四千). 4 fall inside the training pools.
    * 18 downward re-parses (v4 parser defects: 万-shorthand + 左右 misses
      the tolerance; soft 左右 matching outranks explicit hard caps; mixed
      numerals such as 3百). **None of them falls inside the four
      datasets** — recorded as an M4 input (plans/task.md §M3), not fixed
      mid-flight per P8 (would invalidate the frozen M1 admission verdicts).
    """

    @classmethod
    def setUpClass(cls):
        cls.products = audit.load_products(audit.DEFAULT_PRODUCTS)
        cls.instructions = []
        for item in cls.products:
            for instruction in item.get("instructions") or []:
                if instruction.get("attributes"):
                    cls.instructions.append(instruction["instruction"])
        cls.dataset_ids = {
            name: set(audit.task_ids(path))
            for name, path in audit.DEFAULT_DATASETS.items()
            if path.exists()
        }

    def _drift_by_dataset(self):
        drift = {name: [] for name in self.dataset_ids}
        for task_id, text in enumerate(self.instructions):
            v3_value = audit.budget_parser_v3(text)
            if v3_value is None:
                continue
            v4_value = explicit_budget_from_instruction(text)
            if v4_value == v3_value:
                continue
            for name, ids in self.dataset_ids.items():
                if task_id in ids:
                    drift[name].append((task_id, v3_value, v4_value))
        return drift

    def test_evaluation_drift_is_softener_extension_only(self):
        """Frozen-eval comparability: every eval drift must be v4 = v3 * 1.1."""
        offenders = [
            (task_id, v3, v4)
            for task_id, v3, v4 in self._drift_by_dataset()["evaluation"]
            if not (v4 == v3 or abs(v4 - v3 * 1.1) < 1e-6)
        ]
        self.assertEqual(offenders, [])

    def test_no_in_dataset_drift_lowers_the_ceiling(self):
        """A lower ceiling would silently tighten gold gates; never allowed."""
        lowered = []
        for name, rows in self._drift_by_dataset().items():
            lowered.extend((name, *row) for row in rows if row[2] < row[1])
        self.assertEqual(lowered, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
