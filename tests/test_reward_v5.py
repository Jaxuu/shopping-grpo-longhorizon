"""Reward v5 tests: `wrong_purchase` band + 判定不变式。

Scope（`plans/m7-implementation.md` §3/§6）：

* v5 只改 `wrong_purchase` 的**数值**，不改分类：`reward_type`、hard gate 语义、
  `purchase_success`、`reward_valid` 与 v4 逐字节一致。
* band = `[base, cap] = [-0.85, -0.66]`，`driver = match_score × budget_proximity`。
* `driver = 0` 时与 v4 的常数 -0.85 完全一致（最坏情形不变）。
* `cap < repeat_loop(-0.65) < max_steps(-0.50)`：`wrong_purchase` 仍是最差档，
  不许越过「循环 / 步数耗尽」，否则等于恢复恐慌性购买。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
if str(SHOP_ENV) not in sys.path:
    sys.path.append(str(SHOP_ENV))

from web_agent_site.engine.reward import (  # noqa: E402
    DEFAULT_REWARDS,
    REWARD_VERSION,
    SUPPORTED_REWARD_VERSIONS,
    _budget_proximity,
    evaluate_purchase,
)

# v4 的等价常量组：把 v5 的 band 退化成常数 -0.85。
V4_WRONG_REWARDS = {
    "wrong_purchase_base": -0.85,
    "wrong_purchase_scale": 0.0,
    "wrong_purchase_cap": -0.85,
}

BASE = DEFAULT_REWARDS["wrong_purchase_base"]
SCALE = DEFAULT_REWARDS["wrong_purchase_scale"]
CAP = DEFAULT_REWARDS["wrong_purchase_cap"]


def goal(**overrides):
    value = {
        "asin": "GOLD",
        "category": "Electronics›Cup",
        "expected_brand": [],
        "expected_model": [],
        "expected_core_functions": [],
        "required_options_by_key": {},
        "unresolved_option_requirements": [],
        "price_upper": None,
    }
    value.update(overrides)
    return value


def product(**overrides):
    value = {
        "asin": "GOLD",
        "category": "Electronics›Cup",
        "brand": "",
        "shop_name": "",
        "model": "",
        "title": "Cup",
    }
    value.update(overrides)
    return value


def wrong_purchase(match_goal, match_product, *, price=None, rewards=None):
    """构造一个 hard gate 失败（category fail）的购买，返回 RewardResult。"""
    return evaluate_purchase(
        match_product,
        match_goal,
        selected_options={},
        price=price,
        rewards=rewards,
    )


class RewardV5VersionTest(unittest.TestCase):
    def test_runtime_emits_v5(self):
        self.assertEqual(REWARD_VERSION, "shopsimulator-reward-v5")
        self.assertEqual(SUPPORTED_REWARD_VERSIONS[-1], REWARD_VERSION)

    def test_history_versions_remain_readable(self):
        for version in ("shopsimulator-reward-v3", "shopsimulator-reward-v4"):
            self.assertIn(version, SUPPORTED_REWARD_VERSIONS)

    def test_payload_carries_v5(self):
        result = wrong_purchase(goal(), product())
        self.assertEqual(result.to_dict()["reward_version"], "shopsimulator-reward-v5")


class WrongPurchaseBandTest(unittest.TestCase):
    """`min(cap, base + scale × ms × prox)` 的算术与边界。"""

    def test_constants(self):
        self.assertEqual(BASE, -0.85)
        self.assertEqual(SCALE, 0.20)
        self.assertEqual(CAP, -0.66)
        self.assertEqual(DEFAULT_REWARDS["wrong_purchase"], BASE)

    def test_zero_driver_reproduces_v4_constant(self):
        """brand 全不匹配 ⇒ match_score = 0 ⇒ 与 v4 的 -0.85 逐字节一致。"""
        result = wrong_purchase(
            goal(expected_brand=["Sony"]),
            product(category="Electronics›Mug", brand="Anker"),
        )
        self.assertEqual(result.reward_type, "wrong_purchase")
        self.assertEqual(result.weighted_score, 0.0)
        self.assertEqual(result.reward, -0.85)

    def test_cap_binds_when_every_dimension_matches(self):
        """无任何软要求 ⇒ match_score = 1.0 ⇒ 触顶。"""
        result = wrong_purchase(
            goal(),
            product(category="Electronics›Mug"),
        )
        self.assertEqual(result.reward_type, "wrong_purchase")
        self.assertEqual(result.weighted_score, 1.0)
        self.assertEqual(result.reward, CAP)
        self.assertLess(result.reward, DEFAULT_REWARDS["repeat_loop"])

    def test_band_is_monotone_in_match_score(self):
        """brand 命中、model 不命中 ⇒ ms = 0.35/0.60 = 0.5833。"""
        result = wrong_purchase(
            goal(expected_brand=["Sony"], expected_model=["WH-1000XM5"]),
            product(
                category="Electronics›Mug",
                brand="Sony",
                model="WH-1000XM4",
            ),
        )
        self.assertEqual(result.reward_type, "wrong_purchase")
        self.assertAlmostEqual(result.weighted_score, 0.35 / 0.60, places=9)
        expected = BASE + SCALE * result.weighted_score
        self.assertAlmostEqual(result.reward, expected, places=12)
        self.assertGreater(result.reward, BASE)
        self.assertLess(result.reward, CAP)

    def test_band_never_leaves_its_interval(self):
        for brand in ([], ["Sony"]):
            for model in ([], ["WH-1000XM5"]):
                with self.subTest(brand=brand, model=model):
                    result = wrong_purchase(
                        goal(expected_brand=brand, expected_model=model),
                        product(
                            category="Electronics›Mug",
                            brand="Sony",
                            model="WH-1000XM5",
                        ),
                    )
                    self.assertGreaterEqual(result.reward, BASE)
                    self.assertLessEqual(result.reward, CAP)


class BudgetProximityTest(unittest.TestCase):
    """`prox = min(1, required/actual)`；不可算时必须中性（1.0）。"""

    def test_clips_to_one_when_gate_passes(self):
        self.assertEqual(
            _budget_proximity({"required": 100.0, "actual": 80.0}), 1.0
        )

    def test_discounts_overspend(self):
        self.assertAlmostEqual(
            _budget_proximity({"required": 100.0, "actual": 200.0}), 0.5
        )

    def test_neutral_when_budget_not_declared(self):
        self.assertEqual(_budget_proximity({"required": None, "actual": 80.0}), 1.0)

    def test_neutral_when_actual_is_zero_or_missing(self):
        self.assertEqual(_budget_proximity({"required": 100.0, "actual": 0.0}), 1.0)
        self.assertEqual(_budget_proximity({"required": 100.0}), 1.0)

    def test_neutral_on_non_finite(self):
        self.assertEqual(
            _budget_proximity({"required": float("nan"), "actual": 10.0}), 1.0
        )
        self.assertEqual(
            _budget_proximity({"required": 10.0, "actual": float("inf")}), 1.0
        )

    def test_overspend_lowers_the_reward(self):
        """同一 match_score 下，超支越多分越低（这是选 prox 轴的理由）。"""
        near = wrong_purchase(
            goal(price_upper=100.0),
            product(category="Electronics›Mug"),
            price=102.0,
        )
        far = wrong_purchase(
            goal(price_upper=100.0),
            product(category="Electronics›Mug"),
            price=318.0,
        )
        self.assertEqual(near.reward_type, "wrong_purchase")
        self.assertEqual(far.reward_type, "wrong_purchase")
        self.assertEqual(near.weighted_score, far.weighted_score)
        self.assertGreater(near.reward, far.reward)
        # prox = 100/318 ≈ 0.314 ⇒ 仍在带内（不触底，也不触顶）
        self.assertAlmostEqual(far.reward, BASE + SCALE * (100.0 / 318.0), places=9)
        self.assertGreater(far.reward, BASE)


class LadderInvariantTest(unittest.TestCase):
    """档位序：cap 必须低于 repeat_loop / max_steps / early_abstain / partial 地板。"""

    def test_cap_stays_below_frozen_ladder(self):
        self.assertLess(CAP, DEFAULT_REWARDS["repeat_loop"])
        self.assertLess(CAP, DEFAULT_REWARDS["max_steps"])
        self.assertLess(CAP, DEFAULT_REWARDS["early_abstain"])
        self.assertLess(CAP, DEFAULT_REWARDS["partial_purchase_base"])

    def test_band_floor_is_the_v4_constant(self):
        self.assertEqual(BASE, DEFAULT_REWARDS["wrong_purchase"])


class ClassificationUnchangedTest(unittest.TestCase):
    """v5 与 v4 的**唯一**差异是 wrong_purchase 的数值。"""

    CASES = (
        # (goal overrides, product overrides, price)
        ({}, {}, None),  # gold_purchase
        ({"asin": "GOLD"}, {"asin": "OTHER"}, None),  # valid_alternative
        ({"price_upper": 100.0}, {}, 250.0),  # wrong（超预算）
        ({"expected_brand": ["Sony"]}, {"brand": "Anker"}, None),  # partial
        ({"category": "Electronics›Cup"}, {"category": "Electronics›Mug"}, None),
    )

    def test_only_wrong_purchase_reward_differs(self):
        for goal_overrides, product_overrides, price in self.CASES:
            with self.subTest(goal=goal_overrides, product=product_overrides, price=price):
                v5 = wrong_purchase(
                    goal(**goal_overrides),
                    product(**product_overrides),
                    price=price,
                )
                v4 = wrong_purchase(
                    goal(**goal_overrides),
                    product(**product_overrides),
                    price=price,
                    rewards=V4_WRONG_REWARDS,
                )
                # 判定层逐项相同
                self.assertEqual(v5.reward_type, v4.reward_type)
                self.assertEqual(v5.reward_valid, v4.reward_valid)
                self.assertEqual(v5.target_asin_match, v4.target_asin_match)
                self.assertEqual(
                    v5.to_dict()["purchase_success"],
                    v4.to_dict()["purchase_success"],
                )
                self.assertEqual(
                    {name: gate["status"] for name, gate in v5.hard_gates.items()},
                    {name: gate["status"] for name, gate in v4.hard_gates.items()},
                )
                self.assertEqual(v5.weighted_score, v4.weighted_score)
                # 数值层：只有 wrong_purchase 变
                if v5.reward_type == "wrong_purchase":
                    self.assertEqual(v4.reward, -0.85)
                    self.assertGreaterEqual(v5.reward, -0.85)
                    self.assertLessEqual(v5.reward, CAP)
                else:
                    self.assertEqual(v5.reward, v4.reward)

    def test_purchase_success_ignores_the_band(self):
        """purchase_success 只由 reward_type 决定 ⇒ v5 前后同分。"""
        result = wrong_purchase(goal(), product(category="Electronics›Mug"))
        self.assertEqual(result.reward_type, "wrong_purchase")
        self.assertIs(result.to_dict()["purchase_success"], False)


class ConfigAndManifestTest(unittest.TestCase):
    def test_environment_config_declares_v5(self):
        from web_agent_site.engine.config import load_config

        config = load_config(SHOP_ENV / "configs" / "environment.json")
        self.assertEqual(config["reward"]["version"], "shopsimulator-reward-v5")
        for key in ("wrong_purchase_base", "wrong_purchase_scale", "wrong_purchase_cap"):
            self.assertAlmostEqual(
                float(config["reward"][key]), DEFAULT_REWARDS[key], places=12
            )

    def test_manifest_accepts_v5(self):
        from shopping_grpo.environment.manifest import validate_manifest

        manifest = {
            "manifest_version": "shopping-environment-manifest-v1",
            "environment_version": "shopsimulator-environment-v2.1",
            "shopsimulator_commit": "a" * 40,
            "product_data_sha256": "c" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v5"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v2",
            "max_steps": 35,
            "seed": 20260726,
        }
        self.assertIs(validate_manifest(manifest), manifest)

    def test_adapter_accepts_v5(self):
        from shopping_grpo.training.grpo.adapter.runtime import (
            SUPPORTED_REWARD_VERSIONS as ADAPTER_VERSIONS,
        )

        self.assertIn("shopsimulator-reward-v5", ADAPTER_VERSIONS)


if __name__ == "__main__":
    unittest.main()
