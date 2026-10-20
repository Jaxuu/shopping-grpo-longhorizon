"""`scripts/diagnose_m10_d_bucket.py` 的口径固定。

钉死四件事：

1. **选项字符串必须归一**：投影会压平大小写、把 `/` 写成 ` | `、把不间断空格写成两个空格。
   不归一就会把「选了别的」误报成「要求的规格值不在选项里」（实测误报 2/18）。
2. **同一轴多次选择，最后一次生效**——环境是覆盖语义，先选对再选错等于没选对。
3. **分类优先级**：没选 > 漏轴 > 覆盖 > 选第 1 个 > 相邻 > 其它。
4. **判定看训练池的两条证据**：硬例条数（≥4 选项）和硬例里第 0 位的占比。
   硬例够多且位置接近均匀 ⇒ 补数据无效。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import diagnose_m10_d_bucket as module  # noqa: E402


def detail_page(options: dict[str, list[str]]) -> str:
    return (
        "[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\nasin: X\n"
        f"available_options: {module.json.dumps(options, ensure_ascii=False)}"
    )


def trajectory(task_id: int, required: list[str], selected: list[str], options: dict) -> dict:
    steps: list[dict] = [{"tool_name": "open_product", "parameters": {"asin": "X"},
                          "raw_observation": detail_page(options)}]
    for value in selected:
        steps.append({"tool_name": "select_option", "parameters": {"value": value}})
    steps.append({"tool_name": "buy_now", "parameters": {}})
    return {
        "task_id": task_id,
        "steps": steps,
        "terminal_result": {
            "goal": {"goal_options": required},
            "reward_detail": {"target_asin_match": True, "reward_type": "wrong_purchase"},
        },
    }


class NormTest(unittest.TestCase):
    def test_case_folded(self) -> None:
        self.assertEqual(module.norm("A559"), module.norm("a559"))

    def test_pipe_and_slash_are_equal(self) -> None:
        """投影把 `1/2` 写成 `1 | 2`；归一后必须相等。"""
        self.assertEqual(module.norm("半度儿童中阮1/2中阮"), module.norm("半度儿童中阮1 | 2中阮"))

    def test_nbsp_and_double_space_removed(self) -> None:
        self.assertEqual(
            module.norm("全脂纯400克×1袋\xa0 满2袋送羊奶皂1"),
            module.norm("全脂纯400克×1袋  满2袋送羊奶皂1"),
        )

    def test_different_values_stay_different(self) -> None:
        self.assertNotEqual(module.norm("【粉剂】10斤"), module.norm("【粉剂】5斤"))


class AvailableOptionsTest(unittest.TestCase):
    def test_parses_single_axis(self) -> None:
        parsed = module.available_options(detail_page({"颜色分类": ["a", "b"]}))
        self.assertEqual(parsed, {"颜色分类": ["a", "b"]})

    def test_missing_returns_empty(self) -> None:
        self.assertEqual(module.available_options("page_type: search_results"), {})

    def test_bad_json_returns_empty(self) -> None:
        self.assertEqual(module.available_options("available_options: {not json"), {})


class ClassifyTest(unittest.TestCase):
    options = {"颜色分类": ["甲", "乙", "丙", "丁"]}

    def test_no_selection(self) -> None:
        self.assertEqual(module.classify(["乙"], [], self.options)["category"], "no_selection")

    def test_selected_correct(self) -> None:
        self.assertEqual(module.classify(["乙"], ["乙"], self.options)["category"], "selected_correct")

    def test_picked_first(self) -> None:
        self.assertEqual(module.classify(["丙"], ["甲"], self.options)["category"], "picked_first")

    def test_adjacent(self) -> None:
        self.assertEqual(module.classify(["丙"], ["乙"], self.options)["category"], "adjacent")

    def test_other(self) -> None:
        self.assertEqual(module.classify(["丁"], ["乙"], self.options)["category"], "other")

    def test_missing_axis(self) -> None:
        options = {"尺寸": ["1", "2"], "颜色": ["红"]}
        self.assertEqual(module.classify(["红", "2"], ["2"], options)["category"], "missing_axis")

    def test_overwrote(self) -> None:
        """先选对再选错 = 覆盖，不能算「选对了」。"""
        self.assertEqual(module.classify(["乙"], ["乙", "甲"], self.options)["category"], "overwrote")

    def test_required_not_offered(self) -> None:
        self.assertEqual(
            module.classify(["戊"], ["甲"], self.options)["category"], "required_not_offered"
        )


class DiagnoseTrajectoryTest(unittest.TestCase):
    def test_grade_failure_requires_target_match(self) -> None:
        ok = trajectory(1, ["乙"], ["甲"], {"颜色分类": ["甲", "乙"]})
        self.assertTrue(module.is_grade_failure(ok))
        ok["terminal_result"]["reward_detail"]["target_asin_match"] = False
        self.assertFalse(module.is_grade_failure(ok))

    def test_gold_purchase_is_not_grade_failure(self) -> None:
        row = trajectory(1, ["乙"], ["乙"], {"颜色分类": ["甲", "乙"]})
        row["terminal_result"]["reward_detail"]["reward_type"] = "gold_purchase"
        self.assertFalse(module.is_grade_failure(row))

    def test_end_to_end_category(self) -> None:
        row = trajectory(1, ["乙"], ["甲"], {"颜色分类": ["甲", "乙"]})
        self.assertEqual(module.diagnose_trajectory(row)["category"], "picked_first")


class PoolStatsTest(unittest.TestCase):
    def row(self, selections: list[str], options: dict) -> dict:
        messages: list[dict] = [{"role": "tool", "content": detail_page(options)}]
        for value in selections:
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {
                                "name": "select_option",
                                "arguments": module.json.dumps({"value": value}, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
        return {"messages": messages}

    def test_counts_hard_and_position(self) -> None:
        rows = [
            self.row(["丁"], {"颜色分类": ["甲", "乙", "丙", "丁"]}),
            self.row(["甲"], {"颜色分类": ["甲", "乙", "丙", "丁"]}),
        ]
        stats = module.pool_selection_stats(rows)
        self.assertEqual(stats["matched"], 2)
        self.assertEqual(stats["hard_ge4"], 2)
        self.assertEqual(stats["hard_share_at_index0"], 0.5)

    def test_single_option_is_trivial(self) -> None:
        stats = module.pool_selection_stats([self.row(["甲"], {"颜色分类": ["甲"]})])
        self.assertEqual(stats["trivial_single_option"], 1)
        self.assertEqual(stats["hard_ge4"], 0)


class VerdictTest(unittest.TestCase):
    def test_not_enough_hard_examples(self) -> None:
        pool = {"hard_ge4": 10, "hard_share_at_index0": 0.1}
        self.assertIn("数据不够", module.verdict(None, pool))

    def test_positional_bias(self) -> None:
        pool = {"hard_ge4": 1000, "hard_share_at_index0": 0.7}
        self.assertIn("位置偏置", module.verdict(None, pool))

    def test_capacity_limit(self) -> None:
        pool = {"hard_ge4": 1000, "hard_share_at_index0": 0.13}
        self.assertIn("分辨力问题", module.verdict(None, pool))


if __name__ == "__main__":
    unittest.main()
