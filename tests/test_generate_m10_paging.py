"""`scripts/generate_m10_paging.py` 的口径固定。

钉死五件事：

1. **选题带宽 21–60**：第一页渲染前 20 名，所以 21 名以后才需要翻页；60 名是翻 2 页的上限。
2. **查询词取「把 gold 排得最靠前」的那个**，名次相同取先出现的（可复现）。
3. **翻页后必须在渲染出来的页里真的看到 gold**，不能只信离线名次。
4. **规格值要对到页面上真实存在的字符串**（大小写/斜杠/空格归一后比），不能直接抄要求值。
5. **生成的轨迹必须能过采集链路同一套契约**（`acceptance_reasons` + `build_sft_row` +
   `prepare_sft_curriculum._validate_row`），否则就是造了一批进不了训练集的数据。

注意：契约里的动作守卫要求「点击目标必须出现在上一页的按钮列表里」，
且商品 ID 必须是 8–12 位数字（`PRODUCT_ID_CAPTURE`），所以夹具不能用 `A1`/`G1`。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT / "scripts"), str(ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import generate_m10_paging as module  # noqa: E402

GOLD = "620296036484"
OTHER = "11111111"
THIRD = "22222222"
OPTION_PLAIN = "三色聚光灯（送星光镜带夹子）"
OPTION_WITH_MACRO = "三色聚光灯+微距（送星光镜带夹子）37m"


def search_page(rows: list[tuple[int, str]], page: int = 1) -> str:
    lines = [
        "[SHOPPING_OBSERVATION_V2]",
        "page_type: search_results",
        "query: x",
        f"Page {page} of 8 (Total results: 150; ranks {20 * (page - 1) + 1}-{20 * page})",
    ]
    lines += [f"{rank}|{asin}|9.0|品牌|类目|属性|标题" for rank, asin in rows]
    lines.append('可点击的按钮: ["Next >", "< Prev", "Back to Search"]')
    return "\n".join(lines)


def detail_page(
    options: dict[str, list[str]],
    selected: dict[str, str] | None = None,
    *,
    asin: str = GOLD,
) -> str:
    buttons = ["Buy Now", "< Prev", "Back to Search"]
    for values in options.values():
        buttons.extend(values)
    return (
        "[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\n"
        f"asin: {asin}\nprice: 9.0\n"
        f"selected_options: {json.dumps(selected or {}, ensure_ascii=False)}\n"
        f"available_options: {json.dumps(options, ensure_ascii=False)}\n"
        f"可点击的按钮: {json.dumps(buttons, ensure_ascii=False)}"
    )


class PagingBandTest(unittest.TestCase):
    def test_first_page_is_not_paging(self) -> None:
        self.assertFalse(module.in_paging_band(20))

    def test_band_edges(self) -> None:
        self.assertTrue(module.in_paging_band(21))
        self.assertTrue(module.in_paging_band(60))
        self.assertFalse(module.in_paging_band(61))

    def test_none_is_not_paging(self) -> None:
        self.assertFalse(module.in_paging_band(None))


class BestCandidateTest(unittest.TestCase):
    def test_picks_smallest_rank(self) -> None:
        self.assertEqual(module.best_candidate([("a", 40), ("b", 25), ("c", 30)]), ("b", 25))

    def test_ignores_missing(self) -> None:
        self.assertEqual(module.best_candidate([("a", None), ("b", 33)]), ("b", 33))

    def test_all_missing_returns_none(self) -> None:
        self.assertIsNone(module.best_candidate([("a", None)]))

    def test_tie_keeps_first(self) -> None:
        self.assertEqual(module.best_candidate([("a", 25), ("b", 25)]), ("a", 25))


class PageTest(unittest.TestCase):
    def test_asins_in_order(self) -> None:
        page = search_page([(1, OTHER), (2, THIRD)])
        self.assertEqual(module.page_asins(page), [OTHER, THIRD])

    def test_non_search_page_has_no_asins(self) -> None:
        self.assertEqual(module.page_asins(detail_page({"x": ["y"]})), [])

    def test_contains(self) -> None:
        page = search_page([(25, GOLD)])
        self.assertTrue(module.page_contains(page, GOLD))
        self.assertFalse(module.page_contains(page, THIRD))


class MatchOptionTest(unittest.TestCase):
    options = {"颜色分类": ["a559", OPTION_PLAIN, "1 | 2中阮"]}

    def test_case_insensitive(self) -> None:
        self.assertEqual(module.match_option("A559", self.options), "a559")

    def test_pipe_vs_slash(self) -> None:
        self.assertEqual(module.match_option("1/2中阮", self.options), "1 | 2中阮")

    def test_prefix_is_not_a_match(self) -> None:
        self.assertIsNone(module.match_option("三色聚光灯", self.options))

    def test_unknown_returns_none(self) -> None:
        self.assertIsNone(module.match_option("不存在", self.options))


class AssembleTest(unittest.TestCase):
    def executed(self) -> list[dict]:
        return [
            {
                "tool_call": module.tool_call("next_page", {}, "c0"),
                "tool_name": "next_page",
                "parameters": {},
                "env_action": "click[Next >]",
                "observation": search_page([(25, GOLD)], page=2),
                "reward": 0.0,
                "done": False,
            }
        ]

    def test_message_roles(self) -> None:
        messages = module.assemble_messages("sys", "买点东西", self.executed())
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "c0")
        self.assertEqual(messages[3]["tool_call_id"], "c0")

    def test_trajectory_shape(self) -> None:
        trajectory = module.build_trajectory(
            trajectory_id="t1",
            task_id=7,
            instruction="买点东西",
            system_prompt="sys",
            executed=self.executed(),
            terminal={},
        )
        self.assertEqual(trajectory["task_id"], 7)
        self.assertEqual(len(trajectory["steps"]), 1)
        self.assertTrue(trajectory["done"])
        self.assertEqual(trajectory["status"], "done")
        self.assertEqual(trajectory["steps"][0]["step_index"], 0)


class ContractTest(unittest.TestCase):
    """生成的行必须过和老师轨迹同一套契约，否则进不了训练集。"""

    def full_executed(self) -> list[dict]:
        from shopping_grpo.environment.tools import tool_call_to_action

        options = {"颜色分类": [OPTION_PLAIN, OPTION_WITH_MACRO]}
        plan = [
            ("search_products", {"query": "珠宝补光灯"}, search_page([(1, OTHER), (2, THIRD)])),
            ("next_page", {}, search_page([(21, THIRD), (25, GOLD)], page=2)),
            ("open_product", {"asin": GOLD}, detail_page(options)),
            (
                "select_option",
                {"value": OPTION_WITH_MACRO},
                detail_page(options, {"颜色分类": OPTION_WITH_MACRO}),
            ),
            ("buy_now", {}, "[SHOPPING_OBSERVATION_V2]\npage_type: terminal"),
        ]
        executed = []
        for index, (name, arguments, observation) in enumerate(plan):
            executed.append(
                {
                    "tool_call": module.tool_call(name, arguments, f"c{index}"),
                    "tool_name": name,
                    "parameters": arguments,
                    "env_action": tool_call_to_action(name, arguments),
                    "observation": observation,
                    "reward": 1.0 if name == "buy_now" else 0.0,
                    "done": name == "buy_now",
                }
            )
        return executed

    def terminal(self) -> dict:
        return {
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v4",
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "purchase_success": True,
                "termination_reason": "gold_purchase",
            },
            "purchase": {"asin": GOLD},
            "goal": {"asin": GOLD},
            "termination_reason": "gold_purchase",
            "reward_valid": True,
        }

    def test_generated_trajectory_passes_collection_contract(self) -> None:
        from shopping_grpo.collection.sft import acceptance_reasons, build_sft_row
        from scripts.prepare_sft_curriculum import _validate_row

        trajectory = module.build_trajectory(
            trajectory_id="m10paging-1-0",
            task_id=1,
            instruction="买个补光灯",
            system_prompt="sys",
            executed=self.full_executed(),
            terminal=self.terminal(),
        )
        ok, reasons = acceptance_reasons(trajectory)
        self.assertTrue(ok, reasons)
        row = build_sft_row(trajectory)
        self.assertEqual(sorted(row), ["messages", "task_id", "tools", "trajectory_id"])
        _validate_row(row)
        self.assertEqual(row["messages"][-1]["content"], "购买已完成。")

    def test_rejects_when_reward_is_not_gold(self) -> None:
        from shopping_grpo.collection.sft import acceptance_reasons

        terminal = self.terminal()
        terminal["reward_detail"]["reward_type"] = "wrong_purchase"
        trajectory = module.build_trajectory(
            trajectory_id="m10paging-2-0",
            task_id=2,
            instruction="x",
            system_prompt="sys",
            executed=self.full_executed(),
            terminal=terminal,
        )
        ok, reasons = acceptance_reasons(trajectory)
        self.assertFalse(ok)
        self.assertIn("reward_v3_not_gold_purchase", reasons)


if __name__ == "__main__":
    unittest.main()
