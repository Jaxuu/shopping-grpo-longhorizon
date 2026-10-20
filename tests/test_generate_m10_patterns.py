"""M10-R 两个新合成脚本的测试：换查询重搜 / 对比候选。

重点不是覆盖每一行，而是钉住三类会静默错的地方：
1. 选题口径（什么题算「该换查询」「该对比」）；
2. 跑批容错（单题异常、环境提前结束不能带崩整批）；
3. 生成的行必须过和老师轨迹同一套契约。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT / "scripts"), str(ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import generate_m10_compare as compare_module  # noqa: E402
import generate_m10_requery as requery_module  # noqa: E402

GOLD = "620296036484"
DECOY = "11111111"
THIRD = "22222222"
OPTION = "三色聚光灯（送星光镜带夹子）"


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


def detail_page(asin: str = GOLD) -> str:
    options = {"颜色分类": [OPTION]}
    buttons = ["Buy Now", "< Prev", "Back to Search", OPTION]
    return (
        "[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\n"
        f"asin: {asin}\nprice: 9.0\n"
        f"selected_options: {json.dumps({}, ensure_ascii=False)}\n"
        f"available_options: {json.dumps(options, ensure_ascii=False)}\n"
        f"可点击的按钮: {json.dumps(buttons, ensure_ascii=False)}"
    )


def terminal_ok() -> dict:
    return {
        "done": True,
        "over": True,
        "termination_reason": "gold_purchase",
        "reward_detail": {
            "reward_version": "shopsimulator-reward-v4",
            "reward_type": "gold_purchase",
            "reward_valid": True,
            "purchase_success": True,
            "termination_reason": "gold_purchase",
        },
        "purchase": {"asin": GOLD},
        "goal": {"asin": GOLD},
    }


class FakeEnv:
    """按脚本回放的最小环境；记录收到的动作，用于断言没发不该发的动作。"""

    def __init__(self, script: dict[int, list], **kwargs) -> None:
        self.script = script
        self.include_trace_target = False
        self.actions: list[str] = []
        self.task_id = -1
        self.cursor = 0

    def __enter__(self) -> "FakeEnv":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def reset(self, task_id: int) -> dict:
        self.task_id = int(task_id)
        self.cursor = 0
        return {"instruction": "Instruction: 测试题", "goal_options": []}

    def step(self, action: str) -> dict:
        self.actions.append(action)
        responses = self.script[self.task_id]
        if self.cursor >= len(responses):
            raise AssertionError(f"task {self.task_id} 收到了多余的动作：{action}")
        response = responses[self.cursor]
        self.cursor += 1
        if isinstance(response, Exception):
            raise response
        return response


class RequeryPairTest(unittest.TestCase):
    def test_picks_best_good_and_worst_bad(self) -> None:
        pair = requery_module.pick_query_pair([("类目", 5), ("属性", 12), ("开头", 90)])
        self.assertEqual(pair, ("类目", 5, "开头", 90))

    def test_prefers_known_rank_over_missing(self) -> None:
        """差查询优先要「有结果但排在后面」的，而不是「召回里没有」。"""
        pair = requery_module.pick_query_pair([("好", 3), ("差", None), ("更差", 200)])
        self.assertEqual(pair, ("好", 3, "更差", 200))

    def test_missing_rank_is_acceptable_when_alone(self) -> None:
        pair = requery_module.pick_query_pair([("好", 3), ("差", None)])
        self.assertEqual(pair, ("好", 3, "差", None))

    def test_no_pair_when_good_is_absent(self) -> None:
        self.assertIsNone(requery_module.pick_query_pair([("甲", 30), ("乙", 40)]))

    def test_no_pair_when_bad_is_absent(self) -> None:
        self.assertIsNone(requery_module.pick_query_pair([("甲", 3), ("乙", 12)]))

    def test_band_boundaries(self) -> None:
        self.assertIsNotNone(requery_module.pick_query_pair([("好", 20), ("差", 61)]))
        self.assertIsNone(requery_module.pick_query_pair([("好", 21), ("差", 61)]))
        self.assertIsNone(requery_module.pick_query_pair([("好", 20), ("差", 60)]))


class ComparePickTest(unittest.TestCase):
    class Hit:
        def __init__(self, asin: str, rank: int) -> None:
            self.asin = asin
            self.rank = rank

    class Searcher:
        def __init__(self, hits: list) -> None:
            self.hits = hits

        def search(self, query: str, k: int = 150) -> list:
            return self.hits

    def test_returns_gold_rank_and_top_asin(self) -> None:
        searcher = self.Searcher([self.Hit(DECOY, 1), self.Hit(GOLD, 7)])
        self.assertEqual(compare_module._rank_and_top(searcher, "q", GOLD), (7, DECOY))

    def test_missing_gold_returns_none(self) -> None:
        searcher = self.Searcher([self.Hit(DECOY, 1)])
        self.assertIsNone(compare_module._rank_and_top(searcher, "q", GOLD))

    def test_empty_result_returns_no_top(self) -> None:
        searcher = self.Searcher([])
        self.assertIsNone(compare_module._rank_and_top(searcher, "q", GOLD))

    def test_gold_at_top_has_no_decoy(self) -> None:
        """gold 就是第 1 名时没有诱饵，这种题不该进对比批次。"""
        searcher = self.Searcher([self.Hit(GOLD, 1), self.Hit(DECOY, 2)])
        rank, top = compare_module._rank_and_top(searcher, "q", GOLD)
        self.assertEqual(rank, 1)
        self.assertEqual(top, GOLD)


class BatchHarness:
    """把某个脚本的 generate() 跑在假环境上，返回摘要与落盘内容。"""

    def __init__(self, module) -> None:
        self.module = module

    def run(self, script: dict[int, list], targets: list[dict], **kwargs) -> tuple[dict, dict]:
        import shopping_grpo.environment.client as client_module

        captured: dict = {}

        def factory(**init_kwargs):
            env = FakeEnv(script, **init_kwargs)
            captured["env"] = env
            return env

        original = client_module.ShopAgentEnv
        client_module.ShopAgentEnv = factory
        try:
            with tempfile.TemporaryDirectory() as tmp:
                summary = self.module.generate(
                    targets, base_url="http://fake", out_dir=Path(tmp), **kwargs
                )
                for name in ("kept", "raw", "rejections"):
                    path = Path(tmp) / f"{name}.jsonl"
                    captured[name] = path.read_text(encoding="utf-8") if path.is_file() else ""
        finally:
            client_module.ShopAgentEnv = original
        return summary, captured


class RequeryLoopTest(unittest.TestCase):
    harness = BatchHarness(requery_module)

    def target(self, task_id: int = 7) -> dict:
        return {
            "task_id": task_id,
            "asin": GOLD,
            "bad_query": "类目",
            "bad_rank": 200,
            "good_query": "属性",
            "good_rank": 3,
            "instruction": "x",
        }

    def success_script(self) -> list[dict]:
        return [
            {"observation": search_page([(1, DECOY), (2, THIRD)])},
            {"observation": "[SHOPPING_OBSERVATION_V2]\npage_type: search_input"},
            {"observation": search_page([(1, THIRD), (3, GOLD)])},
            {"observation": detail_page()},
            terminal_ok(),
        ]

    def test_success_path(self) -> None:
        summary, captured = self.harness.run({7: self.success_script()}, [self.target()])
        self.assertEqual(summary["kept"], 1, captured["rejections"])
        self.assertEqual(len(captured["kept"].splitlines()), 1)

    def test_rejects_when_bad_query_already_shows_gold(self) -> None:
        script = [{"observation": search_page([(1, GOLD)])}]
        summary, _ = self.harness.run({7: script}, [self.target()])
        self.assertEqual(
            summary["reject_reasons"], {"gold_on_first_page_with_bad_query": 1}
        )

    def test_rejects_when_bad_query_returns_nothing(self) -> None:
        script = [{"observation": search_page([])}]
        summary, _ = self.harness.run({7: script}, [self.target()])
        self.assertEqual(summary["reject_reasons"], {"bad_query_returned_nothing": 1})

    def test_rejects_when_good_query_misses_gold(self) -> None:
        script = [
            {"observation": search_page([(1, DECOY)])},
            {"observation": "[SHOPPING_OBSERVATION_V2]\npage_type: search_input"},
            {"observation": search_page([(1, THIRD)])},
        ]
        summary, _ = self.harness.run({7: script}, [self.target()])
        self.assertEqual(
            summary["reject_reasons"], {"gold_not_on_first_page_with_good_query": 1}
        )

    def test_one_bad_task_does_not_stop_the_batch(self) -> None:
        script = {7: [RuntimeError("断了")], 8: self.success_script()}
        summary, _ = self.harness.run(
            script, [self.target(7), self.target(8)]
        )
        self.assertEqual(summary["kept"], 1)
        self.assertEqual(summary["reject_reasons"], {"exception": 1})


class CompareLoopTest(unittest.TestCase):
    harness = BatchHarness(compare_module)

    def target(self, task_id: int = 7) -> dict:
        return {
            "task_id": task_id,
            "asin": GOLD,
            "decoy_asin": DECOY,
            "query": "q",
            "rank": 7,
            "instruction": "x",
        }

    def success_script(self) -> list[dict]:
        return [
            {"observation": search_page([(1, DECOY), (7, GOLD)])},
            {"observation": detail_page(DECOY)},
            {"observation": search_page([(1, DECOY), (7, GOLD)])},
            {"observation": detail_page(GOLD)},
            terminal_ok(),
        ]

    def test_success_path_opens_decoy_then_gold(self) -> None:
        summary, captured = self.harness.run({7: self.success_script()}, [self.target()])
        self.assertEqual(summary["kept"], 1, captured["rejections"])
        actions = captured["env"].actions
        self.assertEqual(len(actions), 5)
        self.assertIn("click[Back to Search]", actions)

    def test_rejects_when_decoy_not_on_page(self) -> None:
        script = [{"observation": search_page([(7, GOLD)])}]
        summary, _ = self.harness.run({7: script}, [self.target()])
        self.assertEqual(summary["reject_reasons"], {"decoy_not_on_first_page": 1})

    def test_rejects_when_gold_disappears_after_back(self) -> None:
        script = [
            {"observation": search_page([(1, DECOY), (7, GOLD)])},
            {"observation": detail_page(DECOY)},
            {"observation": search_page([(1, DECOY)])},
        ]
        summary, _ = self.harness.run({7: script}, [self.target()])
        self.assertEqual(
            summary["reject_reasons"], {"gold_lost_after_back_to_search": 1}
        )

    def test_environment_terminating_after_open_does_not_crash(self) -> None:
        script = [
            {"observation": search_page([(1, DECOY), (7, GOLD)])},
            {
                "observation": detail_page(DECOY),
                "done": True,
                "termination_reason": "repeat_loop",
                "reward_detail": {"termination_reason": "repeat_loop"},
            },
        ]
        summary, _ = self.harness.run({7: script}, [self.target()])
        self.assertEqual(summary["reject_reasons"], {"terminated_open_decoy": 1})

    def test_one_bad_task_does_not_stop_the_batch(self) -> None:
        script = {7: [RuntimeError("断了")], 8: self.success_script()}
        summary, _ = self.harness.run(script, [self.target(7), self.target(8)])
        self.assertEqual(summary["kept"], 1)
        self.assertEqual(summary["reject_reasons"], {"exception": 1})


if __name__ == "__main__":
    unittest.main()
