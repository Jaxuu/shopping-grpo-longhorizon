"""`scripts/select_teacher_pattern_batch.py`：只收「有目标动作 **且** 买对」的轨迹。

为什么钉死它（2026-10-21）
--------------------------
`collect_sft_data.py` 的去重是「每题留一条**通过验收**的轨迹」，不看有没有目标动作。
探针实测换词重搜题上老师 90.9% 会换词，但同一题 4 次里只要有一次没换词也买对了，
那条就可能被留成「代表」⇒ 增量里混进「一次搜到就买」，正好是我们要纠正的行为。

五条不变量：

1. 目标动作 **和** 验收必须同时满足；只有动作没买对 → 丢（进不了池子）。
2. 每题只留一条；多条合格时留**步数最少**的（老师走完仪式会带一堆无关步骤）。
3. 与现有池子 / 评测集 task_id 重叠的一律跳过（建池脚本会直接报错）。
4. 产出的行必须过 `prepare_sft_curriculum._validate_row`，过不了的单条丢弃并记录，
   不能留到训练期才炸。
5. 动作口径与 `analyze_teacher_action_rate` 同源（直接 import），
   否则会出现「探针报 24.2%、筛出来的却不是那批」。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import select_teacher_pattern_batch as module  # noqa: E402
from shopping_grpo.environment.tools import tool_call_to_action  # noqa: E402

GOLD = "620296036484"
DECOY = "620296036485"
OTHER = "620296036486"


def _call(call_id, name, params):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(params, ensure_ascii=False)},
    }


def _step(name, params, call, done=False):
    return {
        "tool_name": name,
        "parameters": params,
        "tool_call": call,
        "env_action": tool_call_to_action(name, params),
        "done": done,
    }


def search_page(rows):
    lines = [
        "[SHOPPING_OBSERVATION_V2]",
        "page_type: search_results",
        "query: x",
        "Page 1 of 8 (Total results: 150; ranks 1-20)",
    ]
    lines += [f"{rank}|{asin}|9.0|品牌|类目|属性|标题" for rank, asin in rows]
    lines.append('可点击的按钮: ["Next >", "< Prev", "Back to Search"]')
    return "\n".join(lines)


def detail_page(asin):
    return (
        "[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\n"
        f"asin: {asin}\nprice: 9.0\n"
        '可点击的按钮: ["Buy Now", "< Prev", "Back to Search"]'
    )


def raw_trajectory(
    task_id, *, queries, opens, gold=True, attempt_index=0, gold_on_first_page=False
):
    """造一条能过 `acceptance_reasons` 与 `_validate_row` 的采集产物。

    默认第 1 个查询**只显示诱饵**，之后的查询才显示 gold（模拟「换词才找到」）。
    ``gold_on_first_page=True`` 时第 1 页就能看到 gold（对比候选场景需要）。
    """
    steps: list[dict] = []
    messages: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "任务"},
    ]
    last_search = ""
    for index, query in enumerate(queries):
        params = {"query": query}
        call = _call(f"s{index}", "search_products", params)
        visible = gold_on_first_page or index > 0
        rows = [(1, DECOY), (3, GOLD)] if visible else [(1, DECOY)]
        observation = search_page(rows)
        last_search = observation
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
        steps.append(_step("search_products", params, call))

    for index, asin in enumerate(opens):
        if index > 0:
            # 详情页上开不了下一个商品，必须先回搜索页（真实轨迹里也这么做）。
            params: dict = {}
            call = _call(f"k{index}", "back_to_search", params)
            messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": last_search}
            )
            steps.append(_step("back_to_search", params, call))
        params = {"asin": asin}
        call = _call(f"o{index}", "open_product", params)
        observation = detail_page(asin)
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
        steps.append(_step("open_product", params, call))

    params: dict = {}
    call = _call("b1", "buy_now", params)
    messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
    messages.append({"role": "tool", "tool_call_id": "b1", "content": "购买已完成。"})
    steps.append(_step("buy_now", params, call, done=True))

    return {
        "trajectory_id": f"t{task_id}-{attempt_index}",
        "task_id": task_id,
        "attempt_index": attempt_index,
        "status": "done",
        "done": True,
        "steps": steps,
        "messages": messages,
        "terminal_result": {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v5",
                "reward_type": "gold_purchase" if gold else "wrong_purchase",
                "reward_valid": True,
                "purchase_success": gold,
                "termination_reason": "gold_purchase" if gold else "max_steps",
                "target_asin_match": gold,
            },
        },
    }


class SelectTest(unittest.TestCase):
    def test_keeps_trajectory_with_target_action_and_gold(self):
        rows = [raw_trajectory(1, queries=["差查询", "好查询"], opens=[GOLD])]
        chosen, report = module.select(rows, scenario="requery")
        self.assertEqual(len(chosen), 1)
        self.assertEqual(report["counts"]["picked"], 1)

    def test_drops_trajectory_without_target_action(self):
        rows = [
            raw_trajectory(1, queries=["一个词就够"], opens=[GOLD], gold_on_first_page=True)
        ]
        chosen, report = module.select(rows, scenario="requery")
        self.assertEqual(chosen, [])
        self.assertEqual(report["counts"]["rejected_no_target_action"], 1)
        self.assertEqual(report["rejections"][0]["reason"], "no_target_action")

    def test_drops_wrong_purchase(self):
        rows = [raw_trajectory(1, queries=["差查询", "好查询"], opens=[DECOY], gold=False)]
        chosen, report = module.select(rows, scenario="requery")
        self.assertEqual(chosen, [])
        self.assertEqual(report["counts"]["rejected_not_accepted"], 1)

    def test_one_row_per_task_prefers_fewest_steps(self):
        long_row = raw_trajectory(
            7, queries=["a", "b"], opens=[DECOY, GOLD], attempt_index=0, gold_on_first_page=True
        )
        short_row = raw_trajectory(7, queries=["a", "b"], opens=[GOLD], attempt_index=1)
        chosen, report = module.select([long_row, short_row], scenario="requery")
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["trajectory_id"], "t7-1")
        self.assertEqual(report["counts"]["multi_candidate_tasks"], 1)

    def test_targets_filter(self):
        rows = [raw_trajectory(1, queries=["a", "b"], opens=[GOLD]),
                raw_trajectory(2, queries=["a", "b"], opens=[GOLD])]
        chosen, report = module.select(rows, scenario="requery", targets={2})
        self.assertEqual([row["task_id"] for row in chosen], [2])
        self.assertEqual(report["counts"]["skipped_not_in_targets"], 1)

    def test_excluded_task_ids_are_skipped(self):
        rows = [raw_trajectory(1, queries=["a", "b"], opens=[GOLD])]
        chosen, report = module.select(rows, scenario="requery", excluded={1})
        self.assertEqual(chosen, [])
        self.assertEqual(report["counts"]["skipped_excluded"], 1)

    def test_max_steps_filter(self):
        rows = [
            raw_trajectory(
                1, queries=["a", "b"], opens=[DECOY, GOLD], gold_on_first_page=True
            )
        ]
        chosen, report = module.select(rows, scenario="requery", max_steps=3)
        self.assertEqual(chosen, [])
        self.assertEqual(report["counts"]["rejected_too_long"], 1)

    def test_compare_counts_distinct_opens(self):
        rows = [
            raw_trajectory(
                1, queries=["一个词"], opens=[DECOY, GOLD], gold_on_first_page=True
            )
        ]
        chosen, report = module.select(rows, scenario="compare")
        self.assertEqual(len(chosen), 1)
        self.assertEqual(report["action"], "multi_open")

    def test_action_override_decouples_source_scenario_from_action(self):
        """翻页题里「换词找到」的轨迹：默认口径（翻页）会丢，覆盖后能收。

        这是 2026-10-21 全量探针的发现：翻页题 40 道买对 52 次，其中 49 次没翻页，
        是换词找到的。这批轨迹已经在探针 raw 里，改个目标动作就能白捡。
        """
        row = raw_trajectory(9, queries=["差查询", "好查询"], opens=[GOLD])

        dropped, report = module.select([row], scenario="paging")
        self.assertEqual(dropped, [])
        self.assertEqual(report["counts"]["rejected_no_target_action"], 1)
        self.assertEqual(report["action"], "used_next_page")

        chosen, report = module.select(
            [row], scenario="paging", action_override="requeried"
        )
        self.assertEqual(len(chosen), 1)
        self.assertEqual(report["action"], "requeried")
        self.assertEqual(report["action_override"], "requeried")

    def test_targets_scenario_filters_mixed_target_list(self):
        """探针清单三个场景混在一个文件里，必须按 scenario 字段分。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "targets.jsonl"
            path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in [
                        {"task_id": 1, "scenario": "paging"},
                        {"task_id": 2, "scenario": "requery"},
                        {"task_id": 3, "scenario": "paging"},
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(module.task_ids_from(path), {1, 2, 3})
            self.assertEqual(module.task_ids_from(path, "paging"), {1, 3})
            self.assertEqual(module.task_ids_from(path, "compare"), set())


class SftRowTest(unittest.TestCase):
    def test_rows_have_four_keys_and_pass_curriculum_validation(self):
        rows = [raw_trajectory(1, queries=["a", "b"], opens=[GOLD])]
        chosen, _ = module.select(rows, scenario="requery")
        kept, dropped = module.to_sft_rows(chosen)
        self.assertEqual(dropped, [])
        self.assertEqual(set(kept[0]), {"trajectory_id", "task_id", "messages", "tools"})

    def test_assistant_text_is_kept_and_masked_by_loss_scope(self):
        """行里**保留**老师的自然语言，训练时由 `loss_scope="tool-call-only"` 掩掉。

        不要在这里删：历史池子（1,252 行）都带着 content，删了格式就不一致了；
        而且 content 被掩掉不影响梯度，删它只会让审计时看不到老师在想什么。
        """
        row = raw_trajectory(1, queries=["a", "b"], opens=[GOLD])
        row["messages"][2]["content"] = "我先想想……"
        chosen, _ = module.select([row], scenario="requery")
        kept, _ = module.to_sft_rows(chosen)
        assistants = [m for m in kept[0]["messages"] if m["role"] == "assistant"]
        self.assertEqual(assistants[0]["content"], "我先想想……")

    def test_profile_reports_rank_buckets(self):
        rows = [raw_trajectory(1, queries=["a", "b"], opens=[GOLD])]
        chosen, _ = module.select(rows, scenario="requery")
        kept, _ = module.to_sft_rows(chosen)
        info = module.profile(kept)
        # gold 在第二页查询里排第 3 ⇒ 桶是 2-3
        self.assertEqual(info["rank_buckets"], {"2-3": 1})
        self.assertEqual(info["rank_ge2"], 1)


class MainTest(unittest.TestCase):
    def test_writes_kept_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "raw.jsonl"
            raw.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in [
                        raw_trajectory(1, queries=["a", "b"], opens=[GOLD]),
                        raw_trajectory(
                            2, queries=["a"], opens=[GOLD], gold_on_first_page=True
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            pool = tmp_path / "pool.jsonl"
            pool.write_text("", encoding="utf-8")
            evaluation = tmp_path / "eval.jsonl"
            evaluation.write_text("", encoding="utf-8")

            code = module.main(
                [
                    str(raw),
                    "--scenario",
                    "requery",
                    "--out-dir",
                    str(tmp_path / "out"),
                    "--exclude-pool",
                    str(pool),
                    "--exclude-evaluation",
                    str(evaluation),
                ]
            )
            self.assertEqual(code, 0)
            kept = [
                json.loads(line)
                for line in (tmp_path / "out" / "kept.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["task_id"] for row in kept], [1])
            report = json.loads(
                (tmp_path / "out" / "selection_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["kept"], 1)
            self.assertEqual(report["counts"]["rejected_no_target_action"], 1)

    def test_cli_action_and_targets_scenario(self):
        """命令行把 --action / --targets-scenario 接上了（argparse 漏接会静默失效）。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "raw.jsonl"
            raw.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in [
                        raw_trajectory(1, queries=["差查询", "好查询"], opens=[GOLD]),
                        raw_trajectory(2, queries=["差查询", "好查询"], opens=[GOLD]),
                    ]
                ),
                encoding="utf-8",
            )
            targets = tmp_path / "targets.jsonl"
            targets.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in [
                        {"task_id": 1, "scenario": "paging"},
                        {"task_id": 2, "scenario": "requery"},
                    ]
                ),
                encoding="utf-8",
            )
            empty = tmp_path / "empty.jsonl"
            empty.write_text("", encoding="utf-8")

            code = module.main(
                [
                    str(raw),
                    "--scenario",
                    "paging",
                    "--action",
                    "requeried",
                    "--targets",
                    str(targets),
                    "--targets-scenario",
                    "paging",
                    "--out-dir",
                    str(tmp_path / "out"),
                    "--exclude-pool",
                    str(empty),
                    "--exclude-evaluation",
                    str(empty),
                ]
            )
            self.assertEqual(code, 0)
            kept = [
                json.loads(line)
                for line in (tmp_path / "out" / "kept.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            # 只收 paging 场景的题（task 1），换词动作命中
            self.assertEqual([row["task_id"] for row in kept], [1])

    def test_empty_selection_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "raw.jsonl"
            raw.write_text(
                json.dumps(raw_trajectory(1, queries=["a"], opens=[GOLD])) + "\n",
                encoding="utf-8",
            )
            pool = tmp_path / "pool.jsonl"
            pool.write_text("", encoding="utf-8")
            code = module.main(
                [
                    str(raw),
                    "--scenario",
                    "requery",
                    "--out-dir",
                    str(tmp_path / "out"),
                    "--exclude-pool",
                    str(pool),
                    "--exclude-evaluation",
                    str(pool),
                ]
            )
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
