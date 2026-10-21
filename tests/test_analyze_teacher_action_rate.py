"""`scripts/analyze_teacher_action_rate.py` 的判定口径固定。

钉死五件事（改动前先看这里）：

1. **三个场景各有唯一的目标动作**，不许混：翻页 = 出现 `next_page`；
   换词重搜 = **两个及以上不同** 查询词；对比候选 = 打开 **两个及以上不同** ASIN。
2. **`think` 不算动作**：它在环境里不产生新观测，计进步数会让「平均步数」虚高。
3. **题级命中率与轨迹级命中率是两件事**：至少一次命中（`task_hit`）和每次都命中
   （`task_always`）不能混用——采不采得动看轨迹级，够不够覆盖看题级。
4. **阈值 20% 是预登记的**，边界必须是 `>=`（0.20 判可采，0.199 判采不到）。
5. **必须读 `raw.jsonl`**：`accepted.jsonl` 只剩成功轨迹，会把分母换掉。
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

import analyze_teacher_action_rate as module  # noqa: E402


def step(name, **parameters):
    return {"tool_name": name, "parameters": parameters, "observation": ""}


def trajectory(task_id, steps, *, gold=False, attempt_index=0, status="done"):
    return {
        "task_id": task_id,
        "attempt_index": attempt_index,
        "status": status,
        "steps": steps,
        "terminal_result": {"reward_detail": {"target_asin_match": gold}},
    }


class SummarizeTest(unittest.TestCase):
    def test_next_page_detected(self):
        summary = module._summarize(["search_products", "next_page", "buy_now"], [], [])
        self.assertTrue(summary["used_next_page"])
        self.assertEqual(summary["next_page_calls"], 1)

    def test_requery_needs_two_distinct_queries(self):
        same = module._summarize(
            ["search_products", "search_products"], ["lamp", "  Lamp "], []
        )
        self.assertFalse(same["requeried"], "同一个词搜两次不算换词")
        self.assertEqual(same["distinct_queries"], 1)

        different = module._summarize(
            ["search_products", "search_products"], ["lamp", "desk lamp"], []
        )
        self.assertTrue(different["requeried"])

    def test_multi_open_needs_two_distinct_asins(self):
        same = module._summarize(
            ["open_product", "open_product"], [], ["620296036484", "620296036484"]
        )
        self.assertFalse(same["multi_open"])
        different = module._summarize(
            ["open_product", "open_product"], [], ["620296036484", "620296036485"]
        )
        self.assertTrue(different["multi_open"])

    def test_think_is_not_a_step(self):
        summary = module._summarize(["think", "search_products", "think"], ["lamp"], [])
        self.assertEqual(summary["tool_calls"], 1)

    def test_give_up_detected(self):
        summary = module._summarize(["finish_without_purchase"], [], [])
        self.assertTrue(summary["gave_up"])


class ExtractTrajectoryTest(unittest.TestCase):
    def test_reads_steps_and_reward_detail(self):
        row = trajectory(
            7,
            [
                step("search_products", query="lamp"),
                step("open_product", asin="620296036484"),
                step("next_page"),
                step("open_product", asin="620296036485"),
                step("buy_now"),
            ],
            gold=True,
        )
        record = module.extract_trajectory(row)
        self.assertEqual(record["task_id"], 7)
        self.assertTrue(record["used_next_page"])
        self.assertTrue(record["multi_open"])
        self.assertTrue(record["bought_gold"])
        self.assertEqual(record["opened_distinct"], 2)

    def test_empty_steps_does_not_crash(self):
        record = module.extract_trajectory({"task_id": 1, "steps": []})
        self.assertEqual(record["tool_calls"], 0)
        self.assertFalse(record["used_next_page"])

    def test_bought_gold_uses_reward_detail_not_reward_type(self):
        row = trajectory(3, [step("buy_now")], gold=True)
        row["terminal_result"]["reward_detail"]["reward_type"] = "wrong_purchase"
        record = module.extract_trajectory(row)
        self.assertTrue(record["bought_gold"], "买对但超预算也要算买对")


class ExtractMessagesTest(unittest.TestCase):
    def test_reads_pool_format(self):
        row = {
            "task_id": 9,
            "messages": [
                {"role": "system", "content": "..."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_products",
                                "arguments": json.dumps({"query": "lamp"}),
                            }
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_products",
                                "arguments": json.dumps({"query": "desk lamp"}),
                            }
                        },
                        {"function": {"name": "next_page", "arguments": "{}"}},
                    ],
                },
            ],
        }
        record = module.extract_messages(row)
        self.assertTrue(record["requeried"])
        self.assertTrue(record["used_next_page"])

    def test_broken_arguments_are_skipped(self):
        row = {
            "task_id": 9,
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "search_products", "arguments": "{"}}],
                }
            ],
        }
        record = module.extract_messages(row)
        self.assertEqual(record["distinct_queries"], 0)
        self.assertEqual(record["tool_calls"], 1)


class ScenarioSummaryTest(unittest.TestCase):
    def test_task_hit_and_always_are_distinct(self):
        records = [
            {"task_id": 1, "used_next_page": True, "bought_gold": False, "requeried": False, "multi_open": False, "gave_up": False, "tool_calls": 5, "opened_distinct": 1, "distinct_queries": 1},
            {"task_id": 1, "used_next_page": False, "bought_gold": False, "requeried": False, "multi_open": False, "gave_up": False, "tool_calls": 5, "opened_distinct": 1, "distinct_queries": 1},
            {"task_id": 2, "used_next_page": False, "bought_gold": False, "requeried": False, "multi_open": False, "gave_up": False, "tool_calls": 5, "opened_distinct": 1, "distinct_queries": 1},
        ]
        summary = module.summarize_scenario(records, "used_next_page")
        self.assertEqual(summary["tasks"], 2)
        self.assertEqual(summary["trajectories"], 3)
        self.assertEqual(summary["task_hit"], 1)
        self.assertEqual(summary["task_always"], 0)
        self.assertAlmostEqual(summary["hit_rate"], 1 / 3)

    def test_gold_without_action_counts_alternative_route(self):
        records = [
            {"task_id": 1, "used_next_page": False, "bought_gold": True, "requeried": True, "multi_open": False, "gave_up": False, "tool_calls": 9, "opened_distinct": 1, "distinct_queries": 2},
        ]
        summary = module.summarize_scenario(records, "used_next_page")
        self.assertEqual(summary["gold"], 1)
        self.assertEqual(summary["gold_without_action"], 1)
        self.assertEqual(summary["requeried"], 1)

    def test_empty_records_gives_zero_not_crash(self):
        summary = module.summarize_scenario([], "used_next_page")
        self.assertEqual(summary["trajectories"], 0)
        self.assertEqual(summary["hit_rate"], 0.0)


class VerdictTest(unittest.TestCase):
    def test_threshold_is_inclusive(self):
        self.assertEqual(module.verdict(0.20), "可采集")
        self.assertEqual(module.verdict(0.199), "采不到，才考虑合成")

    def test_threshold_constant_is_pregistration(self):
        self.assertAlmostEqual(module.PROBE_THRESHOLD, 0.20)

    def test_custom_threshold(self):
        self.assertEqual(module.verdict(0.5, threshold=0.6), "采不到，才考虑合成")


class RenderTest(unittest.TestCase):
    def test_render_lists_every_scenario(self):
        records = {
            "paging": [
                {"task_id": 1, "used_next_page": True, "bought_gold": False, "requeried": False, "multi_open": False, "gave_up": False, "tool_calls": 4, "opened_distinct": 1, "distinct_queries": 1}
            ]
        }
        summaries = {
            scenario: module.summarize_scenario(rows, module.SCENARIO_ACTION[scenario])
            for scenario, rows in records.items()
        }
        text = module.render(summaries)
        self.assertIn("翻页", text)
        self.assertIn("可采集", text)
        self.assertNotIn("换词重搜（requery） |", text)

    def test_detail_groups_by_task(self):
        records = [
            {"task_id": 5, "used_next_page": True, "bought_gold": False, "requeried": False, "multi_open": False, "gave_up": False, "tool_calls": 4, "opened_distinct": 1, "distinct_queries": 1},
            {"task_id": 5, "used_next_page": False, "bought_gold": True, "requeried": False, "multi_open": False, "gave_up": False, "tool_calls": 4, "opened_distinct": 1, "distinct_queries": 1},
        ]
        text = module.render_detail(records, {5: "paging"})
        self.assertIn("task 5 [paging] 翻页 1/2", text)
        self.assertIn("买对 1/2", text)


class HealthTest(unittest.TestCase):
    """没跑起来的轨迹必须先暴露出来，否则命中率是假 0。"""

    def test_traceback_tail_keeps_last_lines(self):
        text = "Traceback (most recent call last):\n  File a.py\n  File b.py\nValueError: bad"
        self.assertEqual(
            module.traceback_tail(text), "File a.py\nFile b.py\nValueError: bad"
        )
        self.assertEqual(module.traceback_tail(text, lines=1), "ValueError: bad")

    def test_traceback_tail_handles_empty(self):
        self.assertEqual(module.traceback_tail(None), "")
        self.assertEqual(module.traceback_tail("  \n \n"), "")

    def test_healthy_records_report_no_error(self):
        records = [{"status": "done", "error_type": ""}]
        text = module.render_health(records)
        self.assertIn("无异常轨迹", text)
        self.assertNotIn("判定作废", text)

    def test_error_type_and_message_are_printed(self):
        records = [
            {
                "status": "error",
                "error_type": "HTTPError",
                "error_message": "HTTP Error 400: Bad Request",
                "error_tail": "ValueError: bad temperature",
            }
        ]
        text = module.render_health(records)
        self.assertIn("HTTPError", text)
        self.assertIn("400", text)
        self.assertIn("ValueError: bad temperature", text)
        self.assertIn("不可用于判定", text, "100% 异常必须报警")

    def test_error_share_below_ceiling_does_not_warn(self):
        records = [
            {"status": "done", "error_type": ""},
            {"status": "done", "error_type": ""},
            {"status": "done", "error_type": ""},
            {"status": "error", "error_type": "HTTPError", "error_message": "x"},
        ]
        text = module.render_health(records)
        self.assertIn("异常占比 25%", text)
        self.assertNotIn("不可用于判定", text)

    def test_error_share_counts_only_errors(self):
        records = [{"error_type": "HTTPError"}, {"error_type": ""}, {"error_type": ""}]
        self.assertAlmostEqual(module.error_share(records), 1 / 3)

    def test_error_fields_survive_extraction(self):
        row = trajectory(1, [step("buy_now")])
        row["status"] = "error"
        row["error"] = {
            "type": "HTTPError",
            "message": "HTTP Error 400",
            "traceback": "a\nb\nHTTPError: HTTP Error 400",
        }
        record = module.extract_trajectory(row)
        self.assertEqual(record["error_type"], "HTTPError")
        self.assertIn("400", record["error_message"])
        self.assertIn("HTTPError: HTTP Error 400", record["error_tail"])


class LoadScenariosTest(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        self.assertEqual(module.load_scenarios(Path("no/such/file.jsonl")), {})

    def test_reads_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "targets.jsonl"
            path.write_text(
                json.dumps({"task_id": 11, "scenario": "requery"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(module.load_scenarios(path), {11: "requery"})


class MainTest(unittest.TestCase):
    def test_unknown_task_ids_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.jsonl"
            raw.write_text(
                json.dumps(trajectory(999, [step("next_page")])) + "\n", encoding="utf-8"
            )
            manifest = Path(tmp) / "targets.jsonl"
            manifest.write_text(
                json.dumps({"task_id": 1, "scenario": "paging"}) + "\n", encoding="utf-8"
            )
            code = module.main([str(raw), "--targets", str(manifest)])
            self.assertEqual(code, 1, "一条都对不上时必须报错，不能静默给 0")

    def test_json_output_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.jsonl"
            raw.write_text(
                json.dumps(trajectory(1, [step("next_page"), step("buy_now")])) + "\n",
                encoding="utf-8",
            )
            manifest = Path(tmp) / "targets.jsonl"
            manifest.write_text(
                json.dumps({"task_id": 1, "scenario": "paging"}) + "\n", encoding="utf-8"
            )
            code = module.main([str(raw), "--targets", str(manifest), "--json"])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
