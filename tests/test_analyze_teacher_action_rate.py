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
    def _record(self, **overrides):
        base = {
            "task_id": 1,
            "used_next_page": False,
            "bought_gold": False,
            "accepted": False,
            "requeried": False,
            "multi_open": False,
            "gave_up": False,
            "tool_calls": 4,
            "opened_distinct": 1,
            "distinct_queries": 1,
        }
        base.update(overrides)
        return base

    def test_render_lists_every_scenario(self):
        records = {"paging": [self._record(used_next_page=True, bought_gold=True, accepted=True)]}
        summaries = {
            scenario: module.summarize_scenario(rows, module.SCENARIO_ACTION[scenario])
            for scenario, rows in records.items()
        }
        text = module.render(summaries)
        self.assertIn("翻页", text)
        self.assertIn("可采集", text)
        self.assertNotIn("换词重搜（requery） |", text)

    def test_action_without_gold_is_not_harvestable(self):
        """翻了页但没买对 ⇒ 进不了池子，不能算可收。这是用户 2026-10-21 指出的口径。"""
        records = [
            self._record(used_next_page=True, bought_gold=False, accepted=False),
            self._record(used_next_page=True, bought_gold=True, accepted=True),
        ]
        summary = module.summarize_scenario(records, "used_next_page")
        self.assertEqual(summary["hits"], 2)
        self.assertEqual(summary["harvest"], 1)
        self.assertEqual(summary["action_only"], 1)
        self.assertAlmostEqual(summary["harvest_rate"], 0.5)
        self.assertAlmostEqual(summary["hit_rate"], 1.0)

    def test_verdict_uses_harvest_rate_not_action_rate(self):
        """动作率 100% 但一条都没买对 ⇒ 判「采不到」。"""
        records = [
            self._record(used_next_page=True, bought_gold=False, accepted=False),
        ]
        summaries = {"paging": module.summarize_scenario(records, "used_next_page")}
        text = module.render(summaries)
        self.assertIn("可收率 0.0%", text)
        self.assertIn("动作率 100.0%", text)
        self.assertIn("采不到，才考虑合成", text)

    def test_projected_harvest_scales_with_candidates(self):
        records = [self._record(used_next_page=True, bought_gold=True, accepted=True)]
        summary = module.summarize_scenario(records, "used_next_page")
        # paging 全量 158 题 × 题级可收率 100%
        self.assertEqual(module.projected_harvest("paging", summary), 158)

    def test_projection_counts_tasks_not_attempts(self):
        """池子每题只留一条，所以外推**不能**乘每题次数。

        2026-10-21：旧公式 `harvest_rate × 题数 × 次数` 把外推放大 4 倍
        （换词批报 160 条、对比批报 220 条）。同一题采 4 次也只进池子 1 条。
        """
        records = [
            self._record(task_id=1, attempt_index=index, used_next_page=True,
                         bought_gold=True, accepted=True)
            for index in range(4)
        ]
        summary = module.summarize_scenario(records, "used_next_page")
        self.assertEqual(summary["planned_attempts"], 4)
        self.assertEqual(summary["harvest"], 4)          # 轨迹级
        self.assertEqual(summary["task_harvest"], 1)     # 题级（能进池子的条数）
        self.assertEqual(module.projected_harvest("paging", summary), 158)

    def test_conditional_rate_separates_capability_from_success(self):
        """买对的轨迹里有多少做了动作：把「不会做」和「做了但买不对」分开。"""
        records = [
            self._record(used_next_page=True, bought_gold=True, accepted=True),
            self._record(used_next_page=False, bought_gold=True, accepted=True),
            self._record(used_next_page=False, bought_gold=True, accepted=True),
            self._record(used_next_page=False, bought_gold=False, accepted=False),
        ]
        summary = module.summarize_scenario(records, "used_next_page")
        self.assertAlmostEqual(summary["harvest_given_gold"], 1 / 3)
        self.assertAlmostEqual(summary["harvest_rate"], 0.25)

    def test_candidate_counts_match_target_files(self):
        """SCENARIO_CANDIDATES 必须和清单口径一致，否则「预计可收」是假的。"""
        rows = module.load_jsonl(
            ROOT / "data" / "collection" / "m10_paging_targets.jsonl"
        )
        eligible = [row for row in rows if int(row["rank"]) <= 60]
        self.assertEqual(module.SCENARIO_CANDIDATES["paging"], len(eligible))
        for scenario, name in (
            ("requery", "m10_requery_targets.jsonl"),
            ("compare", "m10_compare_targets.jsonl"),
        ):
            count = len(module.load_jsonl(ROOT / "data" / "collection" / name))
            self.assertEqual(module.SCENARIO_CANDIDATES[scenario], count)

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

    def test_high_error_share_exits_3(self):
        """退出码 3 = 判定无效，调用方靠它决定要不要继续跑全量。"""
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.jsonl"
            bad = trajectory(1, [])
            bad["status"] = "error"
            bad["error"] = {"type": "HTTPError", "message": "HTTP Error 400"}
            raw.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            manifest = Path(tmp) / "targets.jsonl"
            manifest.write_text(
                json.dumps({"task_id": 1, "scenario": "paging"}) + "\n", encoding="utf-8"
            )
            code = module.main([str(raw), "--targets", str(manifest)])
            self.assertEqual(code, 3)

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


class CoverageTest(unittest.TestCase):
    """采集中断后只剩部分数据，判定表必须显式标「样本不完整」。

    2026-10-21：480 条跑到 339 条被一条异常中止，若只看判定表会误当成最终结论。
    """

    @staticmethod
    def summaries(tasks):
        return {
            "paging": {
                "tasks": tasks,
                "trajectories": tasks * 4,
                "planned_attempts": 4,
            }
        }

    def test_complete_sample_has_no_warning(self):
        scenarios = {index: "paging" for index in range(40)}
        text = module.render_coverage(self.summaries(40), scenarios)
        self.assertNotIn("样本不完整", text)
        self.assertIn("| 40 | 40 |", text)

    def test_partial_sample_warns_and_points_to_resume(self):
        scenarios = {index: "paging" for index in range(40)}
        text = module.render_coverage(self.summaries(35), scenarios)
        self.assertIn("样本不完整", text)
        self.assertIn("35/40", text)
        self.assertIn("续跑", text)


class TruncatedRawTest(unittest.TestCase):
    """采集中断时最后一行可能只写了一半，判定脚本必须跳过而不是崩。"""

    def test_load_jsonl_skips_truncated_last_line(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            path.write_text(
                '{"task_id": 1, "attempt_index": 0}\n{"task_id": 2, "steps": [',
                encoding="utf-8",
            )
            rows = module.load_jsonl(path)
            self.assertEqual([row["task_id"] for row in rows], [1])


if __name__ == "__main__":
    unittest.main()
