"""验证 GRPO checkpoint 选择脚本的选择规则。

核心判据（见 ``docs/grpo.md``）：验证集只有 49 条，分辨率是 1/49 ≈ 2.04%；
主指标落在「最高分 − 1/val_size」带宽内的 step 视为并列，**并列取步数最大者**。
这里固定这些语义，避免以后有人把它改成「并列取第一个」而悄悄丢掉更充分的训练。
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts.select_grpo_checkpoint import (
    _format_count,
    available_metrics,
    build_rows,
    load_val_metrics,
    main,
    pick_primary,
    pick_secondary,
    select_best,
)

PRIMARY = "val/reward/purchase_success_rate"
SECONDARY = "val/reward/shaped_mean"
TIE_BAND = 1.0 / 49


class SelectGrpoCheckpointTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output = Path(self._tmp.name) / "grpo"
        self.output.mkdir()
        self.diagnostics = self.output / "training_diagnostics.jsonl"

    def write_diagnostics(self, entries):
        """entries: [(step, metrics_dict), ...]"""
        lines = [
            json.dumps(
                {
                    "schema_version": 1,
                    "event": "optimizer_step",
                    "global_step": step,
                    "metrics": metrics,
                }
            )
            for step, metrics in entries
        ]
        self.diagnostics.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def make_checkpoint(self, step):
        actor = self.output / f"global_step_{step}" / "actor"
        actor.mkdir(parents=True, exist_ok=True)
        return actor

    def rows(self):
        return build_rows(load_val_metrics(self.diagnostics), self.output)

    def run_cli(self, argv):
        with mock.patch("sys.argv", ["select_grpo_checkpoint.py", *argv]):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                main()
        return buffer.getvalue()

    # --- 解析 -------------------------------------------------------------

    def test_loads_only_optimizer_step_val_scalars(self):
        self.diagnostics.write_text(
            "\n".join(
                [
                    # 非 optimizer_step 事件必须忽略
                    json.dumps({"event": "generation_batch", "global_step": 10, "metrics": {PRIMARY: 0.9}}),
                    # 非 val/ 指标、非标量、非有限值都必须忽略
                    json.dumps(
                        {
                            "event": "optimizer_step",
                            "global_step": 25,
                            "metrics": {
                                PRIMARY: 0.65,
                                "actor/pg_clipfrac": 0.03,
                                "val/only_list": [1.0, 2.0],
                                "val/not_finite": float("inf"),
                                "val/is_bool": True,
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        per_step = load_val_metrics(self.diagnostics)
        self.assertEqual(per_step, {25: {PRIMARY: 0.65}})

    def test_diagnostics_without_val_metrics_yield_empty_table(self):
        """文件存在但没有 val/ 记录时返回空表（不抛异常），由 main 负责报错。"""
        self.write_diagnostics([(25, {"actor/pg_clipfrac": 0.03})])
        self.assertEqual(load_val_metrics(self.diagnostics), {})

    def test_cli_rejects_run_without_val_metrics(self):
        self.write_diagnostics([(25, {"actor/pg_clipfrac": 0.03})])
        with (
            mock.patch("sys.argv", ["select_grpo_checkpoint.py", str(self.output)]),
            self.assertRaises(SystemExit),
        ):
            main()

    def test_missing_diagnostics_file_is_fatal(self):
        with self.assertRaises(SystemExit):
            load_val_metrics(self.output / "nope.jsonl")

    # --- 主/副指标推断 -----------------------------------------------------

    def test_primary_prefers_success_rate_over_reward(self):
        available = {SECONDARY: [25], PRIMARY: [25]}
        self.assertEqual(pick_primary(available), PRIMARY)

    def test_primary_keyword_fallback(self):
        available = {"val/custom_success_rate": [25]}
        self.assertEqual(pick_primary(available), "val/custom_success_rate")

    def test_primary_returns_none_when_nothing_matches(self):
        self.assertIsNone(pick_primary({"val/penalty/overlong_mean": [25]}))

    def test_secondary_never_equals_primary(self):
        available = {PRIMARY: [25], SECONDARY: [25]}
        self.assertEqual(pick_secondary(available, PRIMARY), SECONDARY)
        self.assertIsNone(pick_secondary({PRIMARY: [25]}, PRIMARY))

    # --- 选择规则 ---------------------------------------------------------

    def test_selects_highest_primary_metric(self):
        self.write_diagnostics(
            [
                (25, {PRIMARY: 0.65, SECONDARY: 0.40}),
                (50, {PRIMARY: 0.69, SECONDARY: 0.45}),
                (75, {PRIMARY: 0.72, SECONDARY: 0.47}),
                (100, {PRIMARY: 0.69, SECONDARY: 0.46}),
            ]
        )
        for step in (25, 50, 75, 100):
            self.make_checkpoint(step)
        chosen, tied, decided_by = select_best(self.rows(), PRIMARY, TIE_BAND)
        self.assertEqual(chosen["step"], 75)
        self.assertEqual([row["step"] for row in tied], [75])
        self.assertEqual(decided_by, "primary")

    def test_tie_band_prefers_later_step_when_no_secondary(self):
        # 0.70 与 0.69 相差 0.01 < 1/49 ≈ 0.0204 → 并列；无副指标 → 取 100
        self.write_diagnostics([(75, {PRIMARY: 0.70}), (100, {PRIMARY: 0.69})])
        for step in (75, 100):
            self.make_checkpoint(step)
        chosen, tied, decided_by = select_best(self.rows(), PRIMARY, TIE_BAND)
        self.assertEqual([row["step"] for row in tied], [75, 100])
        self.assertEqual(chosen["step"], 100)
        self.assertEqual(decided_by, "step")

    def test_secondary_breaks_primary_tie(self):
        # 主指标并列（差 0.01 < 1/49），副指标 75 更高 → 取 75，而不是靠后的 100
        self.write_diagnostics(
            [(75, {PRIMARY: 0.70, SECONDARY: 0.45}), (100, {PRIMARY: 0.69, SECONDARY: 0.40})]
        )
        for step in (75, 100):
            self.make_checkpoint(step)
        chosen, tied, decided_by = select_best(
            self.rows(), PRIMARY, TIE_BAND, secondary=SECONDARY
        )
        self.assertEqual([row["step"] for row in tied], [75, 100])
        self.assertEqual(chosen["step"], 75)
        self.assertEqual(decided_by, "secondary")

    def test_step_breaks_tie_when_secondary_also_tied(self):
        # 主副指标都并列 → 回到「取步数最大者」
        self.write_diagnostics(
            [(75, {PRIMARY: 0.70, SECONDARY: 0.45}), (100, {PRIMARY: 0.69, SECONDARY: 0.45})]
        )
        for step in (75, 100):
            self.make_checkpoint(step)
        chosen, _, decided_by = select_best(
            self.rows(), PRIMARY, TIE_BAND, secondary=SECONDARY
        )
        self.assertEqual(chosen["step"], 100)
        self.assertEqual(decided_by, "step")

    def test_gap_larger_than_tie_band_is_not_a_tie(self):
        # 0.72 与 0.69 相差 0.03 > 1/49 → 不并列，取 75
        self.write_diagnostics([(75, {PRIMARY: 0.72}), (100, {PRIMARY: 0.69})])
        for step in (75, 100):
            self.make_checkpoint(step)
        chosen, tied, decided_by = select_best(self.rows(), PRIMARY, TIE_BAND)
        self.assertEqual([row["step"] for row in tied], [75])
        self.assertEqual(chosen["step"], 75)
        self.assertEqual(decided_by, "primary")

    def test_step_without_checkpoint_is_excluded(self):
        self.write_diagnostics([(75, {PRIMARY: 0.72}), (100, {PRIMARY: 0.60})])
        self.make_checkpoint(100)  # 75 分最高但没有 checkpoint
        rows = self.rows()
        step_75 = next(row for row in rows if row["step"] == 75)
        self.assertFalse(step_75["checkpoint_available"])
        chosen, _, _ = select_best(rows, PRIMARY, TIE_BAND)
        self.assertEqual(chosen["step"], 100)

    def test_no_candidate_when_all_checkpoints_missing(self):
        self.write_diagnostics([(25, {PRIMARY: 0.65})])
        chosen, tied, decided_by = select_best(self.rows(), PRIMARY, TIE_BAND)
        self.assertIsNone(chosen)
        self.assertEqual(tied, [])
        self.assertEqual(decided_by, "none")

    def test_available_metrics_lists_steps_in_order(self):
        self.write_diagnostics([(50, {PRIMARY: 0.5}), (25, {PRIMARY: 0.4})])
        self.assertEqual(available_metrics(load_val_metrics(self.diagnostics)), {PRIMARY: [25, 50]})

    # --- 命中数折算 --------------------------------------------------------

    def test_format_count_turns_rate_into_hits(self):
        self.assertEqual(_format_count(32 / 49, 49), "32/49")
        self.assertEqual(_format_count(1.0, 49), "49/49")
        self.assertEqual(_format_count(0.0, 49), "0/49")

    def test_format_count_skips_non_integral_values(self):
        # 连续型指标（如 mean reward）折算不出整数命中，返回占位符
        self.assertEqual(_format_count(0.4688, 49), "—")

    # --- CLI --------------------------------------------------------------

    def prepare_standard_run(self):
        self.write_diagnostics(
            [
                (25, {PRIMARY: 0.65, SECONDARY: 0.40}),
                (50, {PRIMARY: 0.69, SECONDARY: 0.45}),
                (75, {PRIMARY: 0.72, SECONDARY: 0.47}),
                (100, {PRIMARY: 0.69, SECONDARY: 0.46}),
            ]
        )
        for step in (25, 50, 75, 100):
            self.make_checkpoint(step)

    def test_cli_prints_export_command_for_selected_step(self):
        self.prepare_standard_run()
        text = self.run_cli([str(self.output)])
        self.assertIn("选定 step 75", text)
        self.assertIn("决胜：主指标唯一最高", text)
        self.assertIn("bash scripts/export_grpo.sh", text)
        self.assertIn((self.output / "global_step_75" / "actor").as_posix(), text)

    def test_cli_reports_secondary_decides_a_primary_tie(self):
        self.write_diagnostics(
            [(75, {PRIMARY: 0.70, SECONDARY: 0.45}), (100, {PRIMARY: 0.69, SECONDARY: 0.40})]
        )
        for step in (75, 100):
            self.make_checkpoint(step)
        text = self.run_cli([str(self.output)])
        self.assertIn("由副指标", text)
        self.assertIn("选定 step 75", text)

    def test_cli_json_payload_is_machine_readable(self):
        self.prepare_standard_run()
        payload = json.loads(self.run_cli([str(self.output), "--json"]))
        self.assertEqual(payload["primary_metric"], PRIMARY)
        self.assertEqual(payload["secondary_metric"], SECONDARY)
        self.assertEqual(payload["selected_step"], 75)
        self.assertEqual(payload["tied_steps"], [75])
        self.assertEqual(payload["decided_by"], "primary")
        self.assertEqual(payload["val_size"], 49)
        self.assertEqual(
            payload["selected_checkpoint"],
            (self.output / "global_step_75" / "actor").as_posix(),
        )
        self.assertIn("global_step_75", payload["export_command"])

    def test_cli_list_metrics_does_not_select(self):
        self.prepare_standard_run()
        text = self.run_cli([str(self.output), "--list-metrics"])
        self.assertIn(PRIMARY, text)
        self.assertIn(SECONDARY, text)
        self.assertIn("主指标", text)
        self.assertNotIn("bash scripts/export_grpo.sh", text)

    def test_cli_warns_about_missing_checkpoint(self):
        self.write_diagnostics([(75, {PRIMARY: 0.72}), (100, {PRIMARY: 0.60})])
        self.make_checkpoint(100)
        text = self.run_cli([str(self.output)])
        self.assertIn("trainer.save_freq", text)

    def test_cli_rejects_unknown_primary_metric(self):
        self.prepare_standard_run()
        with (
            mock.patch(
                "sys.argv",
                ["select_grpo_checkpoint.py", str(self.output), "--primary-metric", "val/nope"],
            ),
            self.assertRaises(SystemExit),
        ):
            main()

    def test_cli_respects_explicit_tie_band_zero(self):
        # --tie-band 0 → 严格取最高分，不做并列放宽
        self.write_diagnostics([(75, {PRIMARY: 0.70}), (100, {PRIMARY: 0.69})])
        for step in (75, 100):
            self.make_checkpoint(step)
        text = self.run_cli([str(self.output), "--tie-band", "0"])
        self.assertIn("选定 step 75", text)


if __name__ == "__main__":
    unittest.main()
