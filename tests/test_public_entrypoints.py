"""Public CPU and parameterized GRPO entry-point tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.train_grpo import build_command, build_overrides, main as train_grpo_main, parse_args
from shopping_grpo.cli import main as cli_main
from shopping_grpo.smoke import run_cpu_smoke


class PublicEntrypointTest(unittest.TestCase):
    def test_cpu_smoke_covers_public_contracts(self):
        result = run_cpu_smoke()

        self.assertEqual(
            result["checks"],
            [
                "action_schema",
                "trajectory_normalization",
                "reward_sample",
                "sft_label_mask",
                "dynamic_sampling_grouping",
            ],
        )

    def test_offline_example_cli_runs_without_models_or_environment(self):
        root = Path(__file__).resolve().parents[1]
        with patch.object(
            sys,
            "argv",
            [
                "shopping-grpo",
                "evaluate",
                str(root / "examples/trajectories.jsonl"),
            ],
        ), patch("builtins.print") as output:
            cli_main()

        summary = json.loads(output.call_args.args[0])
        self.assertEqual(summary["trajectory_count"], 3)
        self.assertEqual(summary["strict_gold_success_count"], 1)

    def test_public_grpo_launcher_accepts_sharded_weights_and_console(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                "{}",
                encoding="utf-8",
            )
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            output = temporary / "output"
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(output),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--logger",
                    "console",
                    "--dry-run",
                ],
            ):
                args = parse_args()
            command, environment = build_command(args)

        self.assertIn("verl.trainer.main_ppo", command)
        self.assertEqual(environment["GRPO_MODEL_PATH"], str(model.resolve()))
        self.assertEqual(environment["GRPO_TRAIN_FILE"], str(train.resolve()))
        self.assertEqual(environment["GRPO_VAL_FILE"], str(validation.resolve()))
        self.assertEqual(
            environment["SHOPPING_GRPO_DIAGNOSTICS_PATH"],
            str(output.resolve() / "training_diagnostics.jsonl"),
        )
        self.assertIn("trainer.logger=[console]", command)

    @staticmethod
    def _launcher_argv(temporary: Path, extra=(), resume=False):
        model = temporary / "model"
        model.mkdir(exist_ok=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        (model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        train = temporary / "train.parquet"
        train.write_bytes(b"example")
        validation = temporary / "validation.parquet"
        validation.write_bytes(b"example")
        root = Path(__file__).resolve().parents[1]
        argv = ["train_grpo.py"]
        if resume:
            argv.append("--resume")
        argv += [
            "--model",
            str(model),
            "--train-data",
            str(train),
            "--val-data",
            str(validation),
            "--output",
            str(temporary / "output"),
            "--config",
            str(root / "configs/grpo.yaml"),
            "--logger",
            "console",
            *extra,
        ]
        return argv

    def test_non_empty_output_is_rejected_without_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            output = temporary / "output"
            output.mkdir()
            (output / "training_diagnostics.jsonl").write_text("", encoding="utf-8")
            with patch.object(sys, "argv", self._launcher_argv(temporary)):
                args = parse_args()
            with self.assertRaises(SystemExit) as raised:
                build_command(args)
        self.assertIn("new or empty", str(raised.exception))

    def test_resume_requires_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            output = temporary / "output"
            output.mkdir()
            (output / "training_diagnostics.jsonl").write_text("", encoding="utf-8")
            with patch.object(
                sys, "argv", self._launcher_argv(temporary, resume=True)
            ):
                args = parse_args()
            with self.assertRaises(SystemExit) as raised:
                build_command(args)
        self.assertIn("global_step_", str(raised.exception))

    def test_resume_accepts_output_holding_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            output = temporary / "output"
            (output / "global_step_25" / "actor").mkdir(parents=True)
            with patch.object(
                sys, "argv", self._launcher_argv(temporary, resume=True)
            ):
                args = parse_args()
            command, environment = build_command(args)

        self.assertEqual(environment["GRPO_OUTPUT_DIR"], str(output.resolve()))
        self.assertIn("verl.trainer.main_ppo", command)

    def test_overrides_are_reused_by_both_command_and_preflight(self):
        """回归：main() 的预检曾引用 build_command 的局部变量 overrides/extra（NameError）。

        `build_overrides` 必须是命令尾部 override 的唯一来源，否则两处会漂移。
        """
        with tempfile.TemporaryDirectory() as directory:
            argv = self._launcher_argv(
                Path(directory), extra=["--", "data.train_batch_size=4"]
            )
            with patch.object(sys, "argv", argv):
                args = parse_args()
            command, _ = build_command(args)
            overrides = build_overrides(args)

        self.assertIn("data.train_batch_size=4", overrides)
        self.assertEqual(command[-len(overrides) :], overrides)

    def test_main_runs_preflight_before_training_command(self):
        """回归：main() 必须先跑 check_grpo_runtime.py 预检，再跑 veRL 训练命令。"""
        invocations = []

        def fake_call(command, cwd=None, env=None):
            invocations.append(command)
            return 0

        with tempfile.TemporaryDirectory() as directory:
            argv = self._launcher_argv(
                Path(directory), extra=["--", "trainer.total_training_steps=4"]
            )
            with patch.object(sys, "argv", argv), patch.object(
                subprocess, "call", fake_call
            ):
                with self.assertRaises(SystemExit) as raised:
                    train_grpo_main()

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(len(invocations), 2, "应当先预检、再训练")
        self.assertIn("check_grpo_runtime.py", " ".join(invocations[0]))
        self.assertIn("verl.trainer.main_ppo", " ".join(invocations[1]))
        self.assertIn("trainer.total_training_steps=4", invocations[0])
        self.assertIn("trainer.total_training_steps=4", invocations[1])


if __name__ == "__main__":
    unittest.main()
