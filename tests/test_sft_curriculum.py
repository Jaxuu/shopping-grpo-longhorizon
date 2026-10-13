import tempfile
import unittest
from pathlib import Path

from scripts.run_sft_curriculum import build_stage_commands


class SftCurriculumTest(unittest.TestCase):
    def test_builds_three_sequential_train_and_merge_stages(self):
        manifest = {
            "stages": {
                "a": {"epochs": 1.0, "learning_rate": 1e-4},
                "b": {"epochs": 1.0, "learning_rate": 7e-5},
                "c": {"epochs": 1.0, "learning_rate": 5e-5},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            commands = build_stage_commands(
                manifest,
                manifest_path=root / "manifest.json",
                source=root / "all.jsonl",
                base_model="Qwen/Qwen3.5-2B",
                output_root=root / "outputs",
                python="python",
            )

        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[0]["stage"], "a")
        self.assertEqual(commands[0]["train"][commands[0]["train"].index("--model") + 1], "Qwen/Qwen3.5-2B")
        self.assertTrue(commands[1]["train"][commands[1]["train"].index("--model") + 1].endswith("stage-a/merged"))
        self.assertTrue(commands[2]["merge"][commands[2]["merge"].index("--output") + 1].endswith("stage-c/merged"))
        self.assertIn("--curriculum-manifest", commands[0]["train"])
        self.assertIn("--gradient-checkpointing", commands[0]["train"])
        self.assertEqual(
            commands[0]["train"][commands[0]["train"].index("--loss-scope") + 1],
            "assistant",
        )

    def test_loss_scope_is_threaded_to_every_stage(self):
        """M5 的两臂通过同一课程、同一 loss 口径参数区分。"""
        manifest = {
            "stages": {
                stage: {"epochs": 1.0, "learning_rate": 1e-4}
                for stage in ("a", "b", "c")
            }
        }
        commands = build_stage_commands(
            manifest,
            manifest_path=Path("manifest.json"),
            source=Path("all.jsonl"),
            base_model="base",
            output_root=Path("outputs"),
            python="python",
            loss_scope="tool-call-only",
        )

        self.assertEqual(len(commands), 3)
        for command in commands:
            self.assertEqual(
                command["train"][command["train"].index("--loss-scope") + 1],
                "tool-call-only",
            )

    def test_start_and_stop_select_a_contiguous_stage_range(self):
        manifest = {
            "stages": {
                stage: {"epochs": 1.0, "learning_rate": 1e-4}
                for stage in ("a", "b", "c")
            }
        }
        commands = build_stage_commands(
            manifest,
            manifest_path=Path("manifest.json"),
            source=Path("all.jsonl"),
            base_model="base",
            output_root=Path("outputs"),
            python="python",
            start_stage="b",
            stop_after_stage="b",
            swanlab=True,
        )

        self.assertEqual([command["stage"] for command in commands], ["b"])
        self.assertIn("--swanlab", commands[0]["train"])
        self.assertIn("stage-b/adapter", " ".join(commands[0]["train"]))

        with self.assertRaisesRegex(ValueError, "after"):
            build_stage_commands(
                manifest,
                manifest_path=Path("manifest.json"),
                source=Path("all.jsonl"),
                base_model="base",
                output_root=Path("outputs"),
                python="python",
                start_stage="c",
                stop_after_stage="a",
            )


    def test_max_steps_is_forwarded_to_every_stage_for_smoke_runs(self):
        """冒烟测试要能在课程入口上限制步数（显存/耗时预检）。"""
        manifest = {
            "stages": {
                stage: {"epochs": 1.0, "learning_rate": 1e-4}
                for stage in ("a", "b", "c")
            }
        }
        commands = build_stage_commands(
            manifest,
            manifest_path=Path("manifest.json"),
            source=Path("all.jsonl"),
            base_model="base",
            output_root=Path("outputs"),
            python="python",
            max_steps=20,
        )

        self.assertEqual(len(commands), 3)
        for command in commands:
            self.assertEqual(
                command["train"][command["train"].index("--max-steps") + 1], "20"
            )

    def test_default_curriculum_run_has_no_max_steps(self):
        manifest = {
            "stages": {
                stage: {"epochs": 1.0, "learning_rate": 1e-4}
                for stage in ("a", "b", "c")
            }
        }
        commands = build_stage_commands(
            manifest,
            manifest_path=Path("manifest.json"),
            source=Path("all.jsonl"),
            base_model="base",
            output_root=Path("outputs"),
            python="python",
        )

        for command in commands:
            self.assertNotIn("--max-steps", command["train"])


if __name__ == "__main__":
    unittest.main()
