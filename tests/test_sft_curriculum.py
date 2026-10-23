import tempfile
import unittest
from pathlib import Path

from scripts.run_sft_curriculum import build_stage_commands


def _posix(path) -> str:
    """把 Windows 的 ``outputs\\stage-a\\merged`` 归一成 ``outputs/stage-a/merged``。

    这两个断言本来想查的是「阶段目录挂在哪」，不是「用哪个分隔符」；
    不归一的话在 Windows 上恒失败（长期假阴性，会掩盖真正的路径回归）。
    """
    return str(path).replace("\\", "/")


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
        self.assertTrue(
            _posix(commands[1]["train"][commands[1]["train"].index("--model") + 1]).endswith(
                "stage-a/merged"
            )
        )
        self.assertTrue(
            _posix(commands[2]["merge"][commands[2]["merge"].index("--output") + 1]).endswith(
                "stage-c/merged"
            )
        )
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
        self.assertIn("stage-b/adapter", _posix(" ".join(commands[0]["train"])))

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


    def test_save_steps_is_threaded_and_defaults_to_epoch_only(self):
        """定时关机靠 checkpoint 粒度续跑：0=只在 epoch 末存（历史行为）。"""
        manifest = {
            "stages": {
                stage: {"epochs": 1.0, "learning_rate": 1e-4}
                for stage in ("a", "b")
            }
        }
        common = dict(
            manifest_path=Path("manifest.json"),
            source=Path("all.jsonl"),
            base_model="base",
            output_root=Path("outputs"),
            python="python",
        )
        default = build_stage_commands(manifest, **common)
        for command in default:
            self.assertNotIn("--save-steps", command["train"])

        periodic = build_stage_commands(manifest, save_steps=25, **common)
        for command in periodic:
            self.assertEqual(
                command["train"][command["train"].index("--save-steps") + 1], "25"
            )


class SingleStageCurriculumTest(unittest.TestCase):
    """单阶段（M10-R v2）：一个 ``full`` 阶段，从 62% 的基线热启动。

    这里守的是「热启动真的生效」——底座传错（退回 Qwen3.5-2B）不会报错，
    只会训出一个远不如 62% 的模型，而且 4 小时之后才发现。
    """

    MANIFEST = {"stages": {"full": {"epochs": 1.0, "learning_rate": 5e-5}}}

    def _commands(self, **kwargs):
        return build_stage_commands(
            self.MANIFEST,
            manifest_path=Path("manifest.json"),
            source=Path("all.jsonl"),
            base_model="/models/sft-b-curriculum/stage-c/merged",
            output_root=Path("outputs"),
            python="python",
            **kwargs,
        )

    def test_single_full_stage_trains_from_the_warm_start(self):
        commands = self._commands()

        self.assertEqual([command["stage"] for command in commands], ["full"])
        train = commands[0]["train"]
        self.assertEqual(
            train[train.index("--model") + 1],
            "/models/sft-b-curriculum/stage-c/merged",
        )
        self.assertEqual(
            train[train.index("--curriculum-stage") + 1],
            "full",
        )

    def test_merge_uses_the_warm_start_not_the_original_base(self):
        """合并底座必须与训练底座一致，否则 merged 里丢掉 LoRA 之外的全部改动。"""
        commands = self._commands()

        merge = commands[0]["merge"]
        self.assertEqual(
            merge[merge.index("--base-model") + 1],
            "/models/sft-b-curriculum/stage-c/merged",
        )
        self.assertTrue(
            _posix(merge[merge.index("--output") + 1]).endswith("stage-full/merged")
        )

    def test_stage_specific_hyperparameters_come_from_the_manifest(self):
        commands = self._commands()

        train = commands[0]["train"]
        self.assertEqual(train[train.index("--learning-rate") + 1], "5e-05")
        self.assertEqual(train[train.index("--epochs") + 1], "1.0")


if __name__ == "__main__":
    unittest.main()
