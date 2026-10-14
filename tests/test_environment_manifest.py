import unittest
import hashlib
import json
from pathlib import Path
import tempfile

from scripts.check_grpo_runtime import CURRENT_RUNTIME_FILES
from shopping_grpo.environment.manifest import (
    MANIFEST_VERSION,
    shopsimulator_source_commit,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def lf_sha256(path):
    """按 LF 归一化后计算 SHA-256，等价于仓库 blob 的字节内容。

    `shopping_grpo.environment.manifest.sha256_file` 按原始字节哈希、不做换行归一化，
    所以 Windows（core.autocrlf=true）检出的 CRLF 会算出与 Linux 不同的值。
    本测试显式归一化，使两端结论一致。
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class EnvironmentManifestTest(unittest.TestCase):
    def test_current_environment_contract_is_validated(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "environment_version": "shopsimulator-environment-v2.1",
            "shopsimulator_commit": "a" * 40,
            "product_data_sha256": "c" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v3"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v2",
            "max_steps": 35,
            "seed": 20260726,
        }
        self.assertIs(validate_manifest(manifest), manifest)

    def test_page_size_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_manifest({})

    def test_current_environment_requires_reward_v3(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "environment_version": "shopsimulator-environment-v2.1",
            "shopsimulator_commit": "a" * 40,
            "product_data_sha256": "c" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v3"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v2",
            "max_steps": 35,
            "seed": 20260726,
        }
        self.assertIs(validate_manifest(manifest), manifest)
        manifest["reward"] = {"version": "unsupported-reward"}
        with self.assertRaisesRegex(
            ValueError, "requires reward version one of"
        ):
            validate_manifest(manifest)

    def test_manifest_accepts_reward_v3_and_v4(self):
        base = {
            "manifest_version": MANIFEST_VERSION,
            "environment_version": "shopsimulator-environment-v2.1",
            "shopsimulator_commit": "a" * 40,
            "product_data_sha256": "c" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v2",
            "max_steps": 35,
            "seed": 20260726,
        }
        for version in ("shopsimulator-reward-v3", "shopsimulator-reward-v4"):
            manifest = dict(base, reward={"version": version})
            self.assertIs(validate_manifest(manifest), manifest)

    def test_wrong_tool_contract_is_rejected(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "shopsimulator_commit": "a" * 40,
            "product_data_sha256": "c" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v3"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "unsupported-tools",
            "max_steps": 35,
            "seed": 20260726,
        }
        with self.assertRaisesRegex(ValueError, "Tool v2"):
            validate_manifest(manifest)

    def test_embedded_shopsimulator_commit_is_read_without_nested_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "EMBEDDED_SOURCE.json").write_text(
                json.dumps({"source_commit": "e" * 40}),
                encoding="utf-8",
            )
            self.assertEqual(shopsimulator_source_commit(root), "e" * 40)

    def test_runtime_file_hashes_match_repository_content(self):
        """回归：清单哈希曾在 Windows CRLF 检出上计算，导致 Linux 上 GRPO 预检必然失败。

        `scripts/check_grpo_runtime.py::validate_reward_runtime_files` 会逐文件比对，
        任何一项过期都会让 `bash scripts/grpo.sh` 在预检阶段直接退出。
        """
        manifest = json.loads((ROOT / "data/environment.json").read_text(encoding="utf-8"))
        expected = manifest["runtime_files_sha256"]

        self.assertEqual(
            set(expected),
            set(CURRENT_RUNTIME_FILES),
            "清单的文件集合与运行时契约不一致",
        )

        mismatches = {}
        for name, relative_path in CURRENT_RUNTIME_FILES.items():
            actual = lf_sha256(ROOT / relative_path)
            if actual != expected[name]:
                mismatches[name] = {"expected": expected[name], "actual": actual}

        self.assertEqual(
            mismatches,
            {},
            "data/environment.json 的 runtime_files_sha256 与仓库内容不一致，"
            f"GRPO 预检会因此失败：{mismatches}",
        )

    def test_runtime_files_exist(self):
        for name, relative_path in CURRENT_RUNTIME_FILES.items():
            with self.subTest(file=name):
                self.assertTrue(
                    (ROOT / relative_path).is_file(),
                    f"清单引用的运行时文件不存在：{relative_path}",
                )


if __name__ == "__main__":
    unittest.main()
