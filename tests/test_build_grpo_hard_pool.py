"""难题并入池的构建校验：去重、防评测泄漏、system prompt 逐字继承。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARD = ROOT / "data" / "grpo" / "hard_targets.jsonl"
EVAL = ROOT / "data" / "evaluation" / "tasks.jsonl"


def _base_parquet(path: Path, task_ids: list[int], system: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {
            "data_source": "shopsimulator",
            "prompt": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Instruction: 占位 {task_id}"},
            ],
            "ability": "shopping",
            "reward_model": {"style": "rule", "ground_truth": None},
            "extra_info": {"split": "train", "index": index, "task_id": task_id},
        }
        for index, task_id in enumerate(task_ids)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


class BuildTest(unittest.TestCase):
    def _run(self, base: Path, hard: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_grpo_hard_pool.py"),
                "--base",
                str(base),
                "--hard",
                str(hard),
                "--output",
                str(out),
                *extra,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(ROOT),
        )

    def test_adds_hard_rows_and_skips_duplicates(self):
        hard_ids = [int(json.loads(line)["task_id"]) for line in HARD.read_text(encoding="utf-8").splitlines() if line.strip()]
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            base = tmpdir / "base.parquet"
            _base_parquet(base, [hard_ids[0], 999_001], "SYS-逐字")
            out = tmpdir / "out.parquet"
            result = self._run(base, HARD, out, "--n-hard", "3")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("跳过 已在池中：1", result.stdout)

            import pyarrow.parquet as pq

            rows = pq.read_table(out).to_pylist()
            self.assertEqual(len(rows), 4)  # 2 基础 + 3 难题里新增 2
            self.assertEqual({row["prompt"][0]["content"] for row in rows}, {"SYS-逐字"})
            added = [row for row in rows if row["extra_info"]["task_id"] == hard_ids[1]]
            self.assertEqual(len(added), 1)
            self.assertTrue(added[0]["prompt"][1]["content"].startswith("Instruction: "))

    def test_refuses_eval_overlap(self):
        eval_id = int(json.loads(EVAL.read_text(encoding="utf-8").splitlines()[0])["task_id"])
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            base = tmpdir / "base.parquet"
            _base_parquet(base, [999_001], "SYS")
            hard = tmpdir / "hard.jsonl"
            hard.write_text(json.dumps({"task_id": eval_id}) + "\n", encoding="utf-8")
            out = tmpdir / "out.parquet"
            result = self._run(base, hard, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("跳过 与评测集重叠：1", result.stdout)

            import pyarrow.parquet as pq

            self.assertEqual(pq.read_table(out).num_rows, 1)

    def test_n_easy_subsamples_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            base = tmpdir / "base.parquet"
            _base_parquet(base, [999_001, 999_002, 999_003, 999_004], "SYS")
            hard = tmpdir / "hard.jsonl"
            hard.write_text("", encoding="utf-8")
            out = tmpdir / "out.parquet"
            result = self._run(base, hard, out, "--n-easy", "2")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("原池 2 题 + 新增 0 题 = 2 题", result.stdout)


if __name__ == "__main__":
    unittest.main()
