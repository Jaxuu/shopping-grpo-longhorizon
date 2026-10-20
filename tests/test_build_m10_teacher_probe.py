"""`scripts/build_m10_teacher_probe.py` 的选题口径固定。

钉死四件事：

1. **翻页场景只收名次 ≤60**：环境的循环检测器把翻页卡在 2 次，第 61 名以后
   物理上到不了；拿它们探「老师会不会翻页」会把「翻不到」记成「不翻页」。
2. **同一 task_id 只出现一次**：采集端按 `(task_id, attempt_index)` 去重，
   清单里重复会让「题数 × 次数」算错，进而让 `--target-accepted` 早停。
3. **抽样是确定性的**：同 seed 同结果，且先按 task_id 排序再抽，
   不依赖清单文件的行顺序。
4. **`--dry-run` 不写文件**。
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

import build_m10_teacher_probe as module  # noqa: E402


def write_rows(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


class EligibleTest(unittest.TestCase):
    def test_paging_drops_ranks_above_60(self):
        rows = [{"task_id": 1, "rank": 21}, {"task_id": 2, "rank": 60}, {"task_id": 3, "rank": 61}]
        kept = module.eligible("paging", rows)
        self.assertEqual([row["task_id"] for row in kept], [1, 2])

    def test_paging_drops_rows_without_rank(self):
        rows = [{"task_id": 1}, {"task_id": 2, "rank": None}]
        self.assertEqual(module.eligible("paging", rows), [])

    def test_other_scenarios_keep_everything(self):
        rows = [{"task_id": 1}, {"task_id": 2}]
        self.assertEqual(len(module.eligible("requery", rows)), 2)
        self.assertEqual(len(module.eligible("compare", rows)), 2)


class SampleTest(unittest.TestCase):
    def test_deterministic_for_same_seed(self):
        rows = [{"task_id": index} for index in range(50)]
        first = module.sample(rows, 10, seed=3)
        second = module.sample(list(reversed(rows)), 10, seed=3)
        self.assertEqual(first, second, "行顺序不能影响抽样结果")

    def test_different_seed_can_differ(self):
        rows = [{"task_id": index} for index in range(50)]
        self.assertNotEqual(
            module.sample(rows, 10, seed=1), module.sample(rows, 10, seed=2)
        )

    def test_count_larger_than_pool_returns_all(self):
        rows = [{"task_id": index} for index in range(3)]
        self.assertEqual(len(module.sample(rows, 10, seed=1)), 3)


class BuildTest(unittest.TestCase):
    def test_dedupes_task_ids_across_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            targets = {
                "paging": write_rows(
                    tmp_path / "paging.jsonl", [{"task_id": 1, "rank": 25}]
                ),
                "requery": write_rows(
                    tmp_path / "requery.jsonl", [{"task_id": 1}, {"task_id": 2}]
                ),
            }
            chosen, report = module.build(per_scenario=5, seed=1, targets=targets)
            task_ids = [row["task_id"] for row in chosen]
            self.assertEqual(len(task_ids), len(set(task_ids)))
            self.assertEqual(task_ids.count(1), 1)
            self.assertEqual(report["paging"]["selected"], 1)
            self.assertEqual(report["requery"]["selected"], 1)

    def test_each_row_carries_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            targets = {
                "paging": write_rows(tmp_path / "p.jsonl", [{"task_id": 4, "rank": 30}]),
            }
            chosen, _ = module.build(per_scenario=1, seed=1, targets=targets)
            self.assertEqual(chosen[0]["scenario"], "paging")

    def test_missing_file_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            targets = {
                "paging": tmp_path / "nope.jsonl",
                "compare": write_rows(tmp_path / "c.jsonl", [{"task_id": 8}]),
            }
            chosen, report = module.build(per_scenario=2, seed=1, targets=targets)
            self.assertEqual(report["paging"]["selected"], 0)
            self.assertEqual(len(chosen), 1)

    def test_report_has_all_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            targets = {
                scenario: write_rows(tmp_path / f"{scenario}.jsonl", [{"task_id": 1, "rank": 30}])
                for scenario in ("paging", "requery", "compare")
            }
            _, report = module.build(per_scenario=1, seed=1, targets=targets)
            text = module.render_report(report)
            for scenario in ("paging", "requery", "compare"):
                self.assertIn(scenario, text)


class MainTest(unittest.TestCase):
    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            code = module.main(
                ["--per-scenario", "1", "--output", str(output), "--dry-run"]
            )
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())

    def test_writes_requested_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            code = module.main(
                ["--per-scenario", "3", "--output", str(output), "--seed", "5"]
            )
            self.assertEqual(code, 0)
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 9)
            self.assertEqual(len({row["task_id"] for row in rows}), 9)

    def test_rejects_zero(self):
        self.assertEqual(module.main(["--per-scenario", "0", "--dry-run"]), 2)


if __name__ == "__main__":
    unittest.main()
