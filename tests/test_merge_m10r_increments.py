"""`scripts/merge_m10r_increments.py`：增量合并必须同题只留一条。

为什么钉死它（2026-10-21）
--------------------------
首次跑 `run_m10r_build.sh` 直接死在这里：`build_m10r_pool.py` 有硬契约
「增量里 task_id 不能重复」，而各批清单同源必然重叠
（`m10_paging_targets` ⊂ `m10_rank2_targets`、探针翻页题 ⊂ `m10_paging_targets`）。

四条不变量：

1. 同一 task_id 只出现一次，**先到的来源赢**（命令行顺序 = 优先级）。
2. 被丢掉的题要能查到「被谁占掉」——否则分不清「这批本来就没多少」和「被前一批吃了」。
3. 不存在的来源跳过并标注，不算错误；但全部不存在要返回非零。
4. 写出的 jsonl 行数与报告一致，且保持来源顺序（可复现）。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import merge_m10r_increments as module  # noqa: E402


def write_rows(path: Path, task_ids) -> None:
    path.write_text(
        "".join(json.dumps({"task_id": task_id}) + "\n" for task_id in task_ids),
        encoding="utf-8",
    )


class MergeTest(unittest.TestCase):
    def test_first_source_wins_on_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.jsonl"
            second = Path(tmp) / "b.jsonl"
            write_rows(first, [1, 2])
            write_rows(second, [2, 3])
            merged, report = module.merge([first, second])
            self.assertEqual([row["task_id"] for row in merged], [1, 2, 3])
            self.assertEqual(report[0]["kept"], 2)
            self.assertEqual(report[1]["kept"], 1)
            self.assertEqual(report[1]["duplicate"], 1)
            # 被谁占掉：来自 a.jsonl
            self.assertEqual(report[1]["duplicate_with"], {"a.jsonl": 1})

    def test_order_matters(self):
        """交换来源顺序，留下的那条就换人——这正是「优先级」的实现方式。"""
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.jsonl"
            second = Path(tmp) / "b.jsonl"
            write_rows(first, [1])
            write_rows(second, [1])
            merged_a, _ = module.merge([first, second])
            merged_b, _ = module.merge([second, first])
            self.assertEqual(len(merged_a), 1)
            self.assertEqual(len(merged_b), 1)

    def test_missing_source_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "a.jsonl"
            write_rows(existing, [1])
            missing = Path(tmp) / "nope.jsonl"
            merged, report = module.merge([existing, missing])
            self.assertEqual(len(merged), 1)
            self.assertEqual(report[1]["status"], "missing")

    def test_all_missing_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.jsonl"
            code = module.main(
                [str(missing), "--out", str(Path(tmp) / "out.jsonl")]
            )
            self.assertEqual(code, 1)

    def test_main_writes_jsonl_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.jsonl"
            second = Path(tmp) / "b.jsonl"
            write_rows(first, [1, 2])
            write_rows(second, [2])
            out = Path(tmp) / "merged.jsonl"
            code = module.main([str(first), str(second), "--out", str(out)])
            self.assertEqual(code, 0)
            lines = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([row["task_id"] for row in lines], [1, 2])
            report = json.loads(
                (Path(tmp) / "increment_merge_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["total_rows"], 2)
            self.assertEqual(report["sources"][1]["duplicate"], 1)

    def test_duplicate_rows_within_one_source(self):
        """同一批里自带重复也要去重（源文件写重了不该炸在建池脚本里）。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            write_rows(path, [1, 1, 2])
            merged, report = module.merge([path])
            self.assertEqual([row["task_id"] for row in merged], [1, 2])
            self.assertEqual(report[0]["duplicate"], 1)


if __name__ == "__main__":
    unittest.main()
