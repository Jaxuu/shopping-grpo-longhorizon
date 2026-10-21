"""`inspect_collection_raw.py` 的测试：断点续跑前的盘点口径。

三个容易搞错的点，各钉一个用例：

1. 错误行（402/429）**不能**算已完成——否则续跑会永久跳过它；
2. 末行被截断时要能读出来，并报行号，而不是整个文件炸掉；
3. 「还剩多少条」按**计划次数**算，不用均值（断点时均值偏小）。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import inspect_collection_raw as module


def row(task_id: int, attempt: int, status: str = "done", error: bool = False) -> dict:
    payload = {"task_id": task_id, "attempt_index": attempt, "status": status}
    if error:
        payload["error"] = {"type": "ModelRequestError", "message": "HTTP 402"}
    return payload


class InspectTest(unittest.TestCase):
    def _write(self, tmp: str, rows: list[dict], tail: str = "") -> Path:
        path = Path(tmp) / "raw.jsonl"
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows) + tail
        path.write_text(body, encoding="utf-8")
        return path

    def test_error_rows_are_counted_for_purge_not_as_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                [row(1, 0), row(1, 1, status="error", error=True), row(2, 0)],
            )
            rows, broken = module.read_rows(path)
            report = module.inspect(rows)
            self.assertEqual(broken, [])
            self.assertEqual(report["rows"], 3)
            self.assertEqual(report["purgeable"], 1)
            self.assertEqual(report["terminal"], 2)
            # 题 1 只算完成 1 次（attempt 1 是错误行，不算），题 2 完成 1 次
            self.assertEqual(report["tasks"], 2)
            self.assertAlmostEqual(report["attempts_per_task"], 1.0)

    def test_truncated_last_line_is_reported_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [row(1, 0)], tail='{"task_id": 2, "attempt')
            rows, broken = module.read_rows(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(broken, [2])

    def test_remaining_uses_planned_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 题 1 跑满 3 次，题 2 只跑 1 次，清单里还有题 3 没开始。
            path = self._write(tmp, [row(1, 0), row(1, 1), row(1, 2), row(2, 0)])
            targets = Path(tmp) / "targets.jsonl"
            targets.write_text(
                "".join(json.dumps({"task_id": t}) + "\n" for t in (1, 2, 3)),
                encoding="utf-8",
            )
            rows, _ = module.read_rows(path)
            report = module.inspect(rows, module.task_ids_from(targets))
            self.assertEqual(report["targets"], 3)
            self.assertEqual(report["targets_done"], 2)
            self.assertEqual(report["targets_pending"], 1)
            self.assertEqual(report["planned_attempts"], 3)
            # 题 2 差 2 条 + 题 3 整题 3 条
            self.assertEqual(report["remaining"], 5)

    def test_planned_attempts_must_come_from_caller(self):
        """整批都只跑到第 2 次就断，靠 max_attempt_index 反推会算出「已补齐」。

        2026-10-22：requery 批 386 条覆盖 120 题、每题 3.2 次，观测最大
        attempt_index=3 恰好等于计划次数，侥幸算对；换个断点位置就错。
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [row(1, 0), row(1, 1), row(2, 0), row(2, 1)])
            rows, _ = module.read_rows(path)
            targets = [1, 2]
            guessed = module.inspect(rows, targets)
            self.assertEqual(guessed["planned_attempts"], 2)
            self.assertEqual(guessed["remaining"], 0)
            explicit = module.inspect(rows, targets, attempts=4)
            self.assertEqual(explicit["planned_attempts"], 4)
            self.assertEqual(explicit["remaining"], 4)


if __name__ == "__main__":
    unittest.main()
