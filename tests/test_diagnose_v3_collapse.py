"""诊断脚本的判定口径测试：理由在教「翻页」还是「换词重搜」。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_v3_collapse import has_paging, hits, REQUERY_WORDS, PAGE_WORDS  # noqa: E402


def _row(task_id: int, trajectory_id: str, turns: list[tuple[str, str]]) -> dict:
    messages = [
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [{"function": {"name": name, "arguments": {}}}],
        }
        for name, content in turns
    ]
    return {"task_id": task_id, "trajectory_id": trajectory_id, "messages": messages}


class KeywordTest(unittest.TestCase):
    def test_page_words_detected(self):
        self.assertTrue(hits("这页只有 30ml，翻下一页看看有没有 60ml。", PAGE_WORDS))

    def test_requery_words_detected(self):
        self.assertTrue(hits("颜色没有黑色，改用“黑色 三轮车”重新搜索。", REQUERY_WORDS))

    def test_plain_purchase_matches_nothing(self):
        self.assertFalse(hits("价格 258 元符合预算，下单。", PAGE_WORDS))
        self.assertFalse(hits("价格 258 元符合预算，下单。", REQUERY_WORDS))

    def test_has_paging_ignores_non_paging_turns(self):
        row = _row(1, "a", [("search_products", "先搜一下"), ("buy_now", "下单")])
        self.assertFalse(has_paging(row))


class ReportTest(unittest.TestCase):
    def _run(self, rows: list[dict]) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.jsonl"
            path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            return subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "diagnose_v3_collapse.py"),
                    "--source",
                    str(path),
                    "--skip-tokenize",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=str(ROOT),
            )

    def test_flags_requery_dominant_reasoning(self):
        rows = [
            _row(
                1,
                "a",
                [
                    ("next_page", "这页没有合适的，改用“60ml 防晒”重新搜索。"),
                    ("search_products", "重新搜索更精确的关键词。"),
                    ("search_products", "价格符合预算，下单。"),
                ],
            )
        ]
        result = self._run(rows)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("换词重搜", result.stdout)

    def test_accepts_paging_dominant_reasoning(self):
        rows = [
            _row(
                1,
                "a",
                [
                    ("next_page", "第一页全是 30ml，缺 60ml 规格，翻下一页。"),
                    ("next_page", "第二页仍没有 60ml，继续翻页。"),
                ],
            )
        ]
        result = self._run(rows)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("方向对，问题不在内容", result.stdout)


if __name__ == "__main__":
    unittest.main()
