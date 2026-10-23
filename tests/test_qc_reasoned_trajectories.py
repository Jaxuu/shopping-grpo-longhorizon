"""验证抽检脚本的两条判定口径。

这个脚本本身不训练，但它决定「要不要重跑 10 分钟补写」或「要不要花 4 小时训练」，
所以判错代价很高。2026-10-24 两次误判都值得固化成测试：

1. 语言：prompt 要中文，实际大面积英文 —— 必须和**原池基准**比才能发现；
2. 「翻页理由是否说清这页不够」只用中文关键词统计，把英文理由全判成不合格
   （59.2% 这个数字就是这么来的），实际是统计口径错，不是数据差。
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qc_reasoned_trajectories.py"

sys.path.insert(0, str(ROOT / "scripts"))
from qc_reasoned_trajectories import (  # noqa: E402
    explains_insufficiency,
    language_of,
)


def _row(trajectory_id, task_id, contents, paging=False):
    calls = [("search_products", '{"query":"x"}')]
    if paging:
        calls.append(("next_page", "{}"))
    calls.append(("buy_now", "{}"))
    messages = [{"role": "user", "content": f"买 {task_id}"}]
    for (name, arguments), content in zip(calls, contents):
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
            }
        )
        messages.append({"role": "tool", "content": f"{name} ok"})
    return {"trajectory_id": trajectory_id, "task_id": task_id, "messages": messages}


class LanguageTest(unittest.TestCase):
    def test_classifies_chinese(self):
        self.assertEqual(language_of("这一页只有 30ml，不符合要求。"), "zh")

    def test_classifies_english(self):
        self.assertEqual(language_of("Page 2 has no relevant compressor either."), "en")

    def test_chinese_with_english_product_name_is_chinese(self):
        self.assertEqual(language_of("这一页都是 Mizutani 370 元，超预算。"), "zh")

    def test_empty(self):
        self.assertEqual(language_of("   "), "empty")


class InsufficiencyTest(unittest.TestCase):
    def test_chinese_markers(self):
        self.assertTrue(explains_insufficiency("这一页都缺 60ml 规格。"))
        self.assertFalse(explains_insufficiency("再看看。"))

    def test_english_markers_are_covered(self):
        """只写中文关键词时这条会失败——那正是 59.2% 的来源。"""
        self.assertTrue(explains_insufficiency("Page 2 has no relevant compressor either."))
        self.assertTrue(explains_insufficiency("All items exceed the 260 budget."))
        self.assertFalse(explains_insufficiency("I need to go back to search first."))


class EndToEndTest(unittest.TestCase):
    def _run(self, baseline, reasoned):
        tmpdir = Path(tempfile.mkdtemp())
        b, r = tmpdir / "all.jsonl", tmpdir / "reasoned.jsonl"
        b.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in baseline), encoding="utf-8")
        r.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in reasoned), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--baseline", str(b), "--reasoned", str(r)],
            capture_output=True,
            text=True,
        )

    def test_flags_language_mismatch_against_baseline(self):
        baseline = [
            _row("p1", 1, ["", "价格 260 元在预算内，下单。"], paging=True),
            _row("t1", 2, ["先搜一下 60ml 的防晒。", "价格 89 元符合预算，下单。"]),
        ]
        reasoned = [
            _row("p1", 1, ["I'll search for a 60ml sunscreen.", "价格 260 元在预算内，下单。"], paging=True),
        ]

        result = self._run(baseline, reasoned)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("语言不一致", result.stdout)
        self.assertIn("--force --language zh", result.stdout)
        # 基准是原池非翻页轨迹，不能把被补写的那条算进基准。
        self.assertIn("原池 · 非翻页轨迹", result.stdout)

    def test_accepts_consistent_language(self):
        baseline = [
            _row("p1", 1, ["", "价格 260 元在预算内，下单。"], paging=True),
            _row("t1", 2, ["先搜一下 60ml 的防晒。", "价格 89 元符合预算，下单。"]),
        ]
        reasoned = [
            _row("p1", 1, ["这一页只有 30ml，缺 60ml 规格，翻下一页。", "价格 260 元在预算内，下单。"], paging=True),
        ]

        result = self._run(baseline, reasoned)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("语言与原池一致", result.stdout)
        self.assertIn("本次新补的回合（检查对象）\n   动作回合 1，有内容 1", result.stdout)

    def test_reports_when_nothing_was_filled(self):
        baseline = [_row("p1", 1, ["这一页只有 30ml。", "下单。"], paging=True)]
        reasoned = [_row("p1", 1, ["这一页只有 30ml。", "下单。"], paging=True)]

        result = self._run(baseline, reasoned)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("没有新补的回合", result.stdout)
        self.assertNotIn("语言不一致", result.stdout)

    def test_counts_only_newly_filled_turns_as_fills(self):
        """原池已有的老师原话不能被算成「本次补写」，否则会虚高。"""
        baseline = [
            _row("p1", 1, ["老师原话：这一页只有 30ml，不符合要求。", "价格 260 元在预算内，下单。"], paging=True),
        ]
        reasoned = [
            _row("p1", 1, ["老师原话：这一页只有 30ml，不符合要求。", "价格 260 元在预算内，下单。"], paging=True),
        ]

        result = self._run(baseline, reasoned)

        self.assertIn("本次新补的回合（检查对象）", result.stdout)
        self.assertIn("有内容 0", result.stdout)


if __name__ == "__main__":
    unittest.main()
