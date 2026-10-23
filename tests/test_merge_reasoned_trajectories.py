"""验证合并脚本：替换、打标、以及三种必须拒绝的输入。

这个脚本的作用是把「补好理由的 144 条」拼回「完整 1,210 条」，并给这些轨迹打上
``loss_scope=assistant``。它一旦静默出错（少拼了行、忘了打标），训练照跑、分数
照出，但补的思考根本没进 loss——所以每条拒绝分支都要有测试。
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "merge_reasoned_trajectories.py"


def _row(trajectory_id, task_id, paging, reason=""):
    calls = [("search_products", '{"query":"pillow"}')]
    if paging:
        calls.append(("next_page", "{}"))
    calls.append(("buy_now", "{}"))
    messages = [{"role": "user", "content": f"买 {task_id}"}]
    for name, arguments in calls:
        messages.append(
            {
                "role": "assistant",
                "content": reason,
                "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
            }
        )
        messages.append({"role": "tool", "content": f"{name} ok"})
    return {
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "messages": messages,
        "tools": [{"type": "function", "function": {"name": "buy_now"}}],
    }


class MergeReasonedTrajectoriesTest(unittest.TestCase):
    def _run(self, source_rows, reasoned_rows, extra_args=()):
        tmpdir = tempfile.mkdtemp()
        base = Path(tmpdir)
        source = base / "all.jsonl"
        reasoned = base / "reasoned.jsonl"
        output = base / "out" / "all_v3.jsonl"
        source.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in source_rows),
            encoding="utf-8",
        )
        reasoned.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reasoned_rows),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--source",
                str(source),
                "--reasoned",
                str(reasoned),
                "--output",
                str(output),
                *extra_args,
            ],
            capture_output=True,
            text=True,
        )
        rows = (
            [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            if output.is_file()
            else []
        )
        return result, rows

    def test_replaces_paging_rows_and_keeps_the_rest(self):
        source = [
            _row("t1", 1, paging=True),
            _row("t2", 2, paging=False),
            _row("t3", 3, paging=True),
        ]
        reasoned = [
            _row("t1", 1, paging=True, reason="这一页都没有记忆棉，翻下一页看看。"),
            _row("t3", 3, paging=True, reason="价格都超预算，再翻一页。"),
        ]

        result, rows = self._run(source, reasoned)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r["trajectory_id"] for r in rows], ["t1", "t2", "t3"])
        self.assertEqual(rows[0]["messages"][1]["content"], "这一页都没有记忆棉，翻下一页看看。")
        # 没被补写的模板轨迹必须原样保留（否则池子会被削成 144 条）。
        self.assertEqual(rows[1], _row("t2", 2, paging=False))
        self.assertIn("补写前 0/6", result.stdout)
        self.assertIn("补写后 6/6", result.stdout)

    def test_tags_every_assistant_turn_of_replaced_rows_only(self):
        source = [_row("t1", 1, paging=True), _row("t2", 2, paging=False)]
        reasoned = [_row("t1", 1, paging=True, reason="缺规格，翻页。")]

        result, rows = self._run(source, reasoned, ("--tag-loss-scope", "assistant"))

        self.assertEqual(result.returncode, 0, result.stderr)
        tagged = [
            m["loss_scope"]
            for m in rows[0]["messages"]
            if m["role"] == "assistant"
        ]
        # 每一步都要标：只标翻页那一步，模型在决策前就没有思考可依。
        self.assertEqual(tagged, ["assistant"] * 3)
        untagged = [
            m.get("loss_scope") for m in rows[1]["messages"] if m["role"] == "assistant"
        ]
        self.assertEqual(untagged, [None] * 2)

    def test_rejects_rows_missing_from_source(self):
        source = [_row("t1", 1, paging=True)]
        reasoned = [_row("t9", 9, paging=True, reason="x")]

        result, rows = self._run(source, reasoned)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("不在源池里", result.stderr)
        self.assertEqual(rows, [])

    def test_rejects_rows_without_next_page(self):
        """--paging-only 忘了加时会把整池送进补写；这里挡住，避免误覆盖模板轨迹。"""
        source = [_row("t1", 1, paging=True), _row("t2", 2, paging=False)]
        reasoned = [_row("t2", 2, paging=False, reason="这条不该被补写。")]

        result, rows = self._run(source, reasoned)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("不含 next_page", result.stderr)
        self.assertEqual(rows, [])

    def test_rejects_duplicate_trajectory_ids(self):
        """断点续跑写重了会在这里暴露，而不是悄悄取最后一条。"""
        source = [_row("t1", 1, paging=True)]
        reasoned = [
            _row("t1", 1, paging=True, reason="第一次"),
            _row("t1", 1, paging=True, reason="第二次"),
        ]

        result, rows = self._run(source, reasoned)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("出现了两次", result.stderr)
        self.assertEqual(rows, [])

    def test_fails_when_a_paging_turn_is_still_empty(self):
        source = [_row("t1", 1, paging=True)]
        reasoned = [_row("t1", 1, paging=True, reason="")]

        result, rows = self._run(source, reasoned)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("没有 content", result.stderr)

    def test_allow_partial_lets_a_known_gap_through(self):
        source = [_row("t1", 1, paging=True)]
        reasoned = [_row("t1", 1, paging=True, reason="")]

        result, rows = self._run(source, reasoned, ("--allow-partial",))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
