"""`downsample_sft_pool.py` 的测试：砍模板不能伤到探索轨迹。

三个不变量：

1. **探索轨迹一条不丢**（它们是这次干预的全部内容）；
2. **模板砍到指定条数**，且固定 seed 下可复现（同样的输入永远同样的输出）；
3. **难度标签跟着筛**，不能出现「有行没标签」。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import downsample_sft_pool as module


def _call(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def _row(task_id: int, *, queries: tuple[str, ...] = ("item",), asin: str | None = None, page: bool = False):
    """一条最小轨迹；两个不同 ``queries`` ⇒ 换过词，``page`` ⇒ 翻过页。"""
    asin = asin or f"asin-{task_id}"
    messages = [
        {"role": "system", "content": "use tools"},
        {"role": "user", "content": f"buy {task_id}"},
    ]
    for index, query in enumerate(queries):
        messages += [
            _call("search_products", {"query": query}, f"{task_id}-s{index}"),
            {"role": "tool", "tool_call_id": f"{task_id}-s{index}", "content": asin},
        ]
    if page:
        messages += [
            _call("next_page", {}, f"{task_id}-p"),
            {"role": "tool", "tool_call_id": f"{task_id}-p", "content": "page 2"},
        ]
    messages += [
        _call("open_product", {"asin": asin}, f"{task_id}-2"),
        {"role": "tool", "tool_call_id": f"{task_id}-2", "content": f"asin: {asin}"},
        _call("buy_now", {}, f"{task_id}-3"),
        {"role": "tool", "tool_call_id": f"{task_id}-3", "content": "购买已完成。"},
    ]
    return {"trajectory_id": f"t-{task_id}", "task_id": task_id, "messages": messages}


class DownsampleTest(unittest.TestCase):
    def test_keeps_every_exploratory_row(self):
        rows = [_row(1), _row(2), _row(3, page=True), _row(4, queries=("item", "另一个词"))]
        kept, report = module.downsample(rows, keep_template=1, seed=7)
        kept_ids = {row["task_id"] for row in kept}
        # 3（翻页）和 4（换词）必须留着
        self.assertIn(3, kept_ids)
        self.assertIn(4, kept_ids)
        self.assertEqual(report["buckets_before"]["template"], 2)
        self.assertEqual(report["kept_template"], 1)
        self.assertEqual(len(report["dropped_task_ids"]), 1)

    def test_same_seed_same_result(self):
        rows = [_row(task_id) for task_id in range(1, 21)]
        first, _ = module.downsample(rows, keep_template=5, seed=20260814)
        second, _ = module.downsample(rows, keep_template=5, seed=20260814)
        self.assertEqual(
            [row["task_id"] for row in first], [row["task_id"] for row in second]
        )
        other, _ = module.downsample(rows, keep_template=5, seed=99)
        self.assertNotEqual(
            [row["task_id"] for row in first], [row["task_id"] for row in other]
        )

    def test_keep_none_drops_nothing(self):
        rows = [_row(task_id) for task_id in range(1, 6)]
        kept, report = module.downsample(rows, keep_template=None, seed=1)
        self.assertEqual(len(kept), 5)
        self.assertEqual(report["dropped_template"], 0)

    def test_cli_filters_labels_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [_row(task_id) for task_id in range(1, 11)]
            (root / "all.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            labels = [
                {"task_id": task_id, "difficulty": "simple"} for task_id in range(1, 11)
            ]
            (root / "labels.jsonl").write_text(
                "".join(json.dumps(label) + "\n" for label in labels), encoding="utf-8"
            )
            module.main(
                [
                    "--pool",
                    str(root / "all.jsonl"),
                    "--labels",
                    str(root / "labels.jsonl"),
                    "--keep-template",
                    "4",
                    "--out-dir",
                    str(root / "out"),
                    "--seed",
                    "5",
                ]
            )
            kept_rows = module.read_jsonl(root / "out" / "all.jsonl")
            kept_labels = module.read_jsonl(root / "out" / "difficulty_labels.jsonl")
            self.assertEqual(len(kept_rows), 4)
            self.assertEqual(len(kept_labels), 4)
            self.assertEqual(
                {row["task_id"] for row in kept_rows},
                {label["task_id"] for label in kept_labels},
            )


if __name__ == "__main__":
    unittest.main()
