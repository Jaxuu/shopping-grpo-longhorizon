"""GRPO 池「翻页靶子」审计的口径测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_grpo_paging_targets import bucket_of, load_task_ids  # noqa: E402


class BucketTest(unittest.TestCase):
    def test_first_page_visible(self):
        self.assertEqual(bucket_of(1), "1-20")
        self.assertEqual(bucket_of(20), "1-20")

    def test_paging_bands(self):
        self.assertEqual(bucket_of(21), "21-40")
        self.assertEqual(bucket_of(60), "41-60")
        self.assertEqual(bucket_of(120), "61-120")

    def test_unreachable(self):
        self.assertEqual(bucket_of(None), "名次>150（翻页也看不到）")
        self.assertEqual(bucket_of(151), "121+")


class LoadIdsTest(unittest.TestCase):
    def test_reads_plain_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.jsonl"
            path.write_text(
                "\n".join(json.dumps({"task_id": i}) for i in (5, 9, 13)) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_task_ids(path), [5, 9, 13])

    def test_reads_extra_info_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.jsonl"
            path.write_text(json.dumps({"extra_info": {"task_id": 7}}) + "\n", encoding="utf-8")
            self.assertEqual(load_task_ids(path), [7])


if __name__ == "__main__":
    unittest.main()
