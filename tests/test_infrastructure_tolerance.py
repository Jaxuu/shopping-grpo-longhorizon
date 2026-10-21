"""采集遇到单条基础设施故障时**不能**停掉整批，连续多条才停。

为什么钉死它（2026-10-21）
--------------------------
M10-R 探针全量跑 480 条，跑到第 339 条时被**一条**异常中止（``has_error`` 1 条），
剩下 141 条白等一小时。原来的判定是 ``|=``，一条命中就置真——那是给「管道彻底坏了」
准备的（4xx 会每一条都同样失败），但 8 并发打一个端点时偶发 429/断连很正常。

四条不变量：

1. 连续故障 < ``INFRASTRUCTURE_FAILURE_TOLERANCE``（3）⇒ 继续跑，不抛。
2. 连续 3 条 ⇒ 抛 ``CollectionInfrastructureError``，且**消息里带原因**
   （原来只有一句「collection infrastructure failure」，还得回去翻 raw.jsonl）。
3. 中途成功一条 ⇒ 连续计数清零。
4. ``collect_tasks``（串行路径）与 ``collect_until_target``（并发路径）行为一致。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import collect_sft_data as collector  # noqa: E402
from shopping_grpo.evaluation import rollout  # noqa: E402
from shopping_grpo.evaluation.rollout import (  # noqa: E402
    INFRASTRUCTURE_FAILURE_TOLERANCE,
    CollectionInfrastructureError,
    _infrastructure_detail,
)


def infra(task_id, *, message="HTTP 429: rate limit exceeded"):
    return {
        "task_id": task_id,
        "attempt_index": 0,
        "status": "error",
        "steps": [],
        "error": {"type": "ModelRequestError", "message": message},
    }


def ordinary(task_id):
    """普通任务失败：老师没买对，属数据问题，不该中断采集。"""
    return {"task_id": task_id, "attempt_index": 0, "status": "max_steps", "steps": []}


def make_tasks(count):
    return [{"task_id": index, "instruction": f"task {index}"} for index in range(1, count + 1)]


class CollectUntilTargetToleranceTest(unittest.TestCase):
    def run_collect(self, script):
        """script: {task_id: 返回的轨迹}；未列出的题返回普通失败。"""

        def fake_collect_for_task(task, **kwargs):
            task_id = int(task["task_id"])
            factory = script.get(task_id)
            return factory(task_id) if factory else ordinary(task_id)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "raw.jsonl"
            with mock.patch.object(collector, "collect_for_task", fake_collect_for_task):
                return collector.collect_until_target(
                    tasks=make_tasks(len(script) + 3),
                    target_accepted=999,
                    client=object(),
                    output_path=output,
                    base_url="http://127.0.0.1:5700",
                    max_steps=35,
                    attempts_per_task=1,
                    workers=1,
                )

    def test_two_consecutive_failures_do_not_abort(self):
        written, _ = self.run_collect({1: infra, 2: infra})
        self.assertEqual(len(written), 5, "两条故障不该停掉整批")

    def test_three_consecutive_failures_abort(self):
        with self.assertRaises(CollectionInfrastructureError) as ctx:
            self.run_collect({1: infra, 2: infra, 3: infra})
        message = str(ctx.exception)
        self.assertIn("in a row", message)
        self.assertIn("ModelRequestError", message, "中止必须说明是什么故障")
        self.assertIn("rate limit", message, "中止必须带响应体原文")

    def test_success_resets_the_counter(self):
        """故障 / 正常 / 故障 / 故障 ⇒ 最长连续 2，不该抛。"""
        written, _ = self.run_collect({1: infra, 2: infra, 4: infra, 5: infra})
        self.assertEqual(len(written), 7)

    def test_tolerance_matches_constant(self):
        script = {index: infra for index in range(1, INFRASTRUCTURE_FAILURE_TOLERANCE + 1)}
        with self.assertRaises(CollectionInfrastructureError):
            self.run_collect(script)


class CollectTasksToleranceTest(unittest.TestCase):
    def test_serial_path_also_tolerates_single_failure(self):
        def fake_collect_for_task(task, **kwargs):
            return infra(int(task["task_id"])) if int(task["task_id"]) <= 2 else ordinary(
                int(task["task_id"])
            )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "raw.jsonl"
            with mock.patch.object(rollout, "collect_for_task", fake_collect_for_task):
                written = rollout.collect_tasks(
                    make_tasks(5),
                    client=object(),
                    output_path=output,
                    base_url="http://127.0.0.1:5700",
                )
        self.assertEqual(len(written), 5)

    def test_serial_path_aborts_on_three(self):
        def fake_collect_for_task(task, **kwargs):
            return infra(int(task["task_id"]))

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "raw.jsonl"
            with mock.patch.object(rollout, "collect_for_task", fake_collect_for_task):
                with self.assertRaises(CollectionInfrastructureError):
                    rollout.collect_tasks(
                        make_tasks(5),
                        client=object(),
                        output_path=output,
                        base_url="http://127.0.0.1:5700",
                    )


class InfrastructureDetailTest(unittest.TestCase):
    def test_includes_type_message_and_task(self):
        detail = _infrastructure_detail(infra(77))
        self.assertIn("ModelRequestError", detail)
        self.assertIn("rate limit", detail)
        self.assertIn("77", detail)

    def test_truncates_long_message(self):
        detail = _infrastructure_detail(infra(1, message="x" * 1000))
        self.assertLessEqual(len(detail), 260)

    def test_release_error_takes_priority(self):
        trajectory = {
            "task_id": 5,
            "error": {"type": "ValueError", "message": "other"},
            "release_error": {"type": "ShopEnvironmentError", "message": "lease lost"},
        }
        detail = _infrastructure_detail(trajectory)
        self.assertIn("ShopEnvironmentError", detail)
        self.assertIn("lease lost", detail)

    def test_missing_error_does_not_crash(self):
        self.assertIn("unknown", _infrastructure_detail({"task_id": 9}))


if __name__ == "__main__":
    unittest.main()
