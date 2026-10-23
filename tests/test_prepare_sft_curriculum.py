import json
import unittest

from scripts.prepare_sft_curriculum import build_manifest


def _call(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def _row(task_id):
    asin = f"asin-{task_id}"
    names = ("search_products", "open_product", "select_option", "buy_now")
    messages = [
        {"role": "system", "content": "use tools"},
        {"role": "user", "content": f"buy task {task_id}"},
        _call("search_products", {"query": "item"}, f"{task_id}-1"),
        {"role": "tool", "tool_call_id": f"{task_id}-1", "content": asin},
        _call("open_product", {"asin": asin}, f"{task_id}-2"),
        {
            "role": "tool",
            "tool_call_id": f"{task_id}-2",
            "content": f"asin: {asin}\nprice: 10",
        },
        _call("select_option", {"value": "red"}, f"{task_id}-3"),
        {"role": "tool", "tool_call_id": f"{task_id}-3", "content": "selected"},
        _call("buy_now", {}, f"{task_id}-4"),
        {"role": "tool", "tool_call_id": f"{task_id}-4", "content": "购买已完成。"},
    ]
    return {
        "trajectory_id": f"trajectory-{task_id}",
        "task_id": task_id,
        "messages": messages,
        "tools": [{"type": "function", "function": {"name": name}} for name in names],
    }


def _label(task_id, difficulty, *, rewrite=False, compare=False):
    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "trajectory_complexity": difficulty,
        "needs_query_rewrite": rewrite,
        "needs_candidate_comparison": compare,
    }


class PrepareSftCurriculumTests(unittest.TestCase):
    def test_builds_stable_atomic_buckets_and_cumulative_stages(self):
        rows = [_row(task_id) for task_id in range(1, 31)]
        labels = []
        for task_id in range(1, 11):
            labels.append(_label(task_id, "simple"))
        for task_id in range(11, 21):
            labels.append(_label(task_id, "medium"))
        for task_id in range(21, 26):
            labels.append(_label(task_id, "medium", compare=True))
        for task_id in range(26, 31):
            labels.append(_label(task_id, "hard", rewrite=True, compare=True))

        manifest = build_manifest(
            rows,
            labels,
            evaluation_ids=set(),
            seed=7,
            validation_ratio=0.2,
        )
        again = build_manifest(
            rows,
            labels,
            evaluation_ids=set(),
            seed=7,
            validation_ratio=0.2,
        )

        self.assertEqual(manifest, again)
        self.assertEqual(manifest["counts"]["rows"], 30)
        self.assertEqual(manifest["counts"]["train"], 24)
        self.assertEqual(manifest["counts"]["validation"], 6)
        self.assertEqual(manifest["stages"]["a"]["train_rows"], 8)
        self.assertEqual(manifest["stages"]["b"]["train_rows"], 16)
        self.assertEqual(manifest["stages"]["c"]["train_rows"], 24)
        train_ids = {
            task_id
            for bucket in manifest["buckets"].values()
            for task_id in bucket["train_task_ids"]
        }
        validation_ids = {
            task_id
            for bucket in manifest["buckets"].values()
            for task_id in bucket["validation_task_ids"]
        }
        self.assertFalse(train_ids & validation_ids)

    def test_rejects_evaluation_overlap_and_invalid_tool_arguments(self):
        row = _row(1)
        label = _label(1, "simple")
        with self.assertRaisesRegex(ValueError, "evaluation overlap"):
            build_manifest([row], [label], evaluation_ids={1})

        row["messages"][2]["tool_calls"][0]["function"]["arguments"] = "not-json"
        with self.assertRaisesRegex(ValueError, "invalid tool arguments"):
            build_manifest([row], [label], evaluation_ids=set())

    def test_records_review_flags_without_dropping_valid_hard_cases(self):
        row = _row(1)
        label = _label(1, "hard", compare=True)

        manifest = build_manifest([row], [label], evaluation_ids=set())

        self.assertEqual(manifest["counts"]["rows"], 1)
        self.assertEqual(
            manifest["review_flags"]["candidate_comparison_under_evidenced"],
            [1],
        )

    def test_accepts_sanitized_terminal_observation_after_purchase(self):
        row = _row(1)
        row["messages"][-1]["content"] = (
            "[SHOPPING_OBSERVATION_V2]\npage_type: terminal\n\n"
            "搜索功能是否可用: False\n可点击的按钮: []"
        )

        manifest = build_manifest(
            [row],
            [_label(1, "simple")],
            evaluation_ids=set(),
        )

        self.assertEqual(manifest["counts"]["rows"], 1)

    def test_behavior_mode_buckets_by_action_not_difficulty(self):
        """倒序课程按「做没做探索」分桶，与难度标签无关。

        2026-10-22：旧的三段按难度分桶，模板被练 3 遍、探索只 1 遍（曝光比 0.19）。
        新口径要让探索轨迹单独成桶、且在阶段 A 独占。
        """
        plain = _row(1)
        # 换词：同题搜两个不同查询词
        rewritten = _row(2)
        rewritten["messages"].insert(
            5, _call("search_products", {"query": "other item"}, "2-1b")
        )
        labels = [_label(1, "hard"), _label(2, "simple")]

        manifest = build_manifest(
            [plain, rewritten],
            labels,
            evaluation_ids=set(),
            bucket_mode="behavior",
        )

        self.assertEqual(manifest["bucket_mode"], "behavior")
        self.assertEqual(manifest["buckets"]["template"]["rows"], 1)
        self.assertEqual(manifest["buckets"]["exploratory"]["rows"], 1)
        # 阶段 A 只放探索；阶段 B 才是全量
        self.assertEqual(manifest["stages"]["a"]["buckets"], ["exploratory"])
        self.assertEqual(
            manifest["stages"]["b"]["buckets"], ["exploratory", "template", "other"]
        )
        self.assertEqual(list(manifest["stages"]), ["a", "b"])

    def test_behavior_bucket_ignores_difficulty_label(self):
        """难度标签是 hard 但只搜一次开一个 ⇒ 仍然算模板。"""
        from scripts.prepare_sft_curriculum import behavior_bucket, behavior_profile

        profile = behavior_profile(_row(1))
        self.assertFalse(profile["explored"])
        self.assertEqual(behavior_bucket(profile), "template")

    def test_behavior_bucket_marks_paging_as_exploratory(self):
        from scripts.prepare_sft_curriculum import behavior_bucket, behavior_profile

        row = _row(1)
        row["messages"].insert(5, _call("next_page", {}, "1-page"))
        profile = behavior_profile(row)
        self.assertTrue(profile["paging"])
        self.assertEqual(behavior_bucket(profile), "exploratory")

    def test_single_stage_plan_puts_everything_in_one_stage(self):
        """单阶段（M10-R v2）：一个阶段吃全部桶、1 个 epoch。

        为什么不再分两段：A 段单独评测证明模型**忠实复现数据里的行为比例**
        （数据翻页 28% → 输出 31%），而 B 段从「翻页 30%」出发、用 11.9% 的全量池
        训 137 步，被反向修正**过冲到 1%**。只有一个阶段才不会过冲，所以
        「全量 1 个 epoch」本身就是方案，不是简化。
        """
        rows = [_row(task_id) for task_id in range(1, 5)]
        labels = [_label(task_id, "simple") for task_id in range(1, 5)]

        manifest = build_manifest(
            rows,
            labels,
            evaluation_ids=set(),
            bucket_mode="behavior",
            stage_plan="single",
        )

        self.assertEqual(manifest["stage_plan"], "single")
        self.assertEqual(list(manifest["stages"]), ["full"])
        full = manifest["stages"]["full"]
        self.assertEqual(full["buckets"], ["exploratory", "template", "other"])
        self.assertEqual(full["epochs"], 1.0)
        # lr 与旧课程末段（M5-B stage-c）一致，保证与 62% 基线可比
        self.assertEqual(full["learning_rate"], 5e-5)
        # 单阶段必须吃到全量，漏一个桶就等于「又变成只训探索桶」的 v1
        self.assertEqual(full["train_rows"], manifest["counts"]["train"])

    def test_stage_plan_defaults_to_reverse(self):
        """不传 --stage-plan 时行为必须与 v1 逐字节一致（manifest 只多一个键）。"""
        manifest = build_manifest([_row(1)], [_label(1, "simple")], evaluation_ids=set())

        self.assertEqual(manifest["stage_plan"], "reverse")
        self.assertEqual(list(manifest["stages"]), ["a", "b", "c"])

    def test_rejects_unknown_stage_plan_and_bucket_mode(self):
        with self.assertRaisesRegex(ValueError, "stage_plan"):
            build_manifest(
                [_row(1)],
                [_label(1, "simple")],
                evaluation_ids=set(),
                stage_plan="two-pass",
            )
        with self.assertRaisesRegex(ValueError, "bucket_mode"):
            build_manifest(
                [_row(1)],
                [_label(1, "simple")],
                evaluation_ids=set(),
                bucket_mode="behaviour",
            )


if __name__ == "__main__":
    unittest.main()
