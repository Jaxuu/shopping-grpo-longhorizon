"""`scripts/build_m10r_pool.py` 的口径固定。

钉死五件事：

1. **「买到谁」只能靠「`buy_now` 之前最后一个 `open_product`」推**——`buy_now` 不带参数，
   只能买当前打开的商品；写成 `purchased != opened[-1]` 这类判断会永远为假。
2. **名次取「所有搜索/翻页里最好的那个」**，与 `filter_m10_rank2` 同源。
3. **配比公式**：`r1 = rank_ge2 × (1-share) / share`，且不能超过现有第 1 名条数。
4. **降采样只动第 1 名样本**，固定 seed 可复现，绝不动「答案不在第 1 名」的样本。
5. **合并前必须挡住三类泄漏**：增量与池子 task_id 重叠、增量里混进第 1 名样本、训练池与评测集重叠。
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

import build_m10r_pool as module  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": "search_products", "parameters": {}}},
    {"type": "function", "function": {"name": "open_product", "parameters": {}}},
    {"type": "function", "function": {"name": "select_option", "parameters": {}}},
    {"type": "function", "function": {"name": "buy_now", "parameters": {}}},
]


def search_observation(rows: list[tuple[int, str]]) -> str:
    lines = ["[SHOPPING_OBSERVATION_V2]", "page_type: search_results"]
    lines += [f"{rank}|{asin}|9.0|品牌|类目|属性|标题" for rank, asin in rows]
    return "\n".join(lines)


def detail_observation(asin: str) -> str:
    return f"[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\nasin: {asin}\nprice: 9.0"


def make_row(task_id: int, page: list[tuple[int, str]], opens: list[str]) -> dict:
    """构造一条能过 `prepare_sft_curriculum._validate_row` 的 SFT 行。"""
    messages: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "任务"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "s1",
                    "type": "function",
                    "function": {"name": "search_products", "arguments": '{"query": "x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "s1", "content": search_observation(page)},
    ]
    for index, asin in enumerate(opens):
        call_id = f"o{index}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "open_product", "arguments": json.dumps({"asin": asin})},
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": detail_observation(asin)}
        )
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "b1",
                    "type": "function",
                    "function": {"name": "buy_now", "arguments": "{}"},
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": "b1", "content": "购买已完成。"})
    return {
        "trajectory_id": f"t{task_id}",
        "task_id": task_id,
        "messages": messages,
        "tools": TOOLS,
    }


def make_requery_row(task_id: int, *, second_query: str = "乳胶枕 护颈") -> dict:
    """换词重搜的轨迹：第一次搜不到 gold，换词后 gold 排第 1 名。

    这是 2026-10-21 探针量到的老师真实策略（90.9% 的轨迹会换词）。
    它的 min 名次是 1，但教的不是「开第一个」——所以必须能进增量。
    """
    messages: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "任务"},
    ]
    for index, (query, page) in enumerate(
        [("乳胶枕", [(1, "A")]), (second_query, [(1, "G"), (2, "A")])]
    ):
        call_id = f"q{index}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "arguments": json.dumps({"query": query}, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": search_observation(page)}
        )
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "o1",
                    "type": "function",
                    "function": {"name": "open_product", "arguments": json.dumps({"asin": "G"})},
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": "o1", "content": detail_observation("G")})
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "b1",
                    "type": "function",
                    "function": {"name": "buy_now", "arguments": "{}"},
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": "b1", "content": "购买已完成。"})
    return {
        "trajectory_id": f"t{task_id}",
        "task_id": task_id,
        "messages": messages,
        "tools": TOOLS,
    }


class PurchasedAsinTest(unittest.TestCase):
    def test_last_open_before_buy(self) -> None:
        row = make_row(1, [(1, "A"), (3, "G")], ["A", "G"])
        self.assertEqual(module.purchased_asin(row["messages"]), "G")

    def test_single_open(self) -> None:
        row = make_row(2, [(1, "A"), (4, "G")], ["G"])
        self.assertEqual(module.purchased_asin(row["messages"]), "G")

    def test_without_buy_returns_none(self) -> None:
        row = make_row(3, [(1, "A")], ["A"])
        messages = [
            message
            for message in row["messages"]
            if not any(
                (call.get("function") or {}).get("name") == "buy_now"
                for call in message.get("tool_calls") or []
            )
            and message.get("tool_call_id") != "b1"
        ]
        self.assertIsNone(module.purchased_asin(messages))

    def test_open_after_buy_does_not_count(self) -> None:
        """买了之后再打开别的商品，买的仍然是买之前那个。"""
        row = make_row(4, [(1, "A"), (2, "B")], ["A"])
        messages = row["messages"][:-1] + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "x",
                        "type": "function",
                        "function": {"name": "open_product", "arguments": '{"asin": "B"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "x", "content": detail_observation("B")},
        ]
        # 末尾补回终局 tool 响应，模拟轨迹尾部
        messages.append({"role": "tool", "tool_call_id": "b1", "content": "购买已完成。"})
        self.assertEqual(module.purchased_asin(messages), "A")


class SeenRanksTest(unittest.TestCase):
    def test_best_rank_across_pages(self) -> None:
        messages = [
            {"role": "tool", "content": search_observation([(21, "G")])},
            {"role": "tool", "content": search_observation([(3, "G")])},
        ]
        self.assertEqual(module.seen_ranks(messages)["G"], 3)

    def test_detail_page_ignored(self) -> None:
        messages = [{"role": "tool", "content": detail_observation("G")}]
        self.assertEqual(module.seen_ranks(messages), {})


class DescribeRowTest(unittest.TestCase):
    def test_rank_and_open_count(self) -> None:
        row = make_row(1, [(1, "A"), (5, "G")], ["A", "G"])
        profile = module.describe_row(row)
        self.assertEqual(profile["rank"], 5)
        self.assertEqual(profile["distinct_opens"], 2)
        self.assertEqual(profile["bought"], "G")

    def test_reopening_same_item_is_one_distinct(self) -> None:
        row = make_row(2, [(1, "A"), (5, "G")], ["G", "G"])
        self.assertEqual(module.describe_row(row)["distinct_opens"], 1)

    def test_never_seen_asin_has_no_rank(self) -> None:
        row = make_row(3, [(1, "A")], ["Z"])
        self.assertIsNone(module.describe_row(row)["rank"])

    def test_paging_flag(self) -> None:
        """翻页示范必须单独可数：A 桶只能靠它救，混在「名次≥2」里会看不见。"""
        row = make_row(4, [(1, "A"), (25, "G")], ["G"])
        self.assertFalse(module.describe_row(row)["paging"])
        row["messages"].insert(
            4,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "n1",
                        "type": "function",
                        "function": {"name": "next_page", "arguments": "{}"},
                    }
                ],
            },
        )
        row["messages"].insert(5, {"role": "tool", "tool_call_id": "n1", "content": "ok"})
        self.assertTrue(module.describe_row(row)["paging"])


class RankBucketTest(unittest.TestCase):
    def test_boundaries(self) -> None:
        self.assertEqual(module.rank_bucket(1), "1")
        self.assertEqual(module.rank_bucket(2), "2-3")
        self.assertEqual(module.rank_bucket(3), "2-3")
        self.assertEqual(module.rank_bucket(4), "4-20")
        self.assertEqual(module.rank_bucket(20), "4-20")
        self.assertEqual(module.rank_bucket(21), ">20")
        self.assertEqual(module.rank_bucket(None), "未出现")


class PlanMixTest(unittest.TestCase):
    def test_keep_all_by_default(self) -> None:
        plan = module.plan_mix(784, 851, None)
        self.assertEqual(plan["rank1_kept"], 851)
        self.assertEqual(plan["rank1_dropped"], 0)
        self.assertAlmostEqual(plan["achieved_share"], 784 / 1635)

    def test_sixty_percent(self) -> None:
        plan = module.plan_mix(784, 851, 0.60)
        self.assertEqual(plan["rank1_kept"], 523)
        self.assertEqual(plan["rank1_dropped"], 328)
        self.assertAlmostEqual(plan["achieved_share"], 0.60, places=3)

    def test_unreachable_target_drops_every_rank1(self) -> None:
        """要 99% 但只有 10 条「名次≥2」，只能把第 1 名丢到 0，不会倒过来造数据。"""
        plan = module.plan_mix(10, 5, 0.99)
        self.assertEqual(plan["rank1_kept"], 0)
        self.assertEqual(plan["rank1_dropped"], 5)
        self.assertAlmostEqual(plan["achieved_share"], 1.0)

    def test_rejects_invalid_share(self) -> None:
        for share in (0.0, 1.0, -0.5, 2.0):
            with self.assertRaises(ValueError):
                module.plan_mix(10, 10, share)


class DownsampleTest(unittest.TestCase):
    def profiles(self) -> list[dict]:
        return [
            {"task_id": i, "rank": 1 if i % 2 == 0 else 5, "distinct_opens": 1, "steps": 5}
            for i in range(10)
        ]

    def test_deterministic(self) -> None:
        first = module.downsample_rank1(self.profiles(), 2, 7)
        second = module.downsample_rank1(self.profiles(), 2, 7)
        self.assertEqual(first, second)

    def test_only_rank1_dropped(self) -> None:
        profiles = self.profiles()
        dropped = module.downsample_rank1(profiles, 2, 7)
        self.assertEqual(len(dropped), 3)
        for index in dropped:
            self.assertEqual(profiles[index]["rank"], 1)

    def test_noop_when_keep_everything(self) -> None:
        self.assertEqual(module.downsample_rank1(self.profiles(), 99, 7), set())


class MainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.evaluation = self.dir / "eval.jsonl"
        self.evaluation.write_text(
            json.dumps({"task_id": 900}) + "\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, name: str, rows: list[dict]) -> Path:
        path = self.dir / name
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def run_main(self, pool: Path, increment: Path, *extra: str) -> int:
        return module.main(
            [
                "--pool", str(pool),
                "--increment", str(increment),
                "--out-dir", str(self.dir / "out"),
                "--labels", str(self.dir / "labels.jsonl"),
                "--evaluation", str(self.evaluation),
                *extra,
            ]
        )

    def test_merges_and_carries_labels(self) -> None:
        pool = self.write("pool.jsonl", [make_row(1, [(1, "A")], ["A"])])
        increment = self.write("inc.jsonl", [make_row(2, [(1, "A"), (5, "G")], ["G"])])
        (self.dir / "labels.jsonl").write_text(
            json.dumps({"task_id": 1, "difficulty": "simple"}) + "\n", encoding="utf-8"
        )
        self.assertEqual(self.run_main(pool, increment), 0)

        merged = module.read_jsonl(self.dir / "out" / "all.jsonl")
        self.assertEqual([row["task_id"] for row in merged], [1, 2])
        labels = module.read_jsonl(self.dir / "out" / "difficulty_labels.jsonl")
        self.assertEqual([row["task_id"] for row in labels], [1])
        report = json.loads((self.dir / "out" / "build_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["counts"]["labels_pending"], 1)
        self.assertEqual(report["counts"]["merged_rows"], 2)

    def test_rejects_task_id_overlap(self) -> None:
        pool = self.write("pool.jsonl", [make_row(1, [(1, "A")], ["A"])])
        increment = self.write("inc.jsonl", [make_row(1, [(1, "A"), (5, "G")], ["G"])])
        with self.assertRaises(SystemExit):
            self.run_main(pool, increment)

    def test_rejects_rank1_in_increment(self) -> None:
        """没有探索行为、直接买第 1 名 ⇒ 拒收（这是要挡的那类样本）。"""
        pool = self.write("pool.jsonl", [make_row(1, [(1, "A")], ["A"])])
        increment = self.write("inc.jsonl", [make_row(2, [(1, "G")], ["G"])])
        with self.assertRaises(SystemExit):
            self.run_main(pool, increment)

    def test_admits_requery_row_whose_gold_ends_up_rank1(self) -> None:
        """换词重搜后 gold 排第 1 名 ⇒ 必须收，这是老师真实策略的示范。"""
        pool = self.write("pool.jsonl", [make_row(1, [(1, "A")], ["A"])])
        increment = self.write("inc.jsonl", [make_requery_row(2)])
        self.assertEqual(self.run_main(pool, increment), 0)
        report = json.loads((self.dir / "out" / "build_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["exploratory_rows_in_increment"], 1)
        self.assertEqual(report["requery_rows_in_increment"], 1)
        self.assertEqual(report["plain_rank1_rows_in_increment"], 0)
        # 合并池里还剩 1 条「纯第 1 名」——那是旧池子本来就有的，正是要稀释的对象
        self.assertEqual(report["plain_rank1_rows"], 1)
        # 名次仍是 1（min 跨页），配比不能被它充成「名次≥2」
        self.assertEqual(report["rank_ge2_rows"], 0)

    def test_rejects_evaluation_leak(self) -> None:
        pool = self.write("pool.jsonl", [make_row(1, [(1, "A")], ["A"])])
        increment = self.write("inc.jsonl", [make_row(900, [(1, "A"), (5, "G")], ["G"])])
        with self.assertRaises(SystemExit):
            self.run_main(pool, increment)

    def test_rejects_extra_row_keys(self) -> None:
        row = make_row(1, [(1, "A")], ["A"])
        row["extra"] = 1
        pool = self.write("pool.jsonl", [row])
        increment = self.write("inc.jsonl", [make_row(2, [(1, "A"), (5, "G")], ["G"])])
        with self.assertRaises(SystemExit):
            self.run_main(pool, increment)


if __name__ == "__main__":
    unittest.main()
