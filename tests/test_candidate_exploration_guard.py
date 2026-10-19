"""诊断开关 ``min_opened_before_buy`` 的口径回归。

这个开关只改「允不允许下单」，任何情况下都不能：
* 改变默认行为（默认 0 时必须与历史完全一致）；
* 泄露正确答案（提示里不能出现金标 ASIN 或规格值）；
* 把题目变成无解（候选数不足时要求自动下调）。
"""

import json

import pytest

from shopping_grpo.environment.actions import (
    action_guard_tool_message,
    action_reject_reason,
    required_candidate_opens,
)

PRODUCT_PAGE = """[SHOPPING_OBSERVATION_V2]
page_type: product_detail
asin: 11111111
title: 测试商品
brand: 甲
category: 节庆
price: 5.0
key_attributes: 高档
selected_options: {}
available_options: {}

搜索功能是否可用: False
可点击的按钮: ["back to search", "< prev", "buy now"]
"""

SEARCH_PAGE = """[SHOPPING_OBSERVATION_V2]
page_type: search_results
query: 木梳
格式: rank|asin|price|brand|category|key_attributes|title
1|11111111|5.0|甲|节庆|高档|A款
2|22222222|6.0|乙|节庆|高档|B款
products_shown: 2

搜索功能是否可用: True
可点击的按钮: ["11111111", "22222222"]
"""

BUY = {"name": "buy_now", "arguments": {}}


def _reason(opened, minimum, available=None):
    return action_reject_reason(
        BUY["name"],
        BUY["arguments"],
        PRODUCT_PAGE,
        opened_candidates=opened,
        available_candidates=available,
        min_opened_before_buy=minimum,
    )


def test_switch_off_by_default_keeps_historical_behaviour():
    assert _reason(0, 0) is None
    assert _reason(0, 0, available=20) is None


def test_buy_blocked_until_enough_candidates_opened():
    assert _reason(1, 5, available=20) == "insufficient_candidates_compared:1/5"
    assert _reason(4, 5, available=20) == "insufficient_candidates_compared:4/5"
    assert _reason(5, 5, available=20) is None


def test_requirement_clamped_to_available_candidates():
    # 结果页只有 2 个候选时，不能要求打开 5 个，否则题目无解。
    assert _reason(2, 5, available=2) is None
    assert _reason(1, 5, available=2) == "insufficient_candidates_compared:1/2"


def test_required_candidate_opens_clamps():
    assert required_candidate_opens(0) == 0
    assert required_candidate_opens(5) == 5
    assert required_candidate_opens(5, available_candidates=2) == 2
    assert required_candidate_opens(5, available_candidates=0) == 0


def test_guard_still_rejects_illegal_page_targets_first():
    # 商品页没有 buy now 时，应报页面目标错误，而不是候选数不足。
    page = PRODUCT_PAGE.replace('"buy now"', '"description"')
    reason = action_reject_reason(
        "buy_now", {}, page, opened_candidates=0, available_candidates=20, min_opened_before_buy=5
    )
    assert reason == "click_not_in_previous_observation"


def test_other_tools_are_not_affected_by_the_switch():
    assert (
        action_reject_reason(
            "select_option",
            {"value": "高档"},
            PRODUCT_PAGE.replace("buy now", "高档"),
            opened_candidates=0,
            available_candidates=20,
            min_opened_before_buy=5,
        )
        is None
    )
    assert (
        action_reject_reason(
            "open_product",
            {"asin": "11111111"},
            SEARCH_PAGE,
            opened_candidates=0,
            available_candidates=2,
            min_opened_before_buy=5,
        )
        is None
    )


def test_guard_message_carries_hint_without_leaking_answers():
    message = action_guard_tool_message(
        {"id": "call-1", "function": {"name": "buy_now"}},
        "insufficient_candidates_compared:1/5",
        PRODUCT_PAGE,
        hint="购买前必须先比较至少 5 个不同候选，当前只打开过 1 个。",
    )
    content = message["content"]
    assert "至少 5 个不同候选" in content
    assert "insufficient_candidates_compared:1/5" in content
    # 提示只能讲流程，不能出现任何具体 ASIN 或规格值。
    assert "22222222" not in content
    assert "高档" not in content


def test_rollout_hint_helper_matches_guard_reason():
    from shopping_grpo.evaluation.rollout import _candidate_hint

    hint = _candidate_hint("insufficient_candidates_compared:1/5", 5, 20, 1)
    assert "至少 5 个" in hint
    assert _candidate_hint("click_not_in_previous_observation", 5, 20, 1) is None


class _FakeClient:
    """按脚本返回预设动作，把 buy_now 放在第一个候选之后。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def complete(self, messages, tools):
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        name, args = self.script[index]
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call-{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        }

    def project_observation(self, tool_name, observation, parameters=None):
        return observation, None


class _FakeEnv:
    """最小环境桩：第一次搜索给两页候选，打开商品给商品页。"""

    def __init__(self, **_):
        self.actions = []
        self.opened = []

    def reset(self, task_id):
        return {"observation": "开始", "observation_state": None}

    def step(self, action):
        self.actions.append(action)
        if action.startswith("search["):
            return {"observation_state": _search_state(), "reward": 0.0, "done": False}
        if action.startswith("click[") and action[6:-1] in {"11111111", "22222222"}:
            asin = action[6:-1]
            self.opened.append(asin)
            return {"observation_state": _product_state(asin), "reward": 0.0, "done": False}
        if action == "click[Buy Now]":
            return {
                "observation_state": _terminal_state(),
                "reward": 1.0,
                "done": True,
                "purchase": {"asin": self.opened[-1] if self.opened else None},
            }
        return {"observation_state": _search_state(), "reward": 0.0, "done": False}

    def release(self):
        return None


def _search_state():
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "search_results",
        "query": "木梳",
        "normalized_query": "木梳",
        "page": 1,
        "total_pages": 1,
        "total_results": 2,
        "rank_start": 1,
        "rank_end": 2,
        "products": [
            {
                "rank": 1,
                "asin": "11111111",
                "price": "5.0",
                "brand": "甲",
                "category": "节庆",
                "key_attributes": ["高档"],
                "title": "A款",
            },
            {
                "rank": 2,
                "asin": "22222222",
                "price": "6.0",
                "brand": "乙",
                "category": "节庆",
                "key_attributes": ["高档"],
                "title": "B款",
            },
        ],
        "search_available": True,
        "actions": ["11111111", "22222222"],
    }


def _product_state(asin):
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "product_detail",
        "product": {
            "asin": asin,
            "title": f"商品{asin}",
            "brand": "甲",
            "category": "节庆",
            "price": "5.0",
            "key_attributes": ["高档"],
        },
        "selected_options": {},
        "available_options": {},
        "search_available": False,
        "actions": ["back to search", "< prev", "buy now"],
    }


def _terminal_state():
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "terminal",
        "search_available": False,
        "actions": [],
    }


def test_rollout_blocks_buy_until_minimum_candidates_opened():
    from shopping_grpo.evaluation.rollout import collect_for_task

    script = [
        ("search_products", {"query": "木梳"}),
        ("open_product", {"asin": "11111111"}),
        ("buy_now", {}),
        ("back_to_search", {}),
        ("open_product", {"asin": "22222222"}),
        ("buy_now", {}),
    ]
    client = _FakeClient(script)
    env = _FakeEnv()
    trajectory = collect_for_task(
        {"task_id": 1},
        client=client,
        env_factory=lambda **kwargs: env,
        max_steps=10,
        min_opened_before_buy=2,
    )

    reasons = [item["reason"] for item in trajectory["blocked_tool_calls"]]
    assert reasons == ["insufficient_candidates_compared:1/2"]
    assert trajectory["opened_candidates"] == ["11111111", "22222222"]
    assert trajectory["done"] is True
    assert trajectory["min_opened_before_buy"] == 2


def test_rollout_switch_off_allows_immediate_buy():
    from shopping_grpo.evaluation.rollout import collect_for_task

    script = [
        ("search_products", {"query": "木梳"}),
        ("open_product", {"asin": "11111111"}),
        ("buy_now", {}),
    ]
    client = _FakeClient(script)
    env = _FakeEnv()
    trajectory = collect_for_task(
        {"task_id": 1},
        client=client,
        env_factory=lambda **kwargs: env,
        max_steps=10,
    )
    assert trajectory["blocked_tool_calls"] == []
    assert trajectory["done"] is True


@pytest.mark.parametrize("minimum", [0, 1, 2])
def test_rollout_never_exceeds_step_budget(minimum):
    from shopping_grpo.evaluation.rollout import collect_for_task

    script = [("search_products", {"query": "木梳"}), ("open_product", {"asin": "11111111"})]
    client = _FakeClient(script)
    env = _FakeEnv()
    trajectory = collect_for_task(
        {"task_id": 1},
        client=client,
        env_factory=lambda **kwargs: env,
        max_steps=4,
        min_opened_before_buy=minimum,
    )
    assert len(trajectory["steps"]) <= 4
