"""根据最新 observation 守卫模型的工具调用。

ShopSimulator 的按钮和商品只在当前页面有效。动作守卫在请求到达环境前检查
这一点，并把拒绝原因反馈给模型，避免非法点击污染环境状态。
"""

import json
import re

from shopping_grpo.environment.product_id import PRODUCT_ID_CAPTURE
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS, tool_call_to_action


RUNTIME_GUARD_FIELD = "runtime_action_guard"
NAVIGATION_BUTTONS = {
    "description",
    "features",
    "reviews",
    "attributes",
    "next >",
    "< prev",
    "back to search",
    "buy now",
}
TOOL_ARGUMENT_NAMES = {
    tool["function"]["name"]: set(tool["function"]["parameters"].get("properties", {}))
    for tool in SHOP_TOOL_SCHEMAS
}


def action_reject_reason(
    name,
    arguments,
    observation,
    opened_candidates=0,
    available_candidates=None,
    min_opened_before_buy=0,
):
    """返回动作拒绝原因；``None`` 表示允许执行。

检查顺序很重要：先拦截 schema 外字段，再处理无需页面状态的动作，最后只允许
点击最新 observation 中仍然存在的目标。

``min_opened_before_buy`` 是**诊断开关**（默认 0 = 关闭）：大于 0 时，``buy_now``
在打开过的不同候选数不足之前会被拒绝。它只改"允不允许下单"，不改奖励、不改题目、
不泄露正确答案；用途是量出「多看几个候选」这条路的收益上界，见
``scripts/analyze_candidate_exploration.py``。
"""
    extra_argument_names = _schema_extra_argument_names(name, arguments)
    if extra_argument_names:
        return "schema_extra_arguments:" + ",".join(extra_argument_names)
    if name == "think":
        return None
    if name == "finish_without_purchase":
        if arguments.get("reason") != "no_suitable_product":
            return "invalid_finish_reason"
        return None
    if name == "search_products":
        if "搜索功能是否可用: False" in observation:
            return "search_not_available_on_current_page"
        return None
    if not observation:
        return "missing_previous_observation"
    if name == "open_product":
        asin = str(arguments.get("asin", ""))
        if asin not in product_ids(observation):
            return "click_not_in_previous_observation"
        return None

    try:
        action = tool_call_to_action(name, arguments)
    except Exception:
        return "unknown_or_invalid_tool"
    if not isinstance(action, str) or not action.startswith("click[") or not action.endswith("]"):
        return "click_not_in_previous_observation"
    target = action[6:-1].casefold()
    if name == "select_option" and target in NAVIGATION_BUTTONS:
        return "select_option_is_navigation_button"
    if target not in {button.casefold() for button in clickable_buttons(observation)}:
        return "click_not_in_previous_observation"
    if name == "buy_now":
        required = required_candidate_opens(min_opened_before_buy, available_candidates)
        if required and int(opened_candidates) < required:
            return f"insufficient_candidates_compared:{int(opened_candidates)}/{required}"
    return None


def required_candidate_opens(min_opened_before_buy, available_candidates=None):
    """把「至少要打开几个候选」夹到实际可选范围内，避免题目本身无解。"""
    minimum = int(min_opened_before_buy or 0)
    if minimum <= 0:
        return 0
    if available_candidates is None:
        return minimum
    return max(0, min(minimum, int(available_candidates)))


def _schema_extra_argument_names(name, arguments):
    """只拒绝 schema 未声明字段；缺少必填字段仍由工具动作转换报错。"""
    allowed_names = TOOL_ARGUMENT_NAMES.get(name)
    if allowed_names is None or not isinstance(arguments, dict):
        return []
    return sorted(set(arguments) - allowed_names)


def action_guard_tool_message(tool_call, reason, observation, hint=None):
    """构造标准 tool error observation，让 Agent 感知调用未被执行。"""
    targets = clickable_buttons(observation)
    asins = product_ids(observation)
    parts = []
    if asins:
        parts.append("可打开的商品 ASIN: " + ", ".join(asins[:20]))
    if targets:
        parts.append("可点击按钮: " + ", ".join(targets[:30]))
    allowed = "；".join(parts) or "当前页面没有可用点击目标"
    return_tools = []
    normalized_targets = {target.casefold() for target in targets}
    if "< prev" in normalized_targets:
        return_tools.append("prev_page")
    if "back to search" in normalized_targets:
        return_tools.append("back_to_search")
    only_return_buttons = bool(normalized_targets) and normalized_targets <= {"< prev", "back to search"}
    if only_return_buttons:
        correction = f"你处于信息子页，下一步只能调用 {' 或 '.join(return_tools)}。"
    else:
        correction = "下一步只能从当前页面列出的目标中选择。"
    if hint:
        correction = f"{hint}{correction}"
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": (tool_call.get("function") or {}).get("name"),
        RUNTIME_GUARD_FIELD: True,
        "content": (
            f"上一工具调用被本地动作守卫拒绝（{reason}），未执行。"
            f"{correction}{allowed}。不要重复使用历史页面目标；只调用一个合法工具。"
        ),
    }


def product_ids(observation):
    """提取当前 observation 中可打开的商品 ID，并保持出现顺序。"""
    return list(
        dict.fromkeys(
            re.findall(
                rf"(?m)^\d+\|({PRODUCT_ID_CAPTURE})\|",
                observation,
            )
        )
    )


def clickable_buttons(observation):
    """读取 observation footer 中当前页面实际可点击的按钮。"""
    match = re.search(r"可点击的按钮:\s*(\[[^\n]*\])", observation)
    if not match:
        return []
    try:
        buttons = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return [button for button in buttons if isinstance(button, str)]
