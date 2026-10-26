"""系统提示词变体的口径回归。

换提示词等于换 agent 配置，所以三条必须钉死：
* 默认不传时，行为与冻结口径完全一致（历史分数才可比）；
* 传了就真的进了第一条 system 消息，并记下指纹；
* 仓库里那份变体只许改「搜索查询」那一条规则，别的规则不许动——
  否则跑出来的差异说不清是哪个改动造成的。
"""

import json
import re

import pytest

from shopping_grpo.evaluation import rollout
from shopping_grpo.evaluation.rollout import (
    SYSTEM_PROMPT,
    _initial_messages,
    collect_for_task,
    load_system_prompt,
    system_prompt_digest,
)

VARIANT_PATH = "configs/prompts/query-attributes.txt"


def test_digest_is_stable_and_discriminating():
    assert system_prompt_digest("abc") == system_prompt_digest("abc")
    assert system_prompt_digest("abc") != system_prompt_digest("abd")
    assert len(system_prompt_digest(SYSTEM_PROMPT)) == 16


def test_load_none_returns_none():
    # None 是「用冻结提示」的信号，不是空串。
    assert load_system_prompt(None) is None


def test_load_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_system_prompt(str(path))


def test_load_reads_text(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("你好", encoding="utf-8")
    assert load_system_prompt(str(path)) == "你好"


def test_initial_messages_defaults_to_frozen_prompt():
    messages = _initial_messages({"task_id": 1}, {"instruction": "买木梳"})
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[-1] == {"role": "user", "content": "买木梳"}


def test_initial_messages_uses_override():
    messages = _initial_messages({"task_id": 1}, {"instruction": "买木梳"}, system_prompt="变体")
    assert messages[0] == {"role": "system", "content": "变体"}


def test_task_level_prompt_wins_over_override():
    # 行自带 prompt（训练样本回放）时不该被评测期的变体覆盖。
    task = {"task_id": 1, "prompt": [{"role": "system", "content": "行内提示"}]}
    messages = _initial_messages(task, {"instruction": "买木梳"}, system_prompt="变体")
    assert messages[0] == {"role": "system", "content": "行内提示"}
    assert len(messages) == 2


class _Client:
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


class _Env:
    def __init__(self, **_):
        self.actions = []

    def reset(self, task_id):
        return {"observation": "开始", "observation_state": None}

    def step(self, action):
        self.actions.append(action)
        return {"observation_state": None, "reward": 0.0, "done": False}

    def release(self):
        return None


def test_collect_for_task_records_digest_and_uses_override():
    trajectory = collect_for_task(
        {"task_id": 1},
        client=_Client([("search_products", {"query": "木梳"})]),
        env_factory=lambda **kwargs: _Env(),
        max_steps=2,
        system_prompt="变体提示",
    )
    assert trajectory["messages"][0]["content"] == "变体提示"
    assert trajectory["system_prompt_sha256"] == system_prompt_digest("变体提示")


def test_collect_for_task_default_digest_matches_frozen_prompt():
    trajectory = collect_for_task(
        {"task_id": 1},
        client=_Client([("search_products", {"query": "木梳"})]),
        env_factory=lambda **kwargs: _Env(),
        max_steps=2,
    )
    assert trajectory["system_prompt_sha256"] == system_prompt_digest(SYSTEM_PROMPT)


def _rule_lines(text):
    rules = {}
    for line in text.splitlines():
        match = re.match(r"^(\d+)\.\s", line)
        if match:
            rules[match.group(1)] = line
    return rules


def test_variant_only_changes_the_query_rule():
    variant = load_system_prompt(VARIANT_PATH)
    assert variant is not None
    frozen_rules = _rule_lines(SYSTEM_PROMPT)
    variant_rules = _rule_lines(variant)
    assert set(frozen_rules) == set(variant_rules), "变体的规则条数/编号必须与冻结提示一致"
    changed = [key for key in frozen_rules if frozen_rules[key] != variant_rules[key]]
    assert changed == ["3"], f"只许改第 3 条（搜索查询），实际改了 {changed}"


def test_variant_actually_asks_for_category_word_and_units():
    variant = load_system_prompt(VARIANT_PATH)
    assert "品类词" in variant
    assert "1公斤" in variant and "14g" in variant
    # 冻结提示里的「别复制整段需求」「缩短查询」是漏词失败的疑似成因，变体必须去掉。
    assert "不要机械复制整段需求" not in variant
    assert "缩短查询" not in variant


def test_frozen_prompt_untouched():
    # 任何人不小心改到冻结提示，都会让历史分数失去可比性。
    assert "不要机械复制整段需求" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.endswith("只有环境报告任务结束后才停止。\n")
    assert rollout.SYSTEM_PROMPT is SYSTEM_PROMPT
