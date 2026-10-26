"""`LocalTextTokenCounter` 的单测。

它的存在理由：评测远端 API 模型（老师）时没有 vLLM 的 `/tokenize`，但 observation
预算必须与本地模型同口径。如果这里数错，截断位置就会偏，对比直接失真。
"""

from __future__ import annotations

import pytest

from shopping_grpo.environment.context import LocalTextTokenCounter


class _FakeTokenizer:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        return self.mapping.get(text, list(range(len(text))))


def test_counts_with_local_tokenizer() -> None:
    tokenizer = _FakeTokenizer({"abcdef": [1, 2, 3]})
    counter = LocalTextTokenCounter(tokenizer=tokenizer)
    assert counter("abcdef") == 3
    # 必须显式关闭 special tokens：vLLM /tokenize 也是这个口径。
    assert tokenizer.calls == [("abcdef", False)]


def test_non_string_input_is_stringified() -> None:
    counter = LocalTextTokenCounter(tokenizer=_FakeTokenizer())
    assert counter(12345) == 5


def test_requires_tokenizer_or_path() -> None:
    with pytest.raises(ValueError, match="tokenizer_path"):
        LocalTextTokenCounter()
