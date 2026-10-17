"""CPU-only checks for the entropy preflight guard.

Deliberately hydra-free: it only builds tiny namespaces, so it runs on Windows
where the composed-config tests cannot.

**2026-10-17 行为变更**：守卫原先只拒绝「开了熵但没开
`entropy_from_logits_with_chunking`」。M6-R 第 6 步实测 OOM（`loss.backward()`
尝试分配 12.47 GiB）证明分块救不了 autograd 保存的那份整块
`softmax(logits)`（`[tokens, vocab]` bf16 ≈12.2 GiB）。现在只要开了熵就拒绝。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.check_grpo_runtime import validate_entropy_memory_guard


def _actor(calculate_entropy: bool, chunking: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        calculate_entropy=calculate_entropy,
        fsdp_config=SimpleNamespace(entropy_from_logits_with_chunking=chunking),
    )


class EntropyMemoryGuardTest(unittest.TestCase):
    def test_entropy_disabled_passes(self):
        validate_entropy_memory_guard(_actor(False))

    def test_entropy_disabled_passes_even_with_chunking(self):
        validate_entropy_memory_guard(_actor(False, True))

    def test_entropy_enabled_is_rejected(self):
        with self.assertRaises(SystemExit):
            validate_entropy_memory_guard(_actor(True))

    def test_chunking_no_longer_rescues_entropy(self):
        """分块只约束前向 softmax，压不住反向的 [tokens, vocab] 驻留 ⇒ 仍然拒绝。"""
        with self.assertRaises(SystemExit):
            validate_entropy_memory_guard(_actor(True, True))

    def test_rejection_explains_the_missing_metric(self):
        with self.assertRaisesRegex(SystemExit, "bypass_mode"):
            validate_entropy_memory_guard(_actor(True))

    def test_rejection_explains_the_backward_oom(self):
        with self.assertRaisesRegex(SystemExit, "softmax"):
            validate_entropy_memory_guard(_actor(True))

    def test_rejection_names_the_fix(self):
        with self.assertRaisesRegex(SystemExit, "calculate_entropy=false"):
            validate_entropy_memory_guard(_actor(True))

    def test_actor_without_fsdp_config_does_not_crash(self):
        """守卫本身不能因为字段缺失而炸（只影响报错文案里的 chunking 值）。"""
        with self.assertRaises(SystemExit):
            validate_entropy_memory_guard(SimpleNamespace(calculate_entropy=True))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
