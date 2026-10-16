"""CPU-only checks for the entropy-memory preflight guard.

Deliberately hydra-free: it only builds tiny namespaces, so it runs on Windows
where the composed-config tests cannot.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.check_grpo_runtime import validate_entropy_memory_guard


def _actor(calculate_entropy: bool, chunking: bool) -> SimpleNamespace:
    return SimpleNamespace(
        calculate_entropy=calculate_entropy,
        fsdp_config=SimpleNamespace(entropy_from_logits_with_chunking=chunking),
    )


class EntropyMemoryGuardTest(unittest.TestCase):
    def test_entropy_disabled_needs_no_chunking(self):
        validate_entropy_memory_guard(_actor(False, False))

    def test_entropy_enabled_with_chunking_passes(self):
        validate_entropy_memory_guard(_actor(True, True))

    def test_entropy_enabled_without_chunking_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "bounded entropy computation"):
            validate_entropy_memory_guard(_actor(True, False))

    def test_rejection_names_the_effective_config_path(self):
        with self.assertRaisesRegex(SystemExit, "fsdp_config"):
            validate_entropy_memory_guard(_actor(True, False))

    def test_rejection_warns_the_actor_level_flag_is_ignored(self):
        with self.assertRaisesRegex(SystemExit, "is ignored by"):
            validate_entropy_memory_guard(_actor(True, False))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
