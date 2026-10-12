"""Pre-flight data-contract smoke test (rule P8).

Runs ``scripts/smoke_data_contracts.py`` and requires a clean exit, so the
rebuilt pools can never silently drift out of the shape that the SFT / veRL /
curriculum pipelines consume.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DataContractSmokeTest(unittest.TestCase):
    def test_smoke_data_contracts_passes(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "smoke_data_contracts.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("数据契约冒烟通过", proc.stdout)


if __name__ == "__main__":
    unittest.main()
