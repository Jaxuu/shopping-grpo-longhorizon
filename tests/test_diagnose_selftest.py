"""`scripts/diagnose_selftest.py` 的单测：分桶边界与标准化算术。"""

from __future__ import annotations

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_selftest", ROOT / "scripts" / "diagnose_selftest.py"
)
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


def test_bucket_boundaries_follow_page_size() -> None:
    assert diag.bucket_of(1) == "1-5"
    assert diag.bucket_of(5) == "1-5"
    assert diag.bucket_of(6) == "6-20"
    assert diag.bucket_of(20) == "6-20"
    assert diag.bucket_of(21) == "21-60"
    assert diag.bucket_of(60) == "21-60"
    assert diag.bucket_of(61) == ">60/搜不到"
    assert diag.bucket_of(None) == ">60/搜不到"


def test_bucket_stats_counts_per_bucket() -> None:
    run = {
        1: {"success": True, "reward_type": "gold_purchase"},
        2: {"success": False, "reward_type": "wrong_purchase"},
        3: {"success": True, "reward_type": "gold_purchase"},
    }
    ranks = {1: "1-5", 2: "1-5", 3: "6-20"}

    stats = diag.bucket_stats(run, ranks)

    assert stats["1-5"]["n"] == 2
    assert stats["1-5"]["ok"] == 1
    assert stats["1-5"]["rate"] == 0.5
    assert stats["6-20"]["rate"] == 1.0
    assert stats["21-60"]["n"] == 0
    assert stats["21-60"]["rate"] is None


def test_standardize_reweights_by_other_distribution() -> None:
    source = {
        "1-5": {"n": 10, "ok": 9, "rate": 0.9},
        "21-60": {"n": 10, "ok": 1, "rate": 0.1},
        "6-20": {"n": 0, "ok": 0, "rate": None},
        ">60/搜不到": {"n": 0, "ok": 0, "rate": None},
    }
    weights = {
        "1-5": {"n": 10, "ok": 0, "rate": 0.0},
        "21-60": {"n": 30, "ok": 0, "rate": 0.0},
        "6-20": {"n": 0, "ok": 0, "rate": None},
        ">60/搜不到": {"n": 0, "ok": 0, "rate": None},
    }

    # 10 道落在 1-5（0.9）+ 30 道落在 21-60（0.1）→ (9 + 3) / 40 = 0.3
    assert abs(diag.standardize(source, weights) - 0.3) < 1e-9


def test_standardize_returns_none_without_weights() -> None:
    empty = {name: {"n": 0, "ok": 0, "rate": None} for name in diag.BUCKETS}
    assert diag.standardize(empty, empty) is None


def test_standardize_skips_buckets_missing_in_source() -> None:
    source = {"1-5": {"n": 5, "ok": 5, "rate": 1.0}}
    weights = {"1-5": {"n": 5, "ok": 0, "rate": 0.0}, "21-60": {"n": 5, "ok": 0, "rate": 0.0}}
    # 21-60 在 source 里没有 rate ⇒ 不参与加权，只用 1-5 的 5 道
    assert diag.standardize(source, weights) == 1.0


def test_load_run_reads_purchase_success(tmp_path: pathlib.Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {"task_id": 7, "terminal_result": {"reward_detail": {"purchase_success": True, "reward_type": "gold_purchase"}}},
        {"task_id": 8, "terminal_result": {"reward_detail": {"purchase_success": False, "reward_type": "repeat_loop"}}},
        {"task_id": 9},
    ]
    (run_dir / "trajectories.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    table = diag.load_run(run_dir)

    assert table[7]["success"] is True
    assert table[8]["reward_type"] == "repeat_loop"
    assert table[9]["success"] is False
    assert table[9]["reward_type"] == "unknown"


def test_fmt_rate_handles_zero() -> None:
    assert diag.fmt_rate(0, 0) == "—"
    assert diag.fmt_rate(3, 4) == "3/4 = 75.0%"
