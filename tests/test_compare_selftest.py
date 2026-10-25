"""`scripts/compare_selftest.py` 的单测：判定分叉是预登记的，阈值不许漂。"""

from __future__ import annotations

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_selftest", ROOT / "scripts" / "compare_selftest.py"
)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def _record(task_id: int, *, success: bool, reward_type: str | None = None) -> dict:
    detail = {"purchase_success": success}
    if reward_type is not None:
        detail["reward_type"] = reward_type
    return {"task_id": task_id, "terminal_result": {"reward_detail": detail}}


def test_is_success_reads_purchase_success() -> None:
    assert compare.is_success(_record(1, success=True)) is True
    assert compare.is_success(_record(1, success=False)) is False


def test_is_success_falls_back_to_gold_purchase_type() -> None:
    record = {"task_id": 1, "terminal_result": {"reward_detail": {"reward_type": "gold_purchase"}}}
    assert compare.is_success(record) is True
    other = {"task_id": 1, "terminal_result": {"reward_detail": {"reward_type": "wrong_purchase"}}}
    assert compare.is_success(other) is False


def test_is_success_survives_missing_terminal_result() -> None:
    assert compare.is_success({"task_id": 1}) is False


def test_success_ids_keys_by_task() -> None:
    table = compare.success_ids([_record(3, success=True), _record(4, success=False)])
    assert table == {3: True, 4: False}


def test_branch_verdict_thresholds_are_fixed() -> None:
    assert compare.branch_verdict(0.90)[0] == "泛化问题"
    assert compare.branch_verdict(0.95)[0] == "泛化问题"
    assert compare.branch_verdict(0.70)[0] == "没学会"
    assert compare.branch_verdict(0.55)[0] == "没学会"
    assert compare.branch_verdict(0.80)[0] == "灰区"


def test_wilson_interval_brackets_the_point_estimate() -> None:
    low, high = compare.wilson_interval(180, 300)
    assert low < 0.6 < high
    assert 0.54 < low < 0.55
    assert 0.65 < high < 0.66


def test_wilson_interval_handles_extremes() -> None:
    assert compare.wilson_interval(0, 0) == (0.0, 0.0)
    low, high = compare.wilson_interval(0, 300)
    assert low == 0.0
    assert 0.0 < high < 0.02
    low, high = compare.wilson_interval(300, 300)
    assert 0.98 < low < 1.0
    assert abs(high - 1.0) < 1e-9


def test_format_rate_includes_ci() -> None:
    text = compare.format_rate(180, 300)
    assert text.startswith("180/300 = 60.0%")
    assert "95% CI" in text
    assert compare.format_rate(0, 0) == "—"


def test_load_unsolvable_accepts_both_shapes(tmp_path: pathlib.Path) -> None:
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"task_ids": [1, 2]}), encoding="utf-8")
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps([3, 4]), encoding="utf-8")

    assert compare.load_unsolvable(wrapped) == {1, 2}
    assert compare.load_unsolvable(plain) == {3, 4}
    assert compare.load_unsolvable(tmp_path / "missing.json") == set()


def test_main_reports_trained_rate(tmp_path: pathlib.Path) -> None:
    run_dir = tmp_path / "selftest"
    run_dir.mkdir()
    rows = [_record(task_id, success=task_id < 27) for task_id in range(30)]
    (run_dir / "trajectories.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    exit_code = compare.main(["--run-dir", str(run_dir)])
    assert exit_code == 0
