from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.probe_group_diversity import (
    build_difficulty_labels,
    parse_filter_band,
    write_filter_outputs,
)

ROOT = Path(__file__).resolve().parents[1]


def trajectory(task_id: int, attempt: int, reward: float, reward_type: str) -> dict:
    return {
        "task_id": task_id,
        "attempt_index": attempt,
        "final_reward": reward,
        "steps": [],
        "status": "done",
        "terminal_result": {
            "reward_type": reward_type,
            "reward_valid": True,
            "purchase": None,
        },
    }


def test_parse_filter_band_validates_range_and_format():
    assert parse_filter_band("0.125,0.875") == (0.125, 0.875)
    assert parse_filter_band(None) is None
    with pytest.raises(SystemExit):
        parse_filter_band("0.8")
    with pytest.raises(SystemExit):
        parse_filter_band("-0.1,0.9")


def test_difficulty_labels_select_only_valid_reward_varying_groups():
    grouped = {
        1: [
            trajectory(1, 0, 1.0, "gold_purchase"),
            trajectory(1, 1, 0.2, "partial_alternative_purchase"),
        ],
        2: [
            trajectory(2, 0, 1.0, "gold_purchase"),
            trajectory(2, 1, 1.0, "gold_purchase"),
        ],
        3: [
            trajectory(3, 0, 1.0, "gold_purchase"),
            trajectory(3, 1, 0.0, "reward_unverifiable"),
        ],
        4: [trajectory(4, 0, 1.0, "gold_purchase")],
    }

    labels = build_difficulty_labels(
        grouped,
        [1, 2, 3, 4],
        filter_n=2,
        tolerance=1e-8,
    )
    assert [row["eligible"] for row in labels] == [True, False, False, False]
    assert labels[0]["pass_rate"] == pytest.approx(0.5)
    assert labels[1]["filter_reason"] == "constant_reward"
    assert labels[2]["filter_reason"] == "sampling_invalid"
    assert labels[3]["filter_reason"] == "insufficient_attempts"


def test_pass_rate_band_is_an_additional_filter():
    grouped = {
        1: [trajectory(1, 0, 1.0, "gold_purchase"), trajectory(1, 1, -0.8, "wrong_purchase")],
        2: [trajectory(2, 0, 1.0, "gold_purchase"), trajectory(2, 1, 0.2, "valid_alternative_purchase")],
    }
    labels = build_difficulty_labels(
        grouped,
        [1, 2],
        filter_n=2,
        tolerance=1e-8,
        pass_rate_band=(0.5, 0.5),
    )
    assert labels[0]["eligible"] is True
    assert labels[0]["pass_rate"] == pytest.approx(0.5)
    assert labels[1]["eligible"] is False
    assert labels[1]["pass_rate"] == pytest.approx(1.0)
    assert labels[1]["filter_reason"] == "outside_pass_rate_band"


def test_filter_outputs_write_jsonl_and_refuse_overwriting_source(tmp_path):
    labels = [
        {"task_id": 1, "eligible": True},
        {"task_id": 2, "eligible": False},
    ]
    selected = write_filter_outputs(
        tmp_path / "probe",
        labels,
        train_parquet=tmp_path / "source.parquet",
    )
    assert selected == [1]
    tasks = (tmp_path / "probe" / "filtered_tasks.jsonl").read_text(encoding="utf-8")
    assert json.loads(tasks) == {"task_id": 1}
    label_lines = (tmp_path / "probe" / "difficulty_labels.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(label_lines) == 2

    with pytest.raises(SystemExit, match="拒绝覆盖"):
        write_filter_outputs(
            tmp_path / "probe2",
            labels,
            train_parquet=tmp_path / "same.parquet",
            parquet_out=tmp_path / "same.parquet",
        )


def test_filter_outputs_refuse_empty_selection(tmp_path):
    with pytest.raises(SystemExit, match="没有保留任何任务"):
        write_filter_outputs(
            tmp_path,
            [{"task_id": 1, "eligible": False}],
            train_parquet=tmp_path / "source.parquet",
        )


def test_cli_report_only_filter_end_to_end(tmp_path):
    """锁定 CLI 接线：--report-only + 过滤 + parquet 副本（不需要任何服务）。"""
    plan = {
        101: [(1.0, "gold_purchase"), (0.2, "partial_alternative_purchase")] * 2,
        102: [(1.0, "gold_purchase")] * 4,
        103: [(-0.8, "wrong_purchase")] * 4,
    }
    with (tmp_path / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task_id in plan:
            handle.write(json.dumps({"task_id": task_id}) + "\n")
    with (tmp_path / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for task_id, attempts in plan.items():
            for index, (reward, reward_type) in enumerate(attempts):
                handle.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "attempt_index": index,
                            "final_reward": reward,
                            "status": "done",
                            "steps": [],
                            "blocked_tool_calls": [],
                            "tool_call_truncations": [],
                            "terminal_result": {
                                "reward_type": reward_type,
                                "reward_valid": True,
                                "purchase": None,
                            },
                        }
                    )
                    + "\n"
                )

    command = [
        sys.executable,
        str(ROOT / "scripts" / "probe_group_diversity.py"),
        "--tasks",
        str(tmp_path / "tasks.jsonl"),
        "--out-dir",
        str(tmp_path),
        "--report-only",
        "--compare-n",
        "4",
        "--filter-training-pool",
        "--filter-n",
        "4",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    labels = [
        json.loads(line)
        for line in (tmp_path / "difficulty_labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["task_id"] for row in labels if row["eligible"]] == [101]
    assert {row["task_id"]: row["filter_reason"] for row in labels}[102] == "constant_reward"
    kept = (tmp_path / "filtered_tasks.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in kept] == [{"task_id": 101}]
    assert "M6 重训练任务过滤" in (tmp_path / "report.md").read_text(encoding="utf-8")
