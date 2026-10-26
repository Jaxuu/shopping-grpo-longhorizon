"""`scripts/compare_teacher_vs_ours.py` 的单测。

要点：差集必须只认严格成功；分母不同必须拒绝对比（老师历史 139/200 是旧卷子，
拿它跟 Final-200 Clean 的 127/200 直接相减就是错的）；开候选数要数不同 asin。
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_teacher_vs_ours", ROOT / "scripts" / "compare_teacher_vs_ours.py"
)
cmp_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cmp_mod)


def _summary(task_ids, *, expected=200, successes=None) -> dict:
    return {
        "expected_tasks": expected,
        "strict_successes": len(task_ids) if successes is None else successes,
        "strict_success_task_ids": list(task_ids),
    }


def _trajectory(task_id, *, opens=(), searches=1, paged=0, reward_type="wrong_purchase"):
    steps = []
    for _ in range(searches):
        steps.append({"tool_name": "search_products", "parameters": {"query": "木梳"}})
    for asin in opens:
        steps.append({"tool_name": "open_product", "parameters": {"asin": asin}})
    for _ in range(paged):
        steps.append({"tool_name": "next_page", "parameters": {}})
    steps.append({"tool_name": "buy_now", "parameters": {"asin": "x"}})
    return {
        "task_id": task_id,
        "done": True,
        "steps": steps,
        "terminal_result": {"reward_detail": {"reward_type": reward_type}},
    }


def test_success_ids_rejects_missing_field() -> None:
    with pytest.raises(ValueError, match="strict_success_task_ids"):
        cmp_mod.success_ids({"expected_tasks": 200})


def test_compare_rejects_different_denominators() -> None:
    with pytest.raises(ValueError, match="分母不同"):
        cmp_mod.compare(_summary([1], expected=200), _summary([1], expected=183))


def test_compare_builds_disjoint_sets() -> None:
    report = cmp_mod.compare(_summary([1, 2, 3]), _summary([2, 3, 4]))
    assert report["both"] == [2, 3]
    assert report["teacher_only"] == [1]
    assert report["ours_only"] == [4]
    assert report["net_gap"] == 0
    assert report["teacher_successes"] == 3
    assert report["ours_successes"] == 3


def test_behaviour_counts_distinct_products() -> None:
    # 同一个 asin 打开两次只算 1 个候选；这才是"探索广度"。
    record = _trajectory(7, opens=("A", "A", "B"), searches=2, paged=1)
    result = cmp_mod.behaviour(record)
    assert result["opened_products"] == 2
    assert result["searches"] == 2
    assert result["paged"] == 1
    assert result["bought"] is True
    # 2 次搜索 + 3 次打开 + 1 次翻页 + 1 次购买
    assert result["steps"] == 7
    assert result["reward_type"] == "wrong_purchase"


def test_behaviour_tolerates_missing_trajectory() -> None:
    assert cmp_mod.behaviour(None) == {}


def test_compare_aggregates_gap_behaviour() -> None:
    teacher_traj = {1: _trajectory(1, opens=("A", "B", "C"), searches=2, reward_type="gold_purchase")}
    ours_traj = {1: _trajectory(1, opens=("A",), searches=1, reward_type="wrong_purchase")}
    report = cmp_mod.compare(
        _summary([1]),
        _summary([]),
        teacher_trajectories=teacher_traj,
        ours_trajectories=ours_traj,
    )
    assert report["teacher_only"] == [1]
    assert report["teacher_only_teacher_behaviour"]["mean_opened_products"] == 3.0
    assert report["teacher_only_ours_behaviour"]["mean_opened_products"] == 1.0
    assert report["teacher_only_ours_reward_types"] == {"wrong_purchase": 1}


def test_render_markdown_contains_key_numbers() -> None:
    report = cmp_mod.compare(
        _summary([1, 2]),
        _summary([2]),
        teacher_trajectories={1: _trajectory(1, opens=("A", "B"))},
        ours_trajectories={1: _trajectory(1, opens=("A",))},
    )
    text = cmp_mod.render_markdown(report)
    assert "只有 teacher 对（我们的缺口）" in text
    assert "| 1 |" in text
    assert "平均打开不同候选" in text


def test_main_writes_json_and_markdown(tmp_path) -> None:
    teacher_dir = tmp_path / "teacher"
    ours_dir = tmp_path / "ours"
    teacher_dir.mkdir()
    ours_dir.mkdir()
    (teacher_dir / "summary.json").write_text(
        json.dumps(_summary([1, 2]), ensure_ascii=False), encoding="utf-8"
    )
    (ours_dir / "summary.json").write_text(
        json.dumps(_summary([2]), ensure_ascii=False), encoding="utf-8"
    )
    (teacher_dir / "trajectories.jsonl").write_text(
        json.dumps(_trajectory(1, opens=("A", "B")), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (ours_dir / "trajectories.jsonl").write_text(
        json.dumps(_trajectory(1, opens=("A",)), ensure_ascii=False) + "\n", encoding="utf-8"
    )

    code = cmp_mod.main(
        [
            "--teacher-summary",
            str(teacher_dir / "summary.json"),
            "--ours-summary",
            str(ours_dir / "summary.json"),
            "--teacher-name",
            "老师",
            "--ours-name",
            "冠军",
        ]
    )
    assert code == 0
    payload = json.loads((teacher_dir / "teacher-vs-ours.json").read_text(encoding="utf-8"))
    assert payload["teacher_only"] == [1]
    assert "老师 vs 冠军" in (teacher_dir / "teacher-vs-ours.md").read_text(encoding="utf-8")
