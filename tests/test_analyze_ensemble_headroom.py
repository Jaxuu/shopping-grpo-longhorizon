"""`scripts/analyze_ensemble_headroom.py` 的单测：上界算错就会把结论带偏。"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_ensemble_headroom", ROOT / "scripts" / "analyze_ensemble_headroom.py"
)
hr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hr)


def test_ranked_sorts_by_votes_then_key() -> None:
    assert hr.ranked({"B": 2, "A": 2, "C": 5}) == [("C", 5), ("A", 2), ("B", 2)]


def test_top_k_respects_k() -> None:
    votes = {"A": 5, "B": 3, "C": 1}
    assert hr.top_k(votes, 2) == ["A", "B"]
    assert hr.top_k(votes, 9) == ["A", "B", "C"]


def test_oracle_top_2_counts_only_correct_in_top2() -> None:
    votes = {1: {"W": 5, "G": 2, "X": 1}, 2: {"W": 5, "X": 3}, 3: {"G": 4, "W": 1}}
    good = {1: {"G"}, 2: {"G"}, 3: {"G"}}
    # 题 1：G 在第 2 名 → 命中；题 2：G 没被买到 → 不命中；题 3：G 第 1 名 → 命中
    assert hr.oracle_top_k(votes, good, 2) == 2


def test_oracle_top_k_is_monotone_in_k() -> None:
    votes = {1: {"W": 5, "X": 3, "Y": 2, "G": 1}}
    good = {1: {"G"}}
    assert hr.oracle_top_k(votes, good, 2) == 0
    assert hr.oracle_top_k(votes, good, 3) == 0
    assert hr.oracle_top_k(votes, good, 4) == 1


def test_plurality_hit_tie_is_failure_without_tiebreak() -> None:
    votes = {1: {"G": 2, "W": 2}}
    good = {1: {"G"}}
    assert hr.plurality_hit(votes, good) == 0
    assert hr.plurality_hit(votes, good, tie_break={1: "G"}) == 1


def test_champion_weighted_flips_minority() -> None:
    votes = {1: {"W": 5, "G": 4}}
    good = {1: {"G"}}
    assert hr.plurality_hit(votes, good) == 0
    # 冠军选 G，×2 后 G=5 与 W=5 平票 → 仍不算命中
    assert hr.champion_weighted(votes, {1: "G"}, good, weight=2) == 0
    # ×3 后 G=6 > W=5 → 命中
    assert hr.champion_weighted(votes, {1: "G"}, good, weight=3) == 1


def test_champion_weighted_weight_one_equals_plurality() -> None:
    votes = {1: {"W": 3, "G": 1}, 2: {"G": 2, "W": 1}}
    good = {1: {"G"}, 2: {"G"}}
    assert hr.champion_weighted(votes, {1: "W", 2: "G"}, good, weight=1) == 1


def test_diagnose_reports_margins_and_champion() -> None:
    votes = {
        1: {"W": 5, "G": 3},  # 差 2 票，冠军选 W（错）
        2: {"G": 4, "W": 1},  # 命中，不进错题表
        3: {"W": 2, "X": 2},  # 平票，G 没人买到
    }
    good = {1: {"G"}, 2: {"G"}, 3: {"G"}}
    diag = hr.diagnose(votes, good, {1: "W", 2: "G"})
    assert diag["margins"] == {2: 1}
    assert diag["ties"] == 1
    assert diag["champion_right"] == 0
    assert diag["champion_wrong"] == 1
    assert diag["champion_silent"] == 1
    assert diag["correct_in_top2"] == 1  # 题 1 的 G 是第 2 名
    assert len(diag["wrong_tasks"]) == 2


def test_diagnose_counts_correct_in_top2() -> None:
    votes = {1: {"W": 6, "X": 2, "G": 1}}
    good = {1: {"G"}}
    diag = hr.diagnose(votes, good, {})
    assert diag["correct_in_top2"] == 0
    assert diag["correct_in_top3"] == 1
    assert diag["champion_silent"] == 1
