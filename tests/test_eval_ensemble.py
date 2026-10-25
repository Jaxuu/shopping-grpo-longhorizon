"""`scripts/eval_ensemble.py` 的单测：多数票与平票规则是全部结论的依据。"""

from __future__ import annotations

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "eval_ensemble", ROOT / "scripts" / "eval_ensemble.py"
)
ens = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ens)


def _record(
    task_id: int,
    asin: str | None,
    *,
    reward_type: str | None = None,
    purchase_success: bool = False,
) -> dict:
    return {
        "task_id": task_id,
        "terminal_result": {
            "purchase": {"asin": asin} if asin else {},
            "reward_detail": {
                "reward_type": reward_type,
                "purchase_success": purchase_success,
            },
        },
    }


def test_plurality_picks_most_common() -> None:
    winner, tied = ens.plurality({"A": 3, "B": 1})
    assert winner == "A"
    assert tied is False


def test_plurality_flags_tie() -> None:
    winner, tied = ens.plurality({"A": 2, "B": 2})
    assert winner is None
    assert tied is True


def test_plurality_handles_empty() -> None:
    assert ens.plurality({}) == (None, False)


def test_vote_summary_counts_hits_and_ties() -> None:
    votes = {
        1: {"GOLD": 3, "X": 1},  # 命中
        2: {"GOLD": 2, "X": 2},  # 平票
        3: {"X": 2, "Y": 1},  # 未命中
    }
    good = {1: {"GOLD"}, 2: {"GOLD"}, 3: {"GOLD"}}

    stats = ens.vote_summary(votes, good)

    assert stats["hit"] == 1
    assert stats["tie"] == 1
    assert stats["n"] == 3
    assert stats["hit_after_break"] == 1


def test_vote_summary_tie_break_can_rescue() -> None:
    votes = {2: {"GOLD": 2, "X": 2}}
    good = {2: {"GOLD"}}
    stats = ens.vote_summary(votes, good, tie_break={2: "GOLD"})
    assert stats["hit"] == 0
    assert stats["tie"] == 1
    assert stats["hit_after_break"] == 1


def test_vote_summary_tie_break_that_is_not_good_does_not_rescue() -> None:
    """兜底票落在「没人成功过的身份」上时不能算命中，否则会突破并集。"""
    votes = {2: {"X": 2, "Y": 2}}
    good = {2: {"GOLD"}}
    stats = ens.vote_summary(votes, good, tie_break={2: "X"})
    assert stats["hit_after_break"] == 0


def test_vote_summary_skips_tasks_without_good_keys() -> None:
    stats = ens.vote_summary({9: {"A": 1}}, {})
    assert stats["n"] == 1
    assert stats["hit"] == 0


def test_vote_summary_never_exceeds_union() -> None:
    """命中 ⊆ 并集：这是防止「投票比天花板还高」的硬约束。"""
    votes = {
        1: {"GOLD": 3, "PARTIAL": 2},
        2: {"PARTIAL": 3},
        3: {"A": 1, "B": 1},
    }
    good = {1: {"GOLD"}, 2: {"GOLD"}, 3: set()}
    stats = ens.vote_summary(votes, good, tie_break={1: "GOLD", 2: "PARTIAL", 3: "A"})
    union = sum(1 for keys in good.values() if keys)
    assert stats["hit_after_break"] <= union


def test_purchased_and_success_read_terminal_result() -> None:
    record = _record(1, "ABC", reward_type="gold_purchase", purchase_success=True)
    assert ens.purchased(record) == "ABC"
    assert ens.is_success(record) is True
    assert ens.purchased({"terminal_result": {"purchase": {}}}) is None
    assert ens.is_success({}) is False


def test_is_success_strict_rejects_valid_alternative() -> None:
    """`valid_alternative_purchase` 的 purchase_success 也是 True，但 strict 口径不认它。

    历史上 compare_selftest 报 179、strict 报 178，差的就是这一条。
    """
    record = _record(1, "ABC", reward_type="valid_alternative_purchase", purchase_success=True)
    assert ens.is_success(record) is False
    assert ens.is_success(record, strict=False) is True


def test_purchased_distinguishes_options() -> None:
    """同一 ASIN 不同规格必须是两个身份：买对商品选错规格是 partial，不是成功。"""
    record = {
        "terminal_result": {
            "purchase": {"asin": "ABC", "options": {"color": "red", "size": "M"}},
        }
    }
    assert ens.purchased(record) == "ABC|color=red&size=M"


def test_purchased_without_options_is_plain_asin() -> None:
    record = {"terminal_result": {"purchase": {"asin": "ABC", "options": {}}}}
    assert ens.purchased(record) == "ABC"


def test_is_success_falls_back_to_reward_type() -> None:
    record = {"terminal_result": {"reward_detail": {"reward_type": "gold_purchase"}}}
    assert ens.is_success(record) is True


def test_resolve_runs_skips_small_dirs(tmp_path: pathlib.Path) -> None:
    big = tmp_path / "big"
    small = tmp_path / "small"
    for folder, count in ((big, 5), (small, 1)):
        folder.mkdir()
        rows = [json.dumps(_record(i, "A")) + "\n" for i in range(count)]
        (folder / "trajectories.jsonl").write_text("".join(rows), encoding="utf-8")

    runs = ens.resolve_runs([big, small], min_tasks=3)

    assert list(runs) == ["big"]
    assert len(runs["big"]) == 5


def test_resolve_runs_accepts_file_paths(tmp_path: pathlib.Path) -> None:
    folder = tmp_path / "one"
    folder.mkdir()
    path = folder / "trajectories.jsonl"
    path.write_text(json.dumps(_record(1, "A")) + "\n", encoding="utf-8")
    runs = ens.resolve_runs([path], min_tasks=1)
    assert list(runs) == ["one"]


def test_majority_beats_every_single_voter_when_errors_are_diverse() -> None:
    """三个模型各错不同题，gold 总是多数票 —— 这就是集成的原理。"""
    good = {1: {"GOLD"}, 2: {"GOLD"}, 3: {"GOLD"}}
    votes = {
        1: {"GOLD": 2, "X": 1},
        2: {"GOLD": 2, "Y": 1},
        3: {"GOLD": 2, "Z": 1},
    }
    stats = ens.vote_summary(votes, good)
    assert stats["hit"] == 3
