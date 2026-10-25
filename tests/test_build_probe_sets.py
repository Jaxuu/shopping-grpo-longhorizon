"""`scripts/build_probe_sets.py` 的单测：分层抽样与零重叠是尺子的命门。"""

from __future__ import annotations

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_probe_sets", ROOT / "scripts" / "build_probe_sets.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _goal(price=None, brand=None, model=None, options=None) -> dict:
    return {
        "price_upper": price,
        "expected_brand": brand or [],
        "expected_model": model or [],
        "required_options_by_key": options or {},
    }


def test_hard_axes_counts_each_axis() -> None:
    assert probe.hard_axes(_goal()) == 0
    assert probe.hard_axes(_goal(price=20.0)) == 1
    assert probe.hard_axes(_goal(price="22.0", brand=["Acme"])) == 2
    assert probe.hard_axes(_goal(price="22.0", brand=["A"], model=["M"])) == 3
    assert (
        probe.hard_axes(_goal(price="22.0", brand=["A"], model=["M"], options={"c": 1}))
        == 4
    )


def test_hard_axes_treats_blank_price_as_absent() -> None:
    assert probe.hard_axes(_goal(price="")) == 0
    assert probe.hard_axes(_goal(price=None)) == 0


def test_largest_remainder_sums_to_total() -> None:
    allocation = probe.largest_remainder({1: 10, 2: 163, 3: 25, 4: 2}, 300)
    assert sum(allocation.values()) == 300
    assert set(allocation) == {1, 2, 3, 4}


def test_largest_remainder_is_proportional() -> None:
    allocation = probe.largest_remainder({1: 1, 2: 1}, 300)
    assert allocation == {1: 150, 2: 150}
    skewed = probe.largest_remainder({1: 9, 2: 1}, 300)
    assert skewed == {1: 270, 2: 30}


def test_largest_remainder_ignores_empty_strata() -> None:
    assert probe.largest_remainder({1: 0, 2: 5}, 10) == {2: 10}
    assert probe.largest_remainder({}, 10) == {}


def test_select_dev_hits_size_and_respects_reference() -> None:
    goals = [_goal(price=1.0)] * 50 + [_goal(price=1.0, brand=["A"])] * 50
    candidates = list(range(100))
    reference = {1: 40, 2: 60}

    chosen, targets, actual = probe.select_dev(candidates, goals, reference, 20, seed=7)

    assert len(chosen) == 20
    assert chosen == sorted(set(chosen))
    assert targets == {1: 8, 2: 12}
    assert actual == {1: 8, 2: 12}


def test_select_dev_is_deterministic_for_a_seed() -> None:
    goals = [_goal(price=1.0)] * 100
    first, _, _ = probe.select_dev(range(100), goals, {1: 10}, 30, seed=2026)
    second, _, _ = probe.select_dev(range(100), goals, {1: 10}, 30, seed=2026)
    other, _, _ = probe.select_dev(range(100), goals, {1: 10}, 30, seed=2027)

    assert first == second
    assert first != other


def test_select_dev_backfills_when_a_stratum_is_short() -> None:
    # 参照分布要求 90% 落在轴 2，但候选里只有 3 道轴 2 的题。
    goals = [_goal(price=1.0, brand=["A"])] * 3 + [_goal(price=1.0)] * 97
    chosen, targets, actual = probe.select_dev(range(100), goals, {2: 90, 1: 10}, 20, seed=1)

    assert len(chosen) == 20
    assert actual[2] == 3
    assert targets[2] == 18
    assert sum(actual.values()) == 20


def test_collect_pools_refuses_missing_files(tmp_path: pathlib.Path) -> None:
    present = tmp_path / "present.jsonl"
    present.write_text('{"task_id": 1}\n', encoding="utf-8")
    specs = (("present", present), ("absent", tmp_path / "absent.jsonl"))

    pools, missing = probe.collect_pools(specs, allow_missing=True)
    assert list(pools) == ["present"]
    assert pools["present"] == [1]
    assert len(missing) == 1

    try:
        probe.collect_pools(specs, allow_missing=False)
    except SystemExit as error:
        assert "absent" in str(error)
    else:  # pragma: no cover - 缺池必须报错
        raise AssertionError("缺池时应当 SystemExit")


def test_used_ids_unions_every_pool() -> None:
    pools = {"a": [1, 2], "b": [2, 3]}
    assert probe.used_ids(pools) == {1, 2, 3}


def test_sha256_lf_is_line_ending_stable(tmp_path: pathlib.Path) -> None:
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    lf.write_bytes(b'{"task_id": 1}\n')
    crlf.write_bytes(b'{"task_id": 1}\r\n')
    assert probe.sha256_lf(lf) == probe.sha256_lf(crlf)


def test_admit_all_counts_rejections(tmp_path: pathlib.Path, monkeypatch) -> None:
    def fake_admit(task_id, goal, product, *, held_out_task_ids=()):
        if int(task_id) % 2:
            return {"admitted": True, "reasons": []}
        return {"admitted": False, "reasons": ["A3_zero_price"]}

    monkeypatch.setattr(probe, "admit_task", fake_admit)
    goals = [{"asin": "A"}] * 6
    products = {"A": {"asin": "A"}}

    admitted, rejected = probe.admit_all(range(6), goals, products, set())

    assert admitted == [1, 3, 5]
    assert rejected["A3_zero_price"] == 3


def test_admit_all_flags_missing_product() -> None:
    admitted, rejected = probe.admit_all([0], [{"asin": "ZZZ"}], {}, set())
    assert admitted == []
    assert rejected["no_product"] == 1


def test_write_jsonl_is_sorted_and_deterministic(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "tasks.jsonl"
    probe.write_jsonl(path, [{"task_id": 2}, {"task_id": 1}])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"task_id": 2}, {"task_id": 1}]
    assert lines[0] == '{"task_id": 2}'


class _Hit:
    def __init__(self, asin: str, rank: int) -> None:
        self.asin = asin
        self.rank = rank


class _FakeSearcher:
    """把 task_id 换成 asin：blocked 里的题在前 150 名内永远搜不到。"""

    def __init__(self, blocked: set[int], total: int = 50) -> None:
        self.blocked = blocked
        self.total = total

    def search(self, query: str, k: int):
        hits = []
        rank = 1
        for task_id in range(self.total):
            if task_id in self.blocked:
                continue
            hits.append(_Hit(f"P{task_id}", rank))
            rank += 1
        return hits[:k]


def _plain_goal(task_id: int) -> dict:
    return {
        "asin": f"P{task_id}",
        "instruction_text": "买一个东西",
        "category": "家居›收纳",
        "attributes": ["大号"],
    }


def test_select_reachable_dev_replaces_unreachable_tasks() -> None:
    goals = [_plain_goal(task_id) for task_id in range(50)]
    blocked = {3, 7, 11, 19, 23}
    searcher = _FakeSearcher(blocked)

    chosen, _, _, ranks, dropped = probe.select_reachable_dev(
        list(range(50)), goals, {0: 10}, 20, seed=5, searcher=searcher
    )

    assert len(chosen) == 20
    assert not set(chosen) & blocked
    # 只有被抽中过、且确实搜不到的那几道才算「换掉」，所以 0 < dropped <= 5。
    assert 0 < dropped <= len(blocked)
    assert all(ranks[task_id] is not None for task_id in chosen)


def test_select_reachable_dev_without_searcher_skips_filter() -> None:
    goals = [_plain_goal(task_id) for task_id in range(20)]
    chosen, _, _, ranks, dropped = probe.select_reachable_dev(
        list(range(20)), goals, {0: 10}, 10, seed=5, searcher=None
    )
    assert len(chosen) == 10
    assert ranks == {}
    assert dropped == 0


def test_gold_rank_returns_best_rank_across_queries() -> None:
    class _Ranker:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, query: str, k: int):
            self.calls += 1
            return [_Hit("OTHER", 1), _Hit("P0", self.calls + 1)]

    ranker = _Ranker()
    rank = probe.gold_rank(ranker, _plain_goal(0))
    assert rank == 2
    assert ranker.calls >= 2


def test_gold_rank_returns_none_when_absent() -> None:
    class _Empty:
        def search(self, query: str, k: int):
            return [_Hit("OTHER", 1)]

    assert probe.gold_rank(_Empty(), _plain_goal(0)) is None


def test_gold_rank_skips_category_only_query() -> None:
    seen: list[str] = []

    class _Recorder:
        def search(self, query: str, k: int):
            seen.append(query)
            return []

    probe.gold_rank(_Recorder(), _plain_goal(0))
    # Q3 只有品类（"收纳"），拿它判可达性等于永远返回可达 ⇒ 必须排除。
    assert "收纳" not in seen


def test_rank_profile_summarizes_ranks() -> None:
    profile = probe.rank_profile([1, 2, 3, None, 40])
    assert profile["n"] == 5
    assert profile["unreachable"] == 1
    assert profile["median"] == 3
    assert profile["max"] == 40
    assert probe.rank_profile([])["median"] is None


