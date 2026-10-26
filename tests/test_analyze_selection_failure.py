"""`scripts/analyze_selection_failure.py` 的单测。

核心是**归因不能混**：买错商品和买对商品选错规格是两种病，
混在一起会得出「多开几个商品就能修」的错误结论。
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_selection_failure", ROOT / "scripts" / "analyze_selection_failure.py"
)
sf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sf)


SEARCH = """[SHOPPING_OBSERVATION_V2]
page_type: search_results
query: 木梳
1|831919364430|5.0|漫步喜|节庆|高档|结婚梳子
2|830660260530|6.0|缘来饰你|节庆|结婚|结婚梳子
3|801108777298|7.0|百万千喜|节庆|结婚|红腰带
"""


def _record(
    *, gold: str, bought: str | None, asin_match: bool | None = None, success: bool = False
) -> dict:
    detail: dict = {"reward_type": "gold_purchase" if success else "wrong_purchase"}
    if asin_match is not None:
        detail["target_asin_match"] = asin_match
    return {
        "steps": [{"observation": SEARCH}],
        "terminal_result": {
            "goal": {"asin": gold},
            "purchase": {"asin": bought} if bought else {},
            "reward_detail": detail,
        },
    }


def test_classify_wrong_product() -> None:
    record = _record(gold="831919364430", bought="801108777298")
    assert sf.classify(record) == "买错商品"


def test_classify_spec_mismatch() -> None:
    record = _record(gold="831919364430", bought="831919364430")
    assert sf.classify(record) == "买对商品·规格或属性不符"


def test_classify_prefers_reward_flag() -> None:
    record = _record(gold="831919364430", bought="831919364430", asin_match=False)
    assert sf.classify(record) == "买错商品"


def test_classify_no_purchase() -> None:
    assert sf.classify(_record(gold="831919364430", bought=None)) == "没买"


def test_collect_counts_and_top3() -> None:
    per_model = {
        "champ": {
            # gold 第 1 名，买了第 3 名 ⇒ 买错商品，gold 在前 3
            1: _record(gold="831919364430", bought="801108777298"),
            # gold 第 2 名，买了同一件但规格错 ⇒ 规格类，多开商品没用
            2: _record(gold="830660260530", bought="830660260530"),
            # 做对的题不计入失败
            3: _record(gold="831919364430", bought="831919364430", success=True),
        }
    }
    stats = sf.collect(per_model, champion="champ")
    assert stats["n_failed"] == 2
    assert stats["kinds"] == {"买错商品": 1, "买对商品·规格或属性不符": 1}
    assert stats["kinds_top3"]["买错商品"] == 1
    assert stats["bought_rank_vs_gold"] == {"买了更靠后的": 1}


def test_collect_ignores_tasks_where_gold_never_visible() -> None:
    record = _record(gold="999999999999", bought="801108777298")
    stats = sf.collect({"champ": {1: record}}, champion="champ")
    assert stats["n_failed"] == 0
