"""`scripts/eval_option_selector.py` 的单测。

守两条：
1. 选项打分必须让**长的具体片段**压过短通用词（「双龙眼木板」要赢「定制」）；
2. 只在「买对了商品」的题上动手，商品选错的一律不碰。
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "eval_option_selector", ROOT / "scripts" / "eval_option_selector.py"
)
os_ = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(os_)


DETAIL = """[SHOPPING_OBSERVATION_V2]
page_type: product_detail
asin: 111111111111
title: 乒乓球拍底板
price: 200.0
available_options: {"颜色分类": ["日直拍胚9±1mm送柄块", "双龙眼木板中硬8.5mm±", "双龙眼木板中软10mm±", "设计定制"]}
"""

INSTRUCTION = "查找一款能够DIY定制的乒乓球拍底板，材质为双龙眼木板，尺寸为10毫米左右，手感偏软，价格在200元左右。"


def _record(
    *,
    gold_asin: str,
    bought_asin: str | None,
    gold_options: list[str],
    success: bool,
    observations=(DETAIL,),
) -> dict:
    return {
        "steps": [{"observation": item} for item in observations],
        "initial_result": {"instruction": INSTRUCTION},
        "terminal_result": {
            "goal": {"asin": gold_asin, "goal_options": gold_options},
            "purchase": {"asin": bought_asin} if bought_asin else {},
            "reward_detail": {"reward_type": "gold_purchase" if success else "partial_purchase"},
        },
    }


def test_available_options_parses_json() -> None:
    available = os_.available_options(
        _record(
            gold_asin="111111111111", bought_asin="111111111111", gold_options=[], success=False
        ),
        "111111111111",
    )
    assert "颜色分类" in available
    assert "双龙眼木板中软10mm±" in available["颜色分类"]


def test_available_options_empty_when_product_not_opened() -> None:
    record = _record(
        gold_asin="111111111111", bought_asin="111111111111", gold_options=[], success=False
    )
    assert os_.available_options(record, "999999999999") == {}


def test_best_value_prefers_specific_over_generic() -> None:
    values = ["双龙眼木板中软10mm±", "设计定制"]
    assert os_.best_value(INSTRUCTION, values) == "双龙眼木板中软10mm±"


def test_choose_options_picks_per_axis() -> None:
    available = {"颜色分类": ["日直拍胚9±1mm送柄块", "双龙眼木板中软10mm±", "设计定制"]}
    chosen = os_.choose_options(INSTRUCTION, available)
    assert chosen["颜色分类"] == "双龙眼木板中软10mm±"


def test_gold_satisfied_requires_all_values() -> None:
    assert os_.gold_satisfied({"颜色分类": "双龙眼木板中软10mm±"}, ["双龙眼木板中软10mm±"]) is True
    assert os_.gold_satisfied({"颜色分类": "设计定制"}, ["双龙眼木板中软10mm±"]) is False


def test_evaluate_skips_tasks_with_wrong_product() -> None:
    per_model = {
        "champ": {
            1: _record(
                gold_asin="111111111111",
                bought_asin="222222222222",
                gold_options=["双龙眼木板中软10mm±"],
                success=False,
            ),
        }
    }
    stats = os_.evaluate(per_model, champion="champ")
    assert stats["asin_ok"] == 0
    assert stats["fixed"] == 0


def test_evaluate_counts_fix_and_break() -> None:
    per_model = {
        "champ": {
            # 买对商品、规格错 ⇒ 规则能修好
            1: _record(
                gold_asin="111111111111",
                bought_asin="111111111111",
                gold_options=["双龙眼木板中软10mm±"],
                success=False,
            ),
            # 买对商品、规格本来就对 ⇒ 规则不能改错
            2: _record(
                gold_asin="111111111111",
                bought_asin="111111111111",
                gold_options=["双龙眼木板中软10mm±"],
                success=True,
            ),
        }
    }
    stats = os_.evaluate(per_model, champion="champ")
    assert stats["asin_ok"] == 2
    assert stats["original_ok"] == 1
    assert stats["rule_ok"] == 2
    assert stats["fixed"] == 1
    assert stats["broken"] == 0


def test_canon_folds_case_and_width() -> None:
    assert os_.canon("英伟达 Tesla L20 48G") == os_.canon("英伟达 tesla l20 48g")
    assert os_.canon("６．４ＣＭ") == os_.canon("6.4cm")


def test_gold_satisfied_is_case_insensitive() -> None:
    """环境判 gold 是大小写不敏感的（task 2105 实测），比较时必须折大小写。"""
    assert (
        os_.gold_satisfied({"颜色分类": "英伟达 tesla l20 48g"}, ["英伟达 Tesla L20 48G"]) is True
    )


def test_gate_keeps_original_when_it_has_support() -> None:
    """原选项在题面里有依据时不许乱换——这是 37 道「改错」的根因。"""
    available = {"颜色分类": ["浅蓝色", "浅蓝色【无线款】"]}
    instruction = "想要一款浅蓝色【无线款】的蓝牙耳机，预算200元以内"
    chosen = os_.choose_options(
        instruction, available, original={"颜色分类": "浅蓝色【无线款】"}, max_keep_score=0.0
    )
    assert chosen["颜色分类"] == "浅蓝色【无线款】"
