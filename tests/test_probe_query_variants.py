"""`scripts/probe_query_variants.py` 的单测。

核心是两件事：简化写法要覆盖到「词太多」和「带修饰词」两种情形；
名次解析必须只认模型真的能看到的前 20 行，否则会把「第 60 名」误判成搜到了。
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_query_variants", ROOT / "scripts" / "probe_query_variants.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

GOLD = "831919364430"


def _obs(rows):
    lines = ["[SHOPPING_OBSERVATION_V2]", "page_type: search_results", ""]
    for rank, asin in rows:
        lines.append(f"{rank}|{asin}|5.0|品牌|品类|属性|标题")
    return "\n".join(lines)


def test_variants_drop_tail_and_head() -> None:
    out = probe.variants("女士单肩包 黑色 简约 高端")
    assert out["去尾词"] == "女士单肩包 黑色 简约"
    assert out["前2词"] == "女士单肩包 黑色"


def test_variants_strip_filler_words() -> None:
    out = probe.variants("不锈钢 金属 定型器 白色")
    assert out["去修饰词"] == "定型器"
    # 去掉修饰后只剩 1 个词，不该再生成「前2词」这种退化写法
    assert "去修饰+前2词" not in out


def test_variants_single_word_yields_nothing() -> None:
    assert probe.variants("炼乳") == {}


def test_rank_of_only_counts_visible_rows() -> None:
    observation = _obs([(1, "111111111111"), (3, GOLD), (60, "222222222222")])
    assert probe.rank_of(observation, GOLD) == 3


def test_rank_of_ignores_rows_beyond_top_k() -> None:
    observation = _obs([(1, "111111111111"), (35, GOLD)])
    assert probe.rank_of(observation, GOLD) is None


def test_rank_of_keeps_best_rank_across_pages() -> None:
    observation = _obs([(8, GOLD), (2, GOLD)])
    assert probe.rank_of(observation, GOLD) == 2


def test_build_queries_includes_teacher_baseline() -> None:
    queries = probe.build_queries(["炼乳 小份 10g"], ["炼乳 小包装 10g", "展艺炼乳13g"])
    assert queries["我们原句"] == "炼乳 小份 10g"
    assert queries["老师第1次"] == "炼乳 小包装 10g"
    assert "去尾词" in queries
