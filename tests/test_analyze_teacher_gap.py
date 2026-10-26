"""`scripts/analyze_teacher_gap.py` 的单测。

重点是分组不能混：A（检索没覆盖）要改检索，B（看到没点开）要改识别，
C（点开过还买错）和 D（同商品选错规格）要改选择/规格。混在一起就会得出
「多开几个候选」这种对 A 组完全无效的结论。
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_teacher_gap", ROOT / "scripts" / "analyze_teacher_gap.py"
)
gap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gap)

GOLD = "831919364430"
OTHER = "830660260530"


def _search_obs(asins, *, rank_start=1):
    lines = [
        "[SHOPPING_OBSERVATION_V2]",
        "page_type: search_results",
        "query: 木梳",
        "",
    ]
    for offset, asin in enumerate(asins):
        lines.append(f"{rank_start + offset}|{asin}|5.0|品牌|品类|属性|标题")
    return "\n".join(lines)


def _record(*, gold=GOLD, bought=None, searches=(), opens=(), steps_extra=0):
    steps = []
    for asins in searches:
        steps.append(
            {
                "tool_name": "search_products",
                "parameters": {"query": "q"},
                "observation": _search_obs(asins),
            }
        )
    for asin in opens:
        steps.append({"tool_name": "open_product", "parameters": {"asin": asin}})
    for _ in range(steps_extra):
        steps.append({"tool_name": "view_features", "parameters": {}})
    terminal = {"goal": {"asin": gold}}
    if bought:
        terminal["purchase"] = {"asin": bought}
    return {"task_id": 1, "steps": steps, "terminal_result": terminal}


def test_group_a_when_gold_never_visible() -> None:
    ours = _record(bought=OTHER, searches=[[OTHER]])
    assert gap.classify(ours, GOLD) == "A"


def test_group_b_when_visible_but_not_opened() -> None:
    ours = _record(bought=OTHER, searches=[[OTHER, GOLD]])
    assert gap.classify(ours, GOLD) == "B"


def test_group_c_when_opened_but_bought_other() -> None:
    ours = _record(bought=OTHER, searches=[[OTHER, GOLD]], opens=[GOLD])
    assert gap.classify(ours, GOLD) == "C"


def test_group_d_when_same_asin_but_failed() -> None:
    ours = _record(bought=GOLD, searches=[[GOLD]], opens=[GOLD])
    assert gap.classify(ours, GOLD) == "D"


def test_group_e_when_no_purchase() -> None:
    ours = _record(searches=[[GOLD]], opens=[GOLD])
    assert gap.classify(ours, GOLD) == "E"


def test_first_search_hit_counts_search_order() -> None:
    # 第 1 次搜索没 gold，第 2 次才有 ⇒ 返回 2，这是「换词重搜」的直接证据。
    record = _record(searches=[[OTHER], [GOLD, OTHER]])
    assert gap.first_search_hit(record, GOLD) == 2
    assert gap.first_search_hit(record, "999999999999") is None


def test_build_report_groups_and_hit_positions() -> None:
    ours_by_task = {
        1: _record(bought=OTHER, searches=[[OTHER]]),  # A
        2: _record(bought=OTHER, searches=[[GOLD, OTHER]]),  # B
    }
    teacher_by_task = {
        1: _record(searches=[[OTHER], [GOLD]]),  # 第 2 次才命中
        2: _record(searches=[[GOLD]]),
    }
    report = gap.build_report([1, 2], ours_by_task, teacher_by_task)
    assert report["tasks"] == 2
    assert report["groups"] == {"A": 1, "B": 1}
    assert report["a_teacher_hit_positions"] == {"2": 1}
    assert report["teacher"]["mean_searches"] == 1.5
    assert report["ours"]["mean_searches"] == 1.0


def test_render_markdown_lists_groups() -> None:
    report = gap.build_report(
        [1],
        {1: _record(bought=OTHER, searches=[[OTHER]])},
        {1: _record(searches=[[OTHER], [GOLD]])},
    )
    text = gap.render_markdown(report)
    assert "| A | gold 从未出现在我们的搜索结果 | 1 |" in text
    assert "第 2 次 | 1" in text
    assert "| 1 | A |" in text


def test_opened_asins_in_order_keeps_first_occurrence() -> None:
    record = _record(searches=[[GOLD, OTHER]], opens=[OTHER, GOLD, OTHER])
    assert gap.opened_asins_in_order(record) == [OTHER, GOLD]


def test_teacher_style_separates_eye_from_process() -> None:
    # 老师第一个就点开 gold、只开 1 个 ⇒ 靠「认得准」。
    direct = gap.analyse_task(1, _record(bought=OTHER, searches=[[OTHER]]), _record(searches=[[GOLD]], opens=[GOLD]))
    assert direct["teacher_open_count"] == 1
    assert direct["teacher_first_open_is_gold"] is True

    # 老师先开别的再开 gold ⇒ 靠「逐个核对」。
    probing = gap.analyse_task(
        2, _record(bought=OTHER, searches=[[OTHER]]), _record(searches=[[GOLD, OTHER]], opens=[OTHER, GOLD])
    )
    assert probing["teacher_open_count"] == 2
    assert probing["teacher_first_open_is_gold"] is False

    report = gap.build_report(
        [1, 2],
        {1: _record(bought=OTHER, searches=[[OTHER]]), 2: _record(bought=OTHER, searches=[[OTHER]])},
        {1: _record(searches=[[GOLD]], opens=[GOLD]), 2: _record(searches=[[GOLD, OTHER]], opens=[OTHER, GOLD])},
    )
    style = report["teacher_style"]["A"]
    assert style["tasks"] == 2
    assert style["only_one_open"] == 1
    assert style["first_open_is_gold"] == 1
    assert style["open_counts"] == {1: 1, 2: 1}


def test_main_writes_outputs(tmp_path) -> None:
    teacher_dir = tmp_path / "teacher"
    ours_dir = tmp_path / "ours"
    teacher_dir.mkdir()
    ours_dir.mkdir()
    (teacher_dir / "summary.json").write_text(
        json.dumps({"expected_tasks": 200, "strict_success_task_ids": [1]}), encoding="utf-8"
    )
    (ours_dir / "summary.json").write_text(
        json.dumps({"expected_tasks": 200, "strict_success_task_ids": []}), encoding="utf-8"
    )
    (teacher_dir / "trajectories.jsonl").write_text(
        json.dumps(_record(searches=[[OTHER], [GOLD]])) + "\n", encoding="utf-8"
    )
    (ours_dir / "trajectories.jsonl").write_text(
        json.dumps(_record(bought=OTHER, searches=[[OTHER]])) + "\n", encoding="utf-8"
    )
    code = gap.main(
        [
            "--teacher-summary",
            str(teacher_dir / "summary.json"),
            "--ours-summary",
            str(ours_dir / "summary.json"),
        ]
    )
    assert code == 0
    payload = json.loads((teacher_dir / "teacher-gap.json").read_text(encoding="utf-8"))
    assert payload["groups"] == {"A": 1}
