"""`scripts/eval_pairwise_selector.py` 的单测。

重点守两条：
1. **不能泄漏答案**——prompt 里只能出现 `instruction` 与商品详情，绝不能带 goal；
2. 解析不到就按判错算，不许放水。
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "eval_pairwise_selector", ROOT / "scripts" / "eval_pairwise_selector.py"
)
pw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pw)


def _step(asin: str, *, options: str = "{}") -> dict:
    return {
        "observation": (
            "[SHOPPING_OBSERVATION_V2]\n"
            "page_type: product_detail\n"
            f"asin: {asin}\n"
            f"title: 商品{asin}\n"
            f"selected_options: {options}\n"
        )
    }


def _traj(task_id: int, asins: list[str], *, instruction: str = "买一把红色的梳子") -> dict:
    return {
        "task_id": task_id,
        "initial_result": {"instruction": instruction, "goal_options": ["秘密规格"]},
        "steps": [_step(asin) for asin in asins],
    }


def test_split_identity_roundtrip() -> None:
    assert pw.split_identity("ABC|a=1&b=2") == ("ABC", {"a": "1", "b": "2"})
    assert pw.split_identity("ABC") == ("ABC", {})


def test_product_details_indexes_by_task_and_asin() -> None:
    runs = {"m1": [_traj(7, ["X1", "X2"])]}
    details = pw.product_details(runs)
    assert set(details) == {(7, "X1"), (7, "X2")}


def test_product_details_prefers_version_with_selected_options() -> None:
    row = {
        "task_id": 7,
        "initial_result": {"instruction": "x"},
        "steps": [_step("X1"), _step("X1", options='{"颜色分类": "红色"}')],
    }
    details = pw.product_details({"m1": [row]})
    assert '{"颜色分类": "红色"}' in details[(7, "X1")]


def test_instructions_never_carry_goal() -> None:
    out = pw.instructions({"m1": [_traj(7, ["X1"])]})
    assert out[7] == "买一把红色的梳子"
    assert "秘密规格" not in out[7]


def test_build_prompt_contains_instruction_and_candidates_but_no_goal() -> None:
    prompt = pw.build_prompt(
        "买一把红色的梳子",
        [
            ("A", "X1", {"颜色分类": "红色"}, "asin: X1\ntitle: 红梳子\n可点击的按钮: [a, b]"),
            ("B", "X2", {}, "asin: X2\ntitle: 木梳"),
        ],
    )
    assert "买一把红色的梳子" in prompt
    assert "【候选 A】" in prompt
    assert "title: 红梳子" in prompt
    assert "asin: X2" in prompt
    assert "本次已选规格：无" in prompt
    assert "秘密规格" not in prompt
    assert "goal" not in prompt.lower()


def test_clean_detail_drops_button_noise_and_truncates() -> None:
    text = "[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\nasin: X\n可点击的按钮: [a, b, c]"
    cleaned = pw.clean_detail(text)
    assert "可点击的按钮" not in cleaned
    assert "asin: X" in cleaned
    assert pw.clean_detail("x" * 50, max_chars=10).endswith("…（截断）")


def test_parse_choice_handles_common_forms() -> None:
    labels = ["A", "B"]
    assert pw.parse_choice("A", labels) == "A"
    assert pw.parse_choice("答案：B", labels) == "B"
    assert pw.parse_choice("候选A更好", labels) == "A"
    assert pw.parse_choice("我选 B。", labels) == "B"


def test_parse_choice_rejects_ambiguous_or_empty() -> None:
    labels = ["A", "B"]
    assert pw.parse_choice("AB", labels) is None
    assert pw.parse_choice("", labels) is None
    assert pw.parse_choice("都不行", labels) is None


def test_disputed_tasks_keeps_only_rescuable() -> None:
    headroom = {
        "diagnosis": {
            "wrong_tasks": [
                {
                    "task_id": 1,
                    "correct_in_top2": True,
                    "votes": {"G": 3, "W": 4},
                    "good_keys": ["G"],
                },
                {
                    "task_id": 2,
                    "correct_in_top2": False,
                    "votes": {"W": 9, "X": 1},
                    "good_keys": ["G"],
                },
                {
                    "task_id": 3,
                    "correct_in_top2": True,
                    "votes": {"G": 5},
                    "good_keys": ["G"],
                },
            ]
        }
    }
    picked = pw.disputed_tasks(headroom, top_k=2)
    assert [item["task_id"] for item in picked] == [1]
    assert picked[0]["candidates"] == ["W", "G"]


def test_all_tasks_includes_plurality_correct(monkeypatch) -> None:
    headroom = {
        "diagnosis": {
            "wrong_tasks": [
                {
                    "task_id": 1,
                    "correct_in_top2": True,
                    "votes": {"G": 3, "W": 4},
                    "good_keys": ["G"],
                }
            ],
            "right_tasks": [
                {"task_id": 2, "votes": {"G": 6, "W": 1}, "good_keys": ["G"]},
                {"task_id": 3, "votes": {"G": 9}, "good_keys": ["G"]},
            ],
        }
    }
    picked = pw.all_tasks(headroom, top_k=2)
    assert [item["task_id"] for item in picked] == [1, 2]
    assert picked[1]["plurality_wrong"] is False


def test_run_selector_scores_hits_and_misses(monkeypatch) -> None:
    answers = {1: "A", 2: "A", 3: "乱码"}
    monkeypatch.setattr(
        pw, "chat", lambda prompt, **kwargs: answers[int(prompt.split("task-")[1][0])]
    )
    tasks = [
        {"task_id": 1, "candidates": ["G1", "W1"], "good_keys": ["G1"]},
        {"task_id": 2, "candidates": ["W2", "G2"], "good_keys": ["G2"]},
        {"task_id": 3, "candidates": ["G3", "W3"], "good_keys": ["G3"]},
    ]
    details = {
        (1, "G1"): "d",
        (1, "W1"): "d",
        (2, "W2"): "d",
        (2, "G2"): "d",
        (3, "G3"): "d",
        (3, "W3"): "d",
    }
    results = pw.run_selector(
        tasks,
        details=details,
        instruction_by_task={1: "task-1", 2: "task-2", 3: "task-3"},
        base_url="http://x",
        api_key="k",
        model="m",
        top_k=2,
        verbose=False,
    )
    assert [item["hit"] for item in results] == [True, False, False]


def test_run_selector_skips_tasks_without_detail() -> None:
    tasks = [{"task_id": 1, "candidates": ["G1", "W1"], "good_keys": ["G1"]}]
    results = pw.run_selector(
        tasks,
        details={},
        instruction_by_task={1: "x"},
        base_url="http://x",
        api_key="k",
        model="m",
        top_k=2,
        verbose=False,
    )
    assert results == []
