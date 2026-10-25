#!/usr/bin/env python3
"""读「训练集自测」的结果，按预登记规则判：是泛化问题，还是根本没学会。

判定规则（写死在 `build_probe_sets.py selftest` 的 metadata 里，这里只执行）
--------------------------------------------------------------------------
* 训练集做对率 **≥ 90%** ⇒ 学会了、只是泛化不过去 ⇒ 下一步加**多样性**，加量没用；
* 训练集做对率 **≤ 70%** ⇒ 连见过的题都做不对 ⇒ 下一步加**量/加 epoch**，先别指望新场景；
* 70%–90% ⇒ 灰区，**扩样本再判**，不许直接开训。

「做对」口径与评测一致：`reward_detail.purchase_success is True`
（本地三个评测目录里它恒等于 `reward_type == gold_purchase`，与 127/200 同源）。

用法::

    .venv/bin/python scripts/compare_selftest.py \\
        --run-dir outputs/evaluation/selftest-300 \\
        --probe data/probes/selftest-300 \\
        --eval-run-dir outputs/evaluation/m10r-v2
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNSOLVABLE = ROOT / "data" / "evaluation" / "unsolvable.json"

BRANCH_GENERALIZATION = 0.90
BRANCH_NOT_LEARNED = 0.70


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_success(record: Mapping[str, object]) -> bool:
    terminal = record.get("terminal_result") or {}
    detail = (terminal.get("reward_detail") or {}) if isinstance(terminal, Mapping) else {}
    if detail.get("purchase_success") is True:
        return True
    return detail.get("reward_type") == "gold_purchase"


def success_ids(records: Iterable[Mapping[str, object]]) -> dict[int, bool]:
    table: dict[int, bool] = {}
    for record in records:
        task_id = record.get("task_id")
        if task_id is None:
            continue
        table[int(task_id)] = is_success(record)
    return table


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def branch_verdict(rate: float) -> tuple[str, str]:
    if rate >= BRANCH_GENERALIZATION:
        return (
            "泛化问题",
            "模型学会了训练过的题但换题就不行 ⇒ 下一步加**多样性**（新场景/新查询/难例）。",
        )
    if rate <= BRANCH_NOT_LEARNED:
        return (
            "没学会",
            "连训练过的题都做不对 ⇒ 先加**量/加 epoch**，别急着上 GRPO 或新场景。",
        )
    return (
        "灰区",
        f"落在 {BRANCH_NOT_LEARNED:.0%}–{BRANCH_GENERALIZATION:.0%} 之间 ⇒ **扩样本再判**，不许直接开训。",
    )


def load_unsolvable(path: Path | None) -> set[int]:
    if path is None or not Path(path).exists():
        return set()
    payload = read_json(Path(path))
    ids = payload.get("task_ids") if isinstance(payload, Mapping) else None
    if ids is None:
        ids = payload if isinstance(payload, list) else []
    return {int(value) for value in ids}


def summarize_run(run_dir: Path) -> tuple[dict[int, bool], int]:
    path = Path(run_dir) / "trajectories.jsonl"
    if not path.exists():
        raise SystemExit(f"找不到轨迹文件：{path}")
    table = success_ids(read_jsonl(path))
    return table, len(table)


def format_rate(successes: int, total: int) -> str:
    if total <= 0:
        return "—"
    low, high = wilson_interval(successes, total)
    return f"{successes}/{total} = {successes / total:.1%}（95% CI {low:.1%}–{high:.1%}）"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="训练集自测的评测目录")
    parser.add_argument("--probe", type=Path, default=None, help="自测卷目录（含 metadata.json）")
    parser.add_argument("--eval-run-dir", type=Path, default=None, help="Final-200 对照目录（可选）")
    parser.add_argument("--unsolvable", type=Path, default=DEFAULT_UNSOLVABLE)
    args = parser.parse_args(argv)

    probe_ids: list[int] = []
    if args.probe:
        meta_path = Path(args.probe) / "metadata.json"
        if meta_path.exists():
            probe_ids = [int(value) for value in read_json(meta_path).get("task_ids", [])]

    table, observed = summarize_run(args.run_dir)
    expected = probe_ids or sorted(table)
    missing = [task_id for task_id in expected if task_id not in table]
    successes = sum(1 for task_id in expected if table.get(task_id))
    total = len(expected)
    rate = successes / total if total else 0.0

    print("# 训练集自测（模型做自己训练过的题）")
    print()
    print(f"- 期望题数 {total}；实际跑完 {observed}；缺失 {len(missing)} 道（按失败计入）")
    print(f"- 做对 **{format_rate(successes, total)}**")
    print()

    if args.eval_run_dir:
        eval_table, _ = summarize_run(args.eval_run_dir)
        unsolvable = load_unsolvable(args.unsolvable)
        eval_total = len(eval_table)
        eval_succ = sum(1 for value in eval_table.values() if value)
        solvable_ids = [task_id for task_id in eval_table if task_id not in unsolvable]
        solvable_succ = sum(1 for task_id in solvable_ids if eval_table[task_id])
        print("## 对照：同一模型在 Final-200 上")
        print()
        print(f"- 全部 {format_rate(eval_succ, eval_total)}")
        if unsolvable:
            print(f"- 可解口径 {format_rate(solvable_succ, len(solvable_ids))}（剔除 {len(unsolvable)} 道死题）")
        gap = rate - (eval_succ / eval_total if eval_total else 0.0)
        print(f"- 记忆落差（训练集 − Final-200）**{gap:+.1%}**")
        print()

    label, advice = branch_verdict(rate)
    print("## 预登记判定")
    print()
    print(f"**{label}** —— {advice}")
    print()
    print(f"> 阈值：≥{BRANCH_GENERALIZATION:.0%} 泛化问题 / ≤{BRANCH_NOT_LEARNED:.0%} 没学会 / 中间灰区。")
    if total and total < 200:
        print(f"> 样本只有 {total} 道，CI 宽度约 ±{(wilson_interval(successes, total)[1] - wilson_interval(successes, total)[0]) / 2:.1%}；"
              "判定落在阈值附近时请扩样本，不要靠加一次训练赌。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
